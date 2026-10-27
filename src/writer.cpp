#include "mdz/writer.hpp"

#include <optional>

#include "container.hpp"
#include "mapped_file.hpp"
#include "mdz/error.hpp"
#include "mdz/formats.hpp"
#include "mdz/trajectory.hpp"

namespace mdz {

namespace detail {
std::unique_ptr<OutputStream> open_output_at(const std::string& path, std::uint64_t offset);
}

struct Writer::Impl {
    std::unique_ptr<OutputStream> out;
    std::uint64_t pos = 0;  // file offset of the next record or the footer
    CompressOptions opts;
    std::optional<detail::Schema> schema;  // set by the first group, or by the file appended to
    std::optional<FrameBatch> pending;     // buffered frames, fewer than one group
    std::vector<detail::IndexEntry> index;
    std::uint64_t frames = 0;
    bool finished = false;
    std::optional<FrameBatch> appended;  // static metadata of the file appended to
    detail::Bytes buf;

    Impl(std::unique_ptr<OutputStream> o, CompressOptions c) : out(std::move(o)), opts(c.resolved()) {}

    // Rigid bodies of the file, fixed by its first group. Later groups reuse
    // them rather than detecting their own. If the first group has none,
    // because none were detected or the file appended to has none, the file
    // never has any, since its metadata is fixed with that group.
    Shared<std::int32_t> bodies;
    bool bodies_fixed = false;

    void encode(const FrameBatch& in) {
        if (in.nframes == 0) return;
        const FrameBatch* chunk = &in;
        FrameBatch with_bodies;
        if (opts.rigid && bodies && !in.bodies) {
            with_bodies = in;
            with_bodies.bodies = bodies;
            chunk = &with_bodies;
        }
        CompressOptions o = opts;
        if (bodies_fixed && !bodies && !in.bodies) o.rigid = false;
        CompressedTrajectory ct = compress(*chunk, o);
        if (opts.rigid && !bodies_fixed) {
            Metadata m = ct.decoded_metadata();
            if (m.bodies) bodies = share(std::move(*m.bodies));
            bodies_fixed = true;
        }
        detail::Schema s = detail::Schema::of(ct);
        if (!schema) {
            buf = detail::encode_header(s);
            out->write(buf.data(), buf.size());
            pos += buf.size();
            schema = std::move(s);
        } else {
            detail::check_same_schema(*schema, s);
        }
        std::vector<ByteSpan> blobs(ct.columns.size());
        for (std::size_t gi = 0; gi < ct.gofs.size(); ++gi) {
            for (std::size_t k = 0; k < ct.columns.size(); ++k) blobs[k] = ct.columns[k].blob(gi);
            const auto span = ct.gofs.span(gi);
            const std::uint64_t first = frames + span.start;
            index.push_back({pos, first, static_cast<std::uint32_t>(span.size())});
            buf.clear();
            detail::encode_record(first, span.size(), blobs, buf);
            out->write(buf.data(), buf.size());
            pos += buf.size();
        }
        frames += chunk->nframes;
        // Hand complete groups to the OS now, so a process that dies loses
        // only the frames not yet in a group. This is not an fsync, so an OS
        // crash can still lose recent groups.
        out->flush();
    }

    void finish() {
        if (finished) return;
        if (pending) {
            FrameBatch p = std::move(*pending);
            pending.reset();
            encode(p);
        }
        if (!schema) throw Error(ErrorKind::InvalidInput, "no frames were written, so the file has no columns");
        buf.clear();
        detail::encode_footer(pos, index, buf);
        out->write(buf.data(), buf.size());
        pos += buf.size();
        out->close();
        finished = true;
    }
};

namespace {

// Streamed batches are concatenated, so a short per-frame array would shift
// the values of every later frame onto the wrong frame.
void check_complete_per_frame(const FrameBatch& b) {
    const std::size_t n = b.nframes;
    auto short_ = [&](const std::string& what, std::size_t len) {
        throw Error(ErrorKind::InvalidInput,
                    what + " has " + std::to_string(len) + " values for a " + std::to_string(n) + "-frame batch");
    };
    for (const auto& [k, v] : b.frame_props)
        if (v.size() != n) short_(k, v.size());
    for (const auto& [k, v] : b.int_frame_props)
        if (v.size() != n) short_(k, v.size());
    if (b.box && b.box->size() != n) short_("box", b.box->size());
    if (b.origin && b.origin->size() != n) short_("origin", b.origin->size());
}

}  // namespace

Writer::Writer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Writer::Writer(Writer&&) noexcept = default;
Writer& Writer::operator=(Writer&& o) noexcept {
    if (this != &o) {
        if (impl_ && !impl_->finished) {
            try {
                close();
            } catch (...) {
            }
        }
        impl_ = std::move(o.impl_);
    }
    return *this;
}

Writer::~Writer() {
    if (impl_ && !impl_->finished && (impl_->schema || impl_->pending)) {
        try {
            impl_->finish();
        } catch (...) {
            // A destructor cannot report errors. Callers who need them call
            // close().
        }
    }
}

Writer Writer::create(const std::string& path, CompressOptions opts) {
    return create(open_output(path), std::move(opts));
}

Writer Writer::create(std::unique_ptr<OutputStream> out, CompressOptions opts) {
    opts.resolved().validate();
    return Writer(std::make_unique<Impl>(std::move(out), std::move(opts)));
}

Writer Writer::append(const std::string& path, CompressOptions opts) {
    detail::Schema schema;
    std::vector<detail::IndexEntry> index;
    LoadStatus status;
    std::uint64_t frames = 0;
    FrameBatch meta;
    {
        auto f = detail::MappedFile::open(path);
        ByteSpan b = f->bytes();
        if (b.size() < 4 || b[0] != 'M' || b[1] != 'D' || b[2] != 'Z' || b[3] != 4)
            throw Error(ErrorKind::InvalidInput, path + " is not a .mdz file, so it cannot be appended to");
        detail::Loaded L = detail::load_container(f, b);
        schema = detail::Schema::of(L.ct);
        index = std::move(L.index);
        status = L.status;
        frames = L.ct.nframes;
        meta = decompress(L.ct, 0, 0);
        // Release the mapping here, before the file is truncated.
    }
    auto impl = std::make_unique<Impl>(detail::open_output_at(path, status.data_end), std::move(opts));
    impl->pos = status.data_end;
    impl->opts.variable_atoms = schema.variable_atoms;
    if (meta.bodies) impl->bodies = meta.bodies;
    impl->bodies_fixed = true;
    impl->schema = std::move(schema);
    impl->index = std::move(index);
    impl->frames = frames;
    impl->appended = std::move(meta);
    return Writer(std::move(impl));
}

void Writer::write(const FrameBatch& batch) {
    Impl& w = *impl_;
    if (w.finished) throw Error(ErrorKind::InvalidInput, "writer already finished");
    check_complete_per_frame(batch);
    // Whether the atom count may vary is fixed by the first group. A batch
    // whose count varies, arriving before anything is written, makes it so.
    if (!w.schema && batch.variable()) w.opts.variable_atoms = true;
    if (!w.opts.variable_atoms) {
        const std::size_t have = w.schema ? w.schema->natoms : (w.pending ? w.pending->natoms : batch.natoms);
        if (batch.variable() || batch.natoms != have)
            throw Error(
                ErrorKind::InvalidInput,
                "batch has " + (batch.variable() ? std::string("a varying number of") : std::to_string(batch.natoms)) +
                    " atoms; the file has " + std::to_string(have) +
                    ". To let the atom count change, set CompressOptions::variable_atoms when creating the file");
    }
    if (w.pending) {
        w.pending->append(batch);
    } else {
        w.pending = batch;
    }
    // Encode all complete groups in one call so they are compressed in
    // parallel. The remaining frames stay buffered.
    const std::size_t g = w.opts.gof_size, n = w.pending->nframes, whole = n / g * g;
    if (whole == 0) return;
    if (whole == n) {
        FrameBatch p = std::move(*w.pending);
        w.pending.reset();
        w.encode(p);
        return;
    }
    FrameBatch rest = w.pending->slice_frames(whole, n);
    FrameBatch head = w.pending->slice_frames(0, whole);
    w.pending = std::move(rest);
    w.encode(head);
}

void Writer::flush() {
    Impl& w = *impl_;
    if (w.finished) throw Error(ErrorKind::InvalidInput, "writer already finished");
    if (w.pending) {
        FrameBatch p = std::move(*w.pending);
        w.pending.reset();
        w.encode(p);
    }
    w.out->flush();
}

void Writer::close() {
    if (impl_) impl_->finish();
}

std::uint64_t Writer::frames_written() const noexcept {
    return impl_->frames + (impl_->pending ? impl_->pending->nframes : 0);
}
std::uint64_t Writer::frames_committed() const noexcept { return impl_->frames; }
bool Writer::closed() const noexcept { return impl_->finished; }
const FrameBatch* Writer::appended_metadata() const noexcept { return impl_->appended ? &*impl_->appended : nullptr; }

}  // namespace mdz
