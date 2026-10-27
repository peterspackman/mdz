// Saving and loading whole trajectories.

#include "container.hpp"
#include "file_io.hpp"
#include "mapped_file.hpp"
#include "mdz/trajectory.hpp"
#include "rigid.hpp"

namespace mdz {

namespace {

// Write the header, one record per group and the footer, in the same layout
// that Writer produces.
template <class Sink>
std::uint64_t write_all(const CompressedTrajectory& ct, Sink&& sink) {
    const detail::Schema schema = detail::Schema::of(ct);
    detail::Bytes buf = detail::encode_header(schema);
    std::uint64_t pos = buf.size();
    sink(buf);
    std::vector<detail::IndexEntry> index;
    index.reserve(ct.gofs.size());
    std::vector<ByteSpan> blobs(ct.columns.size());
    for (std::size_t gi = 0; gi < ct.gofs.size(); ++gi) {
        for (std::size_t k = 0; k < ct.columns.size(); ++k) {
            const auto& c = ct.columns[k];
            if (gi >= c.n_gofs())
                throw Error(ErrorKind::InvalidInput,
                            "column '" + c.desc.name + "' has no blob for group " + std::to_string(gi));
            blobs[k] = c.blob(gi);
        }
        const auto s = ct.gofs.span(gi);
        index.push_back({pos, s.start, static_cast<std::uint32_t>(s.size())});
        buf.clear();
        detail::encode_record(s.start, s.size(), blobs, buf);
        pos += buf.size();
        sink(buf);
    }
    buf.clear();
    detail::encode_footer(pos, index, buf);
    pos += buf.size();
    sink(buf);
    return pos;
}

CompressedTrajectory load_mapped(std::shared_ptr<const detail::MappedFile> f, LoadStatus* status) {
    detail::Loaded L = detail::load_container(f, f->bytes());
    if (status) *status = L.status;
    // Build the rigid-body model needed to decode groups coded with it.
    if (L.ct.metadata && !L.ct.metadata->empty() && !L.ct.variable_atoms) {
        Metadata m = Metadata::decode(*L.ct.metadata);
        if (m.bodies) {
            try {
                L.ct.rigid = detail::RigidModel::build(*m.bodies, L.ct.natoms);
            } catch (const Error& e) {
                throw Error(ErrorKind::Corrupt, std::string("corrupt .mdz file: ") + e.what());
            }
        }
    }
    return std::move(L.ct);
}

}  // namespace

std::uint64_t save(const std::string& path, const CompressedTrajectory& ct) {
    detail::OutFile out(path, "wb");
    std::uint64_t n = write_all(ct, [&](const detail::Bytes& b) { out.write(b.data(), b.size()); });
    out.close();
    return n;
}

std::vector<std::uint8_t> to_bytes(const CompressedTrajectory& ct) {
    std::vector<std::uint8_t> out;
    write_all(ct, [&](const detail::Bytes& b) { out.insert(out.end(), b.begin(), b.end()); });
    return out;
}

CompressedTrajectory load(const std::string& path, LoadStatus* status) {
    return load_mapped(detail::MappedFile::open(path), status);
}

CompressedTrajectory load_bytes(std::vector<std::uint8_t> bytes, LoadStatus* status) {
    return load_mapped(detail::MappedFile::from_bytes(std::move(bytes)), status);
}

}  // namespace mdz
