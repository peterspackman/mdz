#include "mdz/reader.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>

#include "bytes.hpp"
#include "codec.hpp"
#include "mapped_file.hpp"
#include "mdz/error.hpp"
#include "parallel.hpp"

namespace mdz {

namespace detail {

DecodedSpan decode_f32_group(const CompressedColumn& col, std::size_t gi, std::size_t frames, std::size_t natoms,
                             Bytes& raw, std::vector<float>& tmp, float* out, std::size_t lo, std::size_t hi,
                             const RigidModel* rigid);

struct GroupCache {
    std::mutex mutex;
    std::size_t group = SIZE_MAX;
    DecodedSpan valid{0, 0};
    std::vector<float> frames;  // room for the whole group, in stored units
};

}  // namespace detail

Reader::Reader(CompressedTrajectory ct, LoadStatus status)
    : ct_(std::move(ct)), status_(status), cache_(std::make_shared<detail::GroupCache>()) {
    length_to_nm_ = ct_.length_to_nm();
    const CompressedColumn* pos = ct_.column("positions");
    precision_ = pos ? static_cast<float>(static_cast<double>(pos->desc.precision) * length_to_nm_) : 0.0f;
    if (const CompressedColumn* box = ct_.column("box"); box && box->desc.per_frame && box->desc.width == 9) {
        DecodedColumn v = decode_column(ct_, *box);
        if (v.dtype == Dtype::F64 && v.f64.size() == ct_.nframes * 9) {
            std::vector<Box32> b(ct_.nframes);
            for (std::size_t i = 0; i < b.size(); ++i)
                for (int k = 0; k < 9; ++k) b[i][k] = static_cast<float>(v.f64[i * 9 + k] * length_to_nm_);
            boxes_ = std::move(b);
        }
    }
}

Reader Reader::open(const std::string& path) {
    LoadStatus st;
    CompressedTrajectory ct = load(path, &st);
    return Reader(std::move(ct), st);
}

Reader Reader::from_bytes(std::vector<std::uint8_t> bytes) {
    LoadStatus st;
    CompressedTrajectory ct = load_bytes(std::move(bytes), &st);
    return Reader(std::move(ct), st);
}

std::vector<std::string> Reader::column_names() const {
    std::vector<std::string> names{"positions"};
    for (const auto& c : ct_.columns)
        if (c.desc.name != "positions") names.push_back(c.desc.name);
    return names;
}

std::vector<std::size_t> Reader::frame_natoms(std::size_t first, std::size_t count) const {
    const std::size_t end = count > ct_.nframes ? ct_.nframes : std::min(ct_.nframes, first + count);
    return ct_.frame_natoms(first, end);
}

std::vector<std::int32_t> Reader::atom_ids(std::size_t frame) const {
    if (frame >= ct_.nframes)
        throw Error(ErrorKind::OutOfRange,
                    "frame " + std::to_string(frame) + " out of range (nframes=" + std::to_string(ct_.nframes) + ")");
    if (ct_.variable_atoms) {
        FrameBatch b = decompress(ct_, frame, frame + 1);
        return std::move(b.int_columns[columns::kId]);
    }
    Metadata m = ct_.decoded_metadata();
    if (m.atom_ids) return std::move(*m.atom_ids);
    std::vector<std::int32_t> ids(ct_.natoms);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int32_t>(i);
    return ids;
}

void Reader::read_positions(std::size_t first, std::size_t count, float* out) const {
    if (ct_.variable_atoms) {
        if (first > ct_.nframes || count > ct_.nframes - first)
            throw Error(ErrorKind::OutOfRange, "frames " + std::to_string(first) + ".." +
                                                   std::to_string(first + count) +
                                                   " out of range (nframes=" + std::to_string(ct_.nframes) + ")");
        DecodedColumn v = decode_column(ct_, *ct_.column("positions"), first, first + count);
        for (std::size_t i = 0; i < v.f32.size(); ++i)
            out[i] =
                length_to_nm_ == 1.0 ? v.f32[i] : static_cast<float>(static_cast<double>(v.f32[i]) * length_to_nm_);
        return;
    }
    if (count == 0 || ct_.natoms == 0) {
        if (first > ct_.nframes || count > ct_.nframes - first)
            throw Error(ErrorKind::OutOfRange, "frames " + std::to_string(first) + ".." +
                                                   std::to_string(first + count) +
                                                   " out of range (nframes=" + std::to_string(ct_.nframes) + ")");
        return;
    }
    if (first > ct_.nframes || count > ct_.nframes - first)
        throw Error(ErrorKind::OutOfRange, "frames " + std::to_string(first) + ".." + std::to_string(first + count) +
                                               " out of range (nframes=" + std::to_string(ct_.nframes) + ")");
    const CompressedColumn* col = ct_.column("positions");
    const std::size_t natoms = ct_.natoms, n3 = natoms * 3, end = first + count;
    const auto groups = ct_.gofs.overlapping(first, end);
    detail::GroupCache& cache = *cache_;
    detail::parallel_for(groups.size(), [&](std::size_t k) {
        const std::size_t gi = groups.start + k;
        const auto s = ct_.gofs.span(gi);
        const std::size_t lo = std::max(first, s.start), hi = std::min(end, s.end);
        const std::size_t rlo = lo - s.start, rhi = hi - s.start;
        thread_local detail::Bytes raw;
        thread_local std::vector<float> tmp, whole;
        float* dst = out + (lo - first) * n3;
        if (lo == s.start && hi == s.end) {
            detail::decode_f32_group(*col, gi, s.size(), natoms, raw, tmp, dst, 0, s.size(), ct_.rigid.get());
        } else {
            // Part of a group. The first request for a group decodes only
            // what it needs: the intra codec decodes single frames, and the
            // others stop after the last frame requested. A second request
            // for the same group decodes all of it into the cache, so
            // stepping through a group one frame at a time decodes it at most
            // twice.
            bool hit = false, again = false;
            {
                std::lock_guard<std::mutex> l(cache.mutex);
                if (cache.group == gi && cache.valid.lo <= rlo && cache.valid.hi >= rhi) {
                    std::memcpy(dst, cache.frames.data() + rlo * n3, (rhi - rlo) * n3 * sizeof(float));
                    hit = true;
                }
                again = cache.group == gi;
            }
            if (!hit) {
                whole.resize(s.size() * n3);
                const detail::DecodedSpan got =
                    detail::decode_f32_group(*col, gi, s.size(), natoms, raw, tmp, whole.data(), again ? 0 : rlo,
                                             again ? s.size() : rhi, ct_.rigid.get());
                std::memcpy(dst, whole.data() + rlo * n3, (rhi - rlo) * n3 * sizeof(float));
                std::lock_guard<std::mutex> l(cache.mutex);
                cache.group = gi;
                cache.valid = got;
                cache.frames.swap(whole);
            }
        }
        if (length_to_nm_ != 1.0)
            for (float* p = dst; p < dst + (hi - lo) * n3; ++p)
                *p = static_cast<float>(static_cast<double>(*p) * length_to_nm_);
    });
}

std::vector<float> Reader::read_positions(std::size_t first, std::size_t count) const {
    if (first > ct_.nframes) first = ct_.nframes;
    count = std::min(count, ct_.nframes - first);
    std::size_t atoms = count * ct_.natoms;
    if (ct_.variable_atoms) {
        atoms = 0;
        for (std::size_t c : ct_.frame_natoms(first, first + count)) atoms += c;
    }
    std::vector<float> out(atoms * 3);
    read_positions(first, count, out.data());
    return out;
}

FrameBatch Reader::read_batch(std::size_t first, std::size_t count) const {
    const std::size_t end = count > ct_.nframes ? ct_.nframes : std::min(ct_.nframes, first + count);
    return decompress(ct_, first, end);
}

DecodedColumn Reader::read_column(std::string_view name, std::size_t first, std::size_t count) const {
    const CompressedColumn* col = ct_.column(name);
    if (!col) throw Error(ErrorKind::InvalidInput, "no column named '" + std::string(name) + "'");
    const std::size_t end = count > ct_.nframes ? ct_.nframes : std::min(ct_.nframes, first + count);
    return decode_column(ct_, *col, first, end);
}

}  // namespace mdz
