#include "mdz/lazy_reader.hpp"

#include <algorithm>
#include <cstring>

#include "bytes.hpp"
#include "container.hpp"
#include "crc32.hpp"
#include "mdz/error.hpp"
#include "mdz/trajectory.hpp"

namespace mdz {

static_assert(LazyReader::kHeaderSize == detail::kHeaderSize);
static_assert(LazyReader::kTrailerSize == detail::kTrailerSize);

namespace {
// Magic, body length, and at least first_frame, nframes and the CRC.
constexpr std::uint64_t kMinRecord = 4 + 4 + 12 + 4;
}  // namespace

struct LazyReader::Impl {
    detail::Schema schema;
    // A trajectory with no groups, for the metadata, units and rigid model.
    CompressedTrajectory proto;
    std::vector<ColumnDesc> columns;
    std::uint64_t file_size = 0;
    std::uint64_t data_start = 0;
    std::uint64_t scan_pos = 0;
    bool complete = false;
    std::vector<Group> groups;
    GofTable gofs;
    double length_to_nm = 1.0;
    float precision = 0.0f;
};

LazyReader::LazyReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LazyReader::LazyReader(LazyReader&&) noexcept = default;
LazyReader& LazyReader::operator=(LazyReader&&) noexcept = default;
LazyReader::~LazyReader() = default;

std::uint64_t LazyReader::head_size(ByteSpan header) {
    if (header.size() < kHeaderSize) detail::corrupt("file is shorter than an .mdz header");
    return kHeaderSize + detail::decode_header(header.first(kHeaderSize)).schema_len;
}

LazyReader LazyReader::open(ByteSpan head, std::uint64_t file_size) {
    const std::uint64_t need = head_size(head);
    if (head.size() < need || file_size < need) detail::corrupt("file is shorter than its header and schema");
    const detail::Header h = detail::decode_header(head.first(kHeaderSize));
    auto impl = std::make_unique<Impl>();
    impl->schema = detail::decode_schema(h, head.subspan(kHeaderSize, h.schema_len));
    impl->proto = detail::assemble(impl->schema, GofTable{},
                                   std::vector<std::vector<ByteSpan>>(impl->schema.columns.size()), nullptr, nullptr);
    detail::attach_rigid_model(impl->proto);
    for (const auto& c : impl->proto.columns) impl->columns.push_back(c.desc);
    impl->file_size = file_size;
    impl->data_start = impl->scan_pos = need;
    impl->length_to_nm = impl->proto.length_to_nm();
    const CompressedColumn* pos = impl->proto.column("positions");
    impl->precision = static_cast<float>(static_cast<double>(pos->desc.precision) * impl->length_to_nm);
    return LazyReader(std::move(impl));
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> LazyReader::footer_range(ByteSpan trailer) const {
    const Impl& m = *impl_;
    if (m.file_size < m.data_start + kTrailerSize) return std::nullopt;
    const std::uint64_t tail = m.file_size - kTrailerSize;
    std::optional<std::uint64_t> off = detail::decode_trailer(trailer);
    if (!off || *off < m.data_start || *off > tail) return std::nullopt;
    return std::make_pair(*off, tail);
}

bool LazyReader::read_footer(std::uint64_t offset, ByteSpan footer) {
    Impl& m = *impl_;
    std::optional<std::vector<detail::IndexEntry>> index = detail::decode_footer(footer);
    if (!index || offset + footer.size() + kTrailerSize != m.file_size) return false;
    // The records must tile [data_start, offset) in frame order.
    std::vector<Group> groups;
    GofTable gofs;
    std::uint64_t pos = m.data_start;
    for (std::size_t i = 0; i < index->size(); ++i) {
        const detail::IndexEntry& e = (*index)[i];
        const std::uint64_t end = i + 1 < index->size() ? (*index)[i + 1].offset : offset;
        if (e.offset != pos || end < pos + kMinRecord || e.first_frame != gofs.nframes()) return false;
        if (!gofs.push(e.nframes)) return false;
        groups.push_back({e.offset, end - e.offset, e.first_frame, e.nframes});
        pos = end;
    }
    if (pos != offset) return false;
    m.groups = std::move(groups);
    m.gofs = std::move(gofs);
    m.complete = true;
    m.scan_pos = offset;
    return true;
}

std::uint64_t LazyReader::scan_offset() const noexcept { return impl_->scan_pos; }

std::uint64_t LazyReader::record_size(ByteSpan head) const {
    const Impl& m = *impl_;
    if (m.complete || head.size() < kRecordHeadSize || std::memcmp(head.data(), "GOF\x01", 4) != 0) return 0;
    const std::uint64_t size = 12 + std::uint64_t(detail::load_le<std::uint32_t>(head.data() + 4));
    if (size < kMinRecord || m.scan_pos + size > m.file_size) return 0;
    return size;
}

bool LazyReader::add_record(ByteSpan record) {
    Impl& m = *impl_;
    if (m.complete) return false;
    detail::ParsedRecord r;
    try {
        r = detail::parse_record(record, 0, record.size(), m.schema.columns.size(), m.gofs.nframes());
    } catch (const Error&) {
        return false;
    }
    if (r.total != record.size() || detail::crc32(r.body.data(), r.body.size()) != r.crc) return false;
    const std::uint64_t first = m.gofs.nframes();
    if (!m.gofs.push(r.nframes)) return false;
    m.groups.push_back({m.scan_pos, record.size(), first, r.nframes});
    m.scan_pos += record.size();
    return true;
}

std::size_t LazyReader::natoms() const noexcept { return impl_->schema.natoms; }
std::size_t LazyReader::nframes() const noexcept { return impl_->gofs.nframes(); }
bool LazyReader::variable_atoms() const noexcept { return impl_->schema.variable_atoms; }
bool LazyReader::complete() const noexcept { return impl_->complete; }
std::uint32_t LazyReader::gof_size() const noexcept { return impl_->schema.gof_size; }
const std::vector<ColumnDesc>& LazyReader::columns() const noexcept { return impl_->columns; }
Metadata LazyReader::metadata() const { return impl_->proto.decoded_metadata(); }
double LazyReader::length_to_nm() const noexcept { return impl_->length_to_nm; }
float LazyReader::precision() const noexcept { return impl_->precision; }
const std::vector<LazyReader::Group>& LazyReader::groups() const noexcept { return impl_->groups; }

std::size_t LazyReader::group_of(std::size_t frame) const {
    if (frame >= nframes())
        throw Error(ErrorKind::OutOfRange,
                    "frame " + std::to_string(frame) + " out of range (nframes=" + std::to_string(nframes()) + ")");
    return impl_->gofs.overlapping(frame, frame + 1).start;
}

FrameBatch LazyReader::decode_group(std::size_t gi, ByteSpan record, const std::vector<std::string>& columns) const {
    const Impl& m = *impl_;
    if (gi >= m.groups.size())
        throw Error(ErrorKind::OutOfRange,
                    "group " + std::to_string(gi) + " out of range (groups=" + std::to_string(m.groups.size()) + ")");
    const Group& g = m.groups[gi];
    if (record.size() != g.size)
        detail::corrupt("group " + std::to_string(gi) + " is " + std::to_string(g.size) + " bytes, given " +
                        std::to_string(record.size()));
    detail::ParsedRecord r = detail::parse_record(record, 0, record.size(), m.schema.columns.size(), g.first_frame);
    if (r.total != record.size() || r.nframes != g.nframes)
        detail::corrupt("group record " + std::to_string(gi) + " does not match the index");
    if (detail::crc32(r.body.data(), r.body.size()) != r.crc)
        detail::corrupt("group record " + std::to_string(gi) + " fails its checksum");

    // A one-group trajectory over the record, with only the columns asked for.
    detail::Schema s = m.schema;
    s.columns.clear();
    std::vector<std::vector<ByteSpan>> blobs;
    for (std::size_t k = 0; k < m.schema.columns.size(); ++k) {
        const std::string& name = m.schema.columns[k].name;
        const bool keep = columns.empty() || name == "positions" || name == kMembershipColumn ||
                          std::find(columns.begin(), columns.end(), name) != columns.end();
        if (!keep) continue;
        s.columns.push_back(m.schema.columns[k]);
        blobs.push_back({r.blobs[k]});
    }
    GofTable one;
    one.push(g.nframes);
    CompressedTrajectory ct = detail::assemble(std::move(s), std::move(one), std::move(blobs), nullptr, nullptr);
    ct.rigid = m.proto.rigid;
    return decompress(ct);
}

}  // namespace mdz
