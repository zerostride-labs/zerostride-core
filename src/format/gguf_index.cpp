#include "format/gguf_index.hpp"

#include "platform/file_io.hpp"
#include "platform/os_resources.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <ostream>

namespace zerostride::format {

namespace {

class PositionalReader {
public:
    explicit PositionalReader(const platform::FileHandle& fh) noexcept : fh_(fh) {}

    [[nodiscard]] std::uint64_t offset() const noexcept { return cursor_; }

    [[nodiscard]] Status read_bytes(void* dst, std::size_t n) noexcept {
        const Status st = fh_.pread_exact(dst, n, cursor_);
        if (st.ok()) {
            cursor_ += n;
        }
        return st;
    }

    template <typename T>
    [[nodiscard]] Status read_pod(T& out) noexcept {
        return read_bytes(&out, sizeof(T));
    }

    [[nodiscard]] Status skip_bytes(std::uint64_t n) noexcept {
        if (cursor_ > fh_.size_bytes() || n > (fh_.size_bytes() - cursor_)) {
            return Status::error(StatusCode::kCorruptedFormat,
                                 "GGUF skip_bytes exceeds file bounds");
        }
        cursor_ += n;
        return Status::success();
    }

    /// Reads a GGUF length-prefixed string into `out_buf` (truncating safely if longer than `cap - 1`).
    [[nodiscard]] Status read_string(char* out_buf, std::size_t cap,
                                     std::uint64_t* out_raw_len = nullptr) noexcept {
        std::uint64_t str_len = 0;
        Status st = read_pod(str_len);
        if (!st.ok()) {
            return st;
        }
        if (out_raw_len != nullptr) {
            *out_raw_len = str_len;
        }
        if (out_buf == nullptr || cap == 0) {
            return skip_bytes(str_len);
        }

        const std::size_t copy_len =
            (str_len < (cap - 1U)) ? static_cast<std::size_t>(str_len) : (cap - 1U);
        if (copy_len > 0) {
            st = read_bytes(out_buf, copy_len);
            if (!st.ok()) {
                return st;
            }
        }
        out_buf[copy_len] = '\0';

        const std::uint64_t remaining = str_len - copy_len;
        if (remaining > 0) {
            return skip_bytes(remaining);
        }
        return Status::success();
    }

    [[nodiscard]] Status skip_metadata_value(GgufMetadataType vtype) noexcept {
        switch (vtype) {
            case GgufMetadataType::kUint8:
            case GgufMetadataType::kInt8:
            case GgufMetadataType::kBool:
                return skip_bytes(1);
            case GgufMetadataType::kUint16:
            case GgufMetadataType::kInt16:
                return skip_bytes(2);
            case GgufMetadataType::kUint32:
            case GgufMetadataType::kInt32:
            case GgufMetadataType::kFloat32:
                return skip_bytes(4);
            case GgufMetadataType::kUint64:
            case GgufMetadataType::kInt64:
            case GgufMetadataType::kFloat64:
                return skip_bytes(8);
            case GgufMetadataType::kString:
                return read_string(nullptr, 0);
            case GgufMetadataType::kArray: {
                std::uint32_t elem_raw = 0;
                std::uint64_t count = 0;
                Status st = read_pod(elem_raw);
                if (!st.ok()) {
                    return st;
                }
                st = read_pod(count);
                if (!st.ok()) {
                    return st;
                }
                const auto elem_type = static_cast<GgufMetadataType>(elem_raw);
                for (std::uint64_t i = 0; i < count; ++i) {
                    st = skip_metadata_value(elem_type);
                    if (!st.ok()) {
                        return st;
                    }
                }
                return Status::success();
            }
        }
        return Status::error(StatusCode::kCorruptedFormat, "Unknown GGUF metadata value type");
    }

private:
    const platform::FileHandle& fh_;
    std::uint64_t cursor_ = 0;
};

[[nodiscard]] bool ends_with(std::string_view str, std::string_view suffix) noexcept {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[nodiscard]] bool contains_substr(std::string_view str, std::string_view needle) noexcept {
    return str.find(needle) != std::string_view::npos;
}

class PositionalWriter {
public:
    explicit PositionalWriter(platform::FileHandle& fh) noexcept : fh_(fh) {}

    [[nodiscard]] std::uint64_t offset() const noexcept { return cursor_; }

    [[nodiscard]] Status write_bytes(const void* src, std::size_t n) noexcept {
        const Status st = fh_.pwrite_exact(src, n, cursor_);
        if (st.ok()) {
            cursor_ += n;
        }
        return st;
    }

    template <typename T>
    [[nodiscard]] Status write_pod(const T& val) noexcept {
        return write_bytes(&val, sizeof(T));
    }

    [[nodiscard]] Status write_string(std::string_view s) noexcept {
        const std::uint64_t len = s.size();
        Status st = write_pod(len);
        if (!st.ok()) {
            return st;
        }
        return write_bytes(s.data(), s.size());
    }

    [[nodiscard]] Status write_kv_string(std::string_view key, std::string_view val) noexcept {
        Status st = write_string(key);
        if (!st.ok()) return st;
        st = write_pod(static_cast<std::uint32_t>(GgufMetadataType::kString));
        if (!st.ok()) return st;
        return write_string(val);
    }

    [[nodiscard]] Status write_kv_u32(std::string_view key, std::uint32_t val) noexcept {
        Status st = write_string(key);
        if (!st.ok()) return st;
        st = write_pod(static_cast<std::uint32_t>(GgufMetadataType::kUint32));
        if (!st.ok()) return st;
        return write_pod(val);
    }

    [[nodiscard]] Status write_tensor_desc(std::string_view name, std::uint32_t n_dims,
                                           const std::uint64_t* dims, GgmlType type,
                                           std::uint64_t rel_offset) noexcept {
        Status st = write_string(name);
        if (!st.ok()) return st;
        st = write_pod(n_dims);
        if (!st.ok()) return st;
        for (std::uint32_t i = 0; i < n_dims; ++i) {
            st = write_pod(dims[i]);
            if (!st.ok()) return st;
        }
        st = write_pod(static_cast<std::uint32_t>(type));
        if (!st.ok()) return st;
        return write_pod(rel_offset);
    }

    [[nodiscard]] Status pad_to_alignment(std::size_t alignment) noexcept {
        const std::uint64_t aligned =
            platform::align_up(static_cast<std::size_t>(cursor_), alignment);
        const std::uint8_t zero = 0;
        while (cursor_ < aligned) {
            const Status st = write_pod(zero);
            if (!st.ok()) return st;
        }
        return Status::success();
    }

private:
    platform::FileHandle& fh_;
    std::uint64_t cursor_ = 0;
};

void format_int(char* buf, std::size_t cap, std::uint32_t val, std::string_view prefix,
                std::string_view suffix) noexcept {
    std::size_t pos = 0;
    auto append_sv = [&](std::string_view sv) {
        for (char c : sv) {
            if (pos + 1 < cap) {
                buf[pos++] = c;
            }
        }
    };
    append_sv(prefix);
    char digits[16]{};
    std::size_t d_count = 0;
    std::uint32_t tmp = val;
    do {
        digits[d_count++] = static_cast<char>('0' + (tmp % 10U));
        tmp /= 10U;
    } while (tmp > 0 && d_count < sizeof(digits));
    while (d_count > 0) {
        if (pos + 1 < cap) {
            buf[pos++] = digits[--d_count];
        }
    }
    append_sv(suffix);
    buf[pos] = '\0';
}

} // namespace

TensorStorageClass GgufIndex::classify_tensor(std::string_view name,
                                              std::uint32_t n_dims) noexcept {
    // Router gating matrices (`ffn_gate_inp`) must remain resident for O(1) L+1 lookahead
    if (contains_substr(name, "ffn_gate_inp")) {
        return TensorStorageClass::kResidentRouter;
    }

    // Packed 3D expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) or `.experts.`
    if (contains_substr(name, "_exps.") || contains_substr(name, ".experts.") ||
        (n_dims == 3 && contains_substr(name, "ffn_"))) {
        return TensorStorageClass::kStreamableMoEExpert;
    }

    return TensorStorageClass::kResidentDense;
}

std::int32_t GgufIndex::parse_layer_index(std::string_view name) noexcept {
    constexpr std::string_view kPrefix = "blk.";
    if (name.size() <= kPrefix.size() || name.compare(0, kPrefix.size(), kPrefix) != 0) {
        return -1;
    }
    std::int32_t layer = 0;
    std::size_t i = kPrefix.size();
    bool has_digits = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        layer = layer * 10 + static_cast<std::int32_t>(name[i] - '0');
        has_digits = true;
        ++i;
    }
    return (has_digits && i < name.size() && name[i] == '.') ? layer : -1;
}

Status GgufIndex::index_file(const char* path) noexcept {
    meta_ = ModelMetadata{};
    total_file_bytes_ = 0;
    tensor_count_ = 0;

    platform::FileHandle fh;
    Status st = fh.open_readonly(path, false);
    if (!st.ok()) {
        return st;
    }
    total_file_bytes_ = fh.size_bytes();

    PositionalReader reader(fh);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t n_tensors = 0;
    std::uint64_t n_kv = 0;

    if (!reader.read_pod(magic).ok() || magic != kGgufMagic) {
        return Status::error(StatusCode::kCorruptedFormat, "Invalid GGUF magic header");
    }
    if (!reader.read_pod(version).ok() || (version != 2 && version != 3)) {
        return Status::error(StatusCode::kUnsupportedFeature,
                             "Unsupported GGUF version (expected v2 or v3)");
    }
    if (!reader.read_pod(n_tensors).ok() || !reader.read_pod(n_kv).ok()) {
        return Status::error(StatusCode::kCorruptedFormat, "Truncated GGUF preamble");
    }
    if (n_tensors > kMaxIndexedTensors) {
        return Status::error(StatusCode::kArenaExhausted,
                             "GGUF tensor count exceeds kMaxIndexedTensors");
    }

    meta_.gguf_version = version;

    // Stream through metadata KV entries, extracting architectural parameters
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        char key_buf[128]{};
        st = reader.read_string(key_buf, sizeof(key_buf));
        if (!st.ok()) return st;

        std::uint32_t raw_vtype = 0;
        st = reader.read_pod(raw_vtype);
        if (!st.ok()) return st;

        const auto vtype = static_cast<GgufMetadataType>(raw_vtype);
        const std::string_view key(key_buf);

        if (key == "general.architecture" && vtype == GgufMetadataType::kString) {
            st = reader.read_string(meta_.architecture, sizeof(meta_.architecture));
        } else if (key == "general.alignment" && vtype == GgufMetadataType::kUint32) {
            std::uint32_t align_val = 0;
            st = reader.read_pod(align_val);
            if (st.ok() && align_val > 0) {
                meta_.data_alignment = align_val;
            }
        } else if (ends_with(key, ".block_count") && vtype == GgufMetadataType::kUint32) {
            st = reader.read_pod(meta_.block_count);
        } else if (ends_with(key, ".expert_count") && vtype == GgufMetadataType::kUint32) {
            st = reader.read_pod(meta_.expert_count);
        } else if (ends_with(key, ".expert_used_count") && vtype == GgufMetadataType::kUint32) {
            st = reader.read_pod(meta_.expert_used_count);
        } else if (ends_with(key, ".embedding_length") && vtype == GgufMetadataType::kUint32) {
            st = reader.read_pod(meta_.embedding_length);
        } else {
            st = reader.skip_metadata_value(vtype);
        }

        if (!st.ok()) return st;
    }

    // Parse tensor descriptor table
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        TensorDescriptor desc{};
        st = reader.read_string(desc.name, sizeof(desc.name));
        if (!st.ok()) return st;

        st = reader.read_pod(desc.n_dims);
        if (!st.ok() || desc.n_dims == 0 || desc.n_dims > kGgufMaxDims) {
            return Status::error(StatusCode::kCorruptedFormat, "Invalid tensor n_dims in GGUF");
        }

        for (std::uint32_t d = 0; d < desc.n_dims; ++d) {
            st = reader.read_pod(desc.dims[d]);
            if (!st.ok()) return st;
        }

        std::uint32_t raw_type = 0;
        st = reader.read_pod(raw_type);
        if (!st.ok()) return st;
        desc.type = static_cast<GgmlType>(raw_type);

        st = reader.read_pod(desc.relative_offset);
        if (!st.ok()) return st;

        desc.size_bytes = compute_tensor_bytes(desc.type, desc.n_dims, desc.dims);
        if (desc.size_bytes == 0) {
            return Status::error(StatusCode::kCorruptedFormat,
                                 "Invalid tensor dimensions or unsupported GgmlType");
        }

        desc.storage_class = classify_tensor(desc.name_view(), desc.n_dims);
        desc.layer_index = parse_layer_index(desc.name_view());

        if (desc.storage_class == TensorStorageClass::kStreamableMoEExpert && desc.n_dims == 3 &&
            desc.dims[2] > 0) {
            desc.expert_count = static_cast<std::uint32_t>(desc.dims[2]);
            desc.expert_slice_bytes = desc.size_bytes / desc.expert_count;
        } else {
            desc.expert_count = 1;
            desc.expert_slice_bytes = desc.size_bytes;
        }

        tensors_[tensor_count_++] = desc;
    }

    meta_.data_section_offset =
        platform::align_up(static_cast<std::size_t>(reader.offset()), meta_.data_alignment);

    // Resolve absolute file offsets and validate against total file size
    for (std::size_t i = 0; i < tensor_count_; ++i) {
        TensorDescriptor& t = tensors_[i];
        t.file_offset = meta_.data_section_offset + t.relative_offset;
        if (t.file_offset > total_file_bytes_ ||
            t.size_bytes > (total_file_bytes_ - t.file_offset)) {
            return Status::error(StatusCode::kCorruptedFormat,
                                 "Tensor byte range exceeds physical GGUF file size");
        }
    }

    return Status::success();
}

WorkingSetReport GgufIndex::compute_working_set(
    std::uint32_t active_experts_k_override) const noexcept {
    WorkingSetReport rep{};
    rep.total_file_bytes = total_file_bytes_;

    const std::uint32_t active_k = (active_experts_k_override > 0)
                                       ? active_experts_k_override
                                       : ((meta_.expert_used_count > 0) ? meta_.expert_used_count : 4U);

    // Track per-layer active expert requirement to find the worst-case layer
    std::uint64_t layer0_single_expert_bytes = 0;
    std::int32_t first_moe_layer = -1;

    for (std::size_t i = 0; i < tensor_count_; ++i) {
        const TensorDescriptor& t = tensors_[i];
        switch (t.storage_class) {
            case TensorStorageClass::kResidentDense:
                rep.resident_dense_bytes += t.size_bytes;
                ++rep.resident_tensor_count;
                break;
            case TensorStorageClass::kResidentRouter:
                rep.resident_router_bytes += t.size_bytes;
                ++rep.router_tensor_count;
                break;
            case TensorStorageClass::kStreamableMoEExpert:
                rep.streamable_expert_bytes += t.size_bytes;
                ++rep.expert_tensor_count;
                if (first_moe_layer < 0) {
                    first_moe_layer = t.layer_index;
                }
                if (t.layer_index == first_moe_layer) {
                    layer0_single_expert_bytes += t.expert_slice_bytes;
                }
                break;
        }
    }

    rep.single_layer_top_k_bytes =
        layer0_single_expert_bytes * static_cast<std::uint64_t>(active_k);
    rep.total_resident_Arena_bytes = rep.resident_dense_bytes + rep.resident_router_bytes +
                                     (2ULL * rep.single_layer_top_k_bytes);
    return rep;
}

const TensorDescriptor* GgufIndex::find_tensor(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < tensor_count_; ++i) {
        if (tensors_[i].name_view() == name) {
            return &tensors_[i];
        }
    }
    return nullptr;
}

void GgufIndex::print_summary(std::ostream& os, std::size_t memory_ceiling_bytes,
                              std::uint32_t active_experts_k_override) const {
    const WorkingSetReport rep = compute_working_set(active_experts_k_override);
    const auto to_mib = [](std::uint64_t b) -> double {
        return static_cast<double>(b) / (1024.0 * 1024.0);
    };

    const std::uint32_t active_k = (active_experts_k_override > 0)
                                       ? active_experts_k_override
                                       : ((meta_.expert_used_count > 0) ? meta_.expert_used_count : 4U);

    const double savings_pct =
        (rep.total_file_bytes > 0)
            ? (1.0 - static_cast<double>(rep.total_resident_Arena_bytes) /
                         static_cast<double>(rep.total_file_bytes)) *
                  100.0
            : 0.0;

    os << std::fixed << std::setprecision(2);
    os << "====================================================================\n";
    os << " ZeroStride GGUF Zero-Load Indexer Report\n";
    os << "====================================================================\n";
    os << " Architecture            : " << meta_.architecture << " (GGUF v" << meta_.gguf_version << ")\n";
    os << " Layers / Experts (K/E)  : " << meta_.block_count << " layers | top-" << active_k
       << " of " << meta_.expert_count << " experts\n";
    os << " Hidden Dimension        : " << meta_.embedding_length << "\n";
    os << " Data Section Offset     : " << meta_.data_section_offset << " bytes (align "
       << meta_.data_alignment << "B)\n";
    os << " Total Indexed Tensors   : " << tensor_count_ << " (" << rep.resident_tensor_count
       << " dense, " << rep.router_tensor_count << " routers, " << rep.expert_tensor_count
       << " MoE packed)\n";
    os << "--------------------------------------------------------------------\n";
    os << " Memory Footprint & StaticArena Partition Breakdown\n";
    os << "   Total GGUF File Size  : " << to_mib(rep.total_file_bytes) << " MiB\n";
    os << "   Resident Dense        : " << to_mib(rep.resident_dense_bytes) << " MiB\n";
    os << "   Resident Routers      : " << to_mib(rep.resident_router_bytes) << " MiB\n";
    os << "   Streamable MoE (SSD)  : " << to_mib(rep.streamable_expert_bytes) << " MiB\n";
    os << "   Staging Buffer (each) : " << to_mib(rep.single_layer_top_k_bytes) << " MiB (top-"
       << active_k << " experts/layer)\n";
    os << "   Required StaticArena  : " << to_mib(rep.total_resident_Arena_bytes) << " MiB ("
       << savings_pct << "% RAM reduction vs full load)\n";
    os << "   Configured Ceiling    : " << to_mib(memory_ceiling_bytes) << " MiB -> "
       << ((rep.total_resident_Arena_bytes <= memory_ceiling_bytes) ? "[FITS IN CEILING]"
                                                                    : "[EXCEEDS CEILING]")
       << "\n";
    os << "====================================================================\n";
}

Status GgufIndex::write_synthetic_moe_gguf(const char* path,
                                           const SyntheticMoESpec& spec) noexcept {
    platform::FileHandle fh;
    Status st = fh.open_write_trunc(path);
    if (!st.ok()) return st;

    PositionalWriter writer(fh);

    // 1 Global tensor (`token_embd.weight`) + per-layer:
    //   1 attention (`attn_qkv.weight`), 1 router (`ffn_gate_inp.weight`),
    //   and 3 packed 3D expert tensors (`ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`)
    const std::uint64_t total_tensors = 1ULL + static_cast<std::uint64_t>(spec.num_layers) * 5ULL;
    const std::uint64_t total_kv = 6ULL;

    if (!writer.write_pod(kGgufMagic).ok() ||
        !writer.write_pod(static_cast<std::uint32_t>(3)).ok() ||
        !writer.write_pod(total_tensors).ok() || !writer.write_pod(total_kv).ok()) {
        return Status::error(StatusCode::kIoError, "Failed to write GGUF header");
    }

    if (!writer.write_kv_string("general.architecture", "qwen2moe").ok() ||
        !writer.write_kv_u32("general.alignment", static_cast<std::uint32_t>(kGgufDefaultAlignment)).ok() ||
        !writer.write_kv_u32("qwen2moe.block_count", spec.num_layers).ok() ||
        !writer.write_kv_u32("qwen2moe.expert_count", spec.num_experts).ok() ||
        !writer.write_kv_u32("qwen2moe.expert_used_count", spec.active_experts_k).ok() ||
        !writer.write_kv_u32("qwen2moe.embedding_length", spec.hidden_dim).ok()) {
        return Status::error(StatusCode::kIoError, "Failed to write GGUF KV metadata");
    }

    std::uint64_t rel_offset = 0;
    auto advance_rel_offset = [&](std::uint64_t tensor_bytes) {
        rel_offset = platform::align_up(static_cast<std::size_t>(rel_offset + tensor_bytes),
                                        kGgufDefaultAlignment);
    };

    // Tensor 0: token_embd.weight [hidden_dim, 1024] in Q8_0
    const std::uint64_t embd_dims[4] = {spec.hidden_dim, 1024, 1, 1};
    const std::uint64_t embd_bytes = compute_tensor_bytes(GgmlType::kQ8_0, 2, embd_dims);
    st = writer.write_tensor_desc("token_embd.weight", 2, embd_dims, GgmlType::kQ8_0, rel_offset);
    if (!st.ok()) return st;
    advance_rel_offset(embd_bytes);

    for (std::uint32_t l = 0; l < spec.num_layers; ++l) {
        char name_buf[64]{};

        // Resident Dense: blk.<l>.attn_qkv.weight [hidden_dim, hidden_dim * 3] in Q8_0
        format_int(name_buf, sizeof(name_buf), l, "blk.", ".attn_qkv.weight");
        const std::uint64_t attn_dims[4] = {spec.hidden_dim, spec.hidden_dim * 3ULL, 1, 1};
        const std::uint64_t attn_bytes = compute_tensor_bytes(GgmlType::kQ8_0, 2, attn_dims);
        st = writer.write_tensor_desc(name_buf, 2, attn_dims, GgmlType::kQ8_0, rel_offset);
        if (!st.ok()) return st;
        advance_rel_offset(attn_bytes);

        // Resident Router: blk.<l>.ffn_gate_inp.weight [hidden_dim, num_experts] in F32
        format_int(name_buf, sizeof(name_buf), l, "blk.", ".ffn_gate_inp.weight");
        const std::uint64_t router_dims[4] = {spec.hidden_dim, spec.num_experts, 1, 1};
        const std::uint64_t router_bytes = compute_tensor_bytes(GgmlType::kF32, 2, router_dims);
        st = writer.write_tensor_desc(name_buf, 2, router_dims, GgmlType::kF32, rel_offset);
        if (!st.ok()) return st;
        advance_rel_offset(router_bytes);

        // Streamable MoE 3D packed tensors [hidden_dim, ffn_dim, num_experts] in Q4_K
        const std::uint64_t exp_dims[4] = {spec.hidden_dim, spec.ffn_dim, spec.num_experts, 1};
        const std::uint64_t exp_bytes = compute_tensor_bytes(GgmlType::kQ4_K, 3, exp_dims);

        format_int(name_buf, sizeof(name_buf), l, "blk.", ".ffn_gate_exps.weight");
        st = writer.write_tensor_desc(name_buf, 3, exp_dims, GgmlType::kQ4_K, rel_offset);
        if (!st.ok()) return st;
        advance_rel_offset(exp_bytes);

        format_int(name_buf, sizeof(name_buf), l, "blk.", ".ffn_up_exps.weight");
        st = writer.write_tensor_desc(name_buf, 3, exp_dims, GgmlType::kQ4_K, rel_offset);
        if (!st.ok()) return st;
        advance_rel_offset(exp_bytes);

        format_int(name_buf, sizeof(name_buf), l, "blk.", ".ffn_down_exps.weight");
        st = writer.write_tensor_desc(name_buf, 3, exp_dims, GgmlType::kQ4_K, rel_offset);
        if (!st.ok()) return st;
        advance_rel_offset(exp_bytes);
    }

    st = writer.pad_to_alignment(kGgufDefaultAlignment);
    if (!st.ok()) return st;

    // Extend the file to the exact end of the data section by writing a trailing marker byte
    if (rel_offset > 0) {
        const std::uint8_t marker = 0x5A;
        st = fh.pwrite_exact(&marker, 1, writer.offset() + rel_offset - 1);
        if (!st.ok()) return st;
    }

    return Status::success();
}

} // namespace zerostride::format
