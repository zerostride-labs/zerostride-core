#pragma once

#include "format/gguf_types.hpp"
#include "zerostride/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string_view>

namespace zerostride::format {

/// Indexed metadata and exact disk offset for a single GGUF tensor.
struct TensorDescriptor {
    char name[kGgufMaxNameLen + 1]{};
    std::uint32_t n_dims = 0;
    std::uint64_t dims[kGgufMaxDims]{1, 1, 1, 1};
    GgmlType type = GgmlType::kF32;
    TensorStorageClass storage_class = TensorStorageClass::kResidentDense;
    std::int32_t layer_index = -1;            ///< Layer index (0..L-1) or -1 for global tensors
    std::uint64_t relative_offset = 0;        ///< Offset relative to GGUF data section
    std::uint64_t file_offset = 0;            ///< Absolute byte offset from start of GGUF file
    std::uint64_t size_bytes = 0;             ///< Total tensor size in bytes
    std::uint32_t expert_count = 1;           ///< Number of experts packed in this tensor (dims[2])
    std::uint64_t expert_slice_bytes = 0;     ///< Byte size of a single expert slice

    [[nodiscard]] std::string_view name_view() const noexcept { return std::string_view(name); }

    /// Computes the exact disk file offset for `expert_id` within a packed 3D expert tensor.
    [[nodiscard]] constexpr std::uint64_t expert_file_offset(
        std::uint32_t expert_id) const noexcept {
        return file_offset + static_cast<std::uint64_t>(expert_id) * expert_slice_bytes;
    }
};

/// Architectural metadata extracted from the GGUF KV header.
struct ModelMetadata {
    char architecture[32]{"unknown"};
    std::uint32_t gguf_version = 0;
    std::size_t data_alignment = kGgufDefaultAlignment;
    std::uint64_t data_section_offset = 0;
    std::uint32_t block_count = 0;        ///< Number of transformer layers (L)
    std::uint32_t expert_count = 0;       ///< Total experts per MoE layer (E)
    std::uint32_t expert_used_count = 0;  ///< Active top-K experts per token (K)
    std::uint32_t embedding_length = 0;   ///< Hidden dimension (d_model)
};

/// Calculated RAM working-set breakdown for `StaticArena` provisioning.
struct WorkingSetReport {
    std::uint64_t total_file_bytes = 0;
    std::uint64_t resident_dense_bytes = 0;
    std::uint64_t resident_router_bytes = 0;
    std::uint64_t streamable_expert_bytes = 0;
    std::uint64_t single_layer_top_k_bytes = 0; ///< Bytes needed in ONE staging buffer for top-K experts
    std::uint64_t total_resident_Arena_bytes = 0; ///< Dense + Router + 2 * Staging Buffer (A & B)
    std::size_t resident_tensor_count = 0;
    std::size_t router_tensor_count = 0;
    std::size_t expert_tensor_count = 0;
};

/// Parameters for generating a spec-compliant synthetic Qwen-MoE GGUF v3 file for testing/CLI.
struct SyntheticMoESpec {
    std::uint32_t num_layers = 4;
    std::uint32_t num_experts = 16;
    std::uint32_t active_experts_k = 4;
    std::uint32_t hidden_dim = 256;
    std::uint32_t ffn_dim = 512;
};

/// Zero-load GGUF offset indexer. Parses metadata and tensor file offsets via positional `pread`
/// without mapping or loading model weights into physical RAM.
class GgufIndex {
public:
    GgufIndex() noexcept = default;

    /// Parses the GGUF header, metadata, and tensor table from `path` using positional reads.
    [[nodiscard]] Status index_file(const char* path) noexcept;

    /// Computes the RAM working-set budget for a given `active_experts_k` override (0 = use model default).
    [[nodiscard]] WorkingSetReport compute_working_set(
        std::uint32_t active_experts_k_override = 0) const noexcept;

    /// Prints a human-readable summary of the indexed model and StaticArena memory budget.
    void print_summary(std::ostream& os, std::size_t memory_ceiling_bytes,
                       std::uint32_t active_experts_k_override = 0) const;

    [[nodiscard]] const ModelMetadata& metadata() const noexcept { return meta_; }

    [[nodiscard]] std::span<const TensorDescriptor> tensors() const noexcept {
        return std::span<const TensorDescriptor>(tensors_.data(), tensor_count_);
    }

    /// Looks up a tensor descriptor by exact name, returning nullptr if not found.
    [[nodiscard]] const TensorDescriptor* find_tensor(std::string_view name) const noexcept;

    /// Classifies a tensor name and shape into ResidentDense, ResidentRouter, or StreamableMoEExpert.
    [[nodiscard]] static TensorStorageClass classify_tensor(std::string_view name,
                                                            std::uint32_t n_dims) noexcept;

    /// Parses the layer index `L` from a GGUF tensor name `blk.<L>.*`, or returns -1.
    [[nodiscard]] static std::int32_t parse_layer_index(std::string_view name) noexcept;

    /// Generates a valid, spec-compliant GGUF v3 Qwen-MoE file on disk at `path`.
    [[nodiscard]] static Status write_synthetic_moe_gguf(const char* path,
                                                         const SyntheticMoESpec& spec) noexcept;

private:
    ModelMetadata meta_{};
    std::uint64_t total_file_bytes_ = 0;
    std::size_t tensor_count_ = 0;
    std::array<TensorDescriptor, kMaxIndexedTensors> tensors_{};
};

} // namespace zerostride::format
