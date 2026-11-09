#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace zerostride::format {

/// GGUF magic bytes "GGUF" in little-endian (0x46554747).
inline constexpr std::uint32_t kGgufMagic = 0x46554747U;

/// Default GGUF data section byte alignment when `general.alignment` is not specified.
inline constexpr std::size_t kGgufDefaultAlignment = 32;

/// Maximum tensor dimensions in GGUF format.
inline constexpr std::size_t kGgufMaxDims = 4;

/// Maximum tensor name length in bytes.
inline constexpr std::size_t kGgufMaxNameLen = 64;

/// Maximum number of tensor descriptors indexed in the fixed table (zero dynamic growth).
inline constexpr std::size_t kMaxIndexedTensors = 2048;

/// GGUF metadata value type identifiers (v2 / v3 specification).
enum class GgufMetadataType : std::uint32_t {
    kUint8 = 0,
    kInt8 = 1,
    kUint16 = 2,
    kInt16 = 3,
    kUint32 = 4,
    kInt32 = 5,
    kFloat32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kUint64 = 10,
    kInt64 = 11,
    kFloat64 = 12,
};

/// GGML tensor quantization and element types.
enum class GgmlType : std::uint32_t {
    kF32 = 0,
    kF16 = 1,
    kQ4_0 = 2,
    kQ4_1 = 3,
    kQ5_0 = 6,
    kQ5_1 = 7,
    kQ8_0 = 8,
    kQ8_1 = 9,
    kQ2_K = 10,
    kQ3_K = 11,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kQ8_K = 15,
    kIQ2_XXS = 16,
    kIQ2_XS = 17,
    kIQ3_XXS = 18,
    kIQ1_S = 19,
    kIQ4_NL = 20,
    kIQ3_S = 21,
    kIQ2_S = 22,
    kIQ4_XS = 23,
    kI8 = 24,
    kI16 = 25,
    kI32 = 26,
    kI64 = 27,
    kF64 = 28,
    kBF16 = 30,
};

/// Quantization block traits for computing exact tensor byte footprints without loading ggml.
struct GgmlTypeTraits {
    std::string_view name = "UNKNOWN";
    std::uint32_t block_elements = 1; ///< Number of scalar weights per quantization block (qk)
    std::uint32_t block_bytes = 0;    ///< Size of one quantization block in bytes
};

/// Returns the quantization block geometry for a given `GgmlType`.
[[nodiscard]] constexpr GgmlTypeTraits get_ggml_type_traits(GgmlType type) noexcept {
    switch (type) {
        case GgmlType::kF32:
            return {"F32", 1, 4};
        case GgmlType::kF16:
            return {"F16", 1, 2};
        case GgmlType::kBF16:
            return {"BF16", 1, 2};
        case GgmlType::kQ4_0:
            return {"Q4_0", 32, 18};
        case GgmlType::kQ4_1:
            return {"Q4_1", 32, 20};
        case GgmlType::kQ5_0:
            return {"Q5_0", 32, 22};
        case GgmlType::kQ5_1:
            return {"Q5_1", 32, 24};
        case GgmlType::kQ8_0:
            return {"Q8_0", 32, 34};
        case GgmlType::kQ8_1:
            return {"Q8_1", 32, 36};
        case GgmlType::kQ2_K:
            return {"Q2_K", 256, 84};
        case GgmlType::kQ3_K:
            return {"Q3_K", 256, 110};
        case GgmlType::kQ4_K:
            return {"Q4_K", 256, 144};
        case GgmlType::kQ5_K:
            return {"Q5_K", 256, 176};
        case GgmlType::kQ6_K:
            return {"Q6_K", 256, 210};
        case GgmlType::kQ8_K:
            return {"Q8_K", 256, 292};
        case GgmlType::kIQ2_XXS:
            return {"IQ2_XXS", 256, 66};
        case GgmlType::kIQ2_XS:
            return {"IQ2_XS", 256, 74};
        case GgmlType::kIQ3_XXS:
            return {"IQ3_XXS", 256, 98};
        case GgmlType::kIQ1_S:
            return {"IQ1_S", 256, 50};
        case GgmlType::kIQ4_NL:
            return {"IQ4_NL", 32, 18};
        case GgmlType::kIQ3_S:
            return {"IQ3_S", 256, 110};
        case GgmlType::kIQ2_S:
            return {"IQ2_S", 256, 82};
        case GgmlType::kIQ4_XS:
            return {"IQ4_XS", 256, 136};
        case GgmlType::kI8:
            return {"I8", 1, 1};
        case GgmlType::kI16:
            return {"I16", 1, 2};
        case GgmlType::kI32:
            return {"I32", 1, 4};
        case GgmlType::kI64:
            return {"I64", 1, 8};
        case GgmlType::kF64:
            return {"F64", 1, 8};
    }
    return {"UNKNOWN", 0, 0};
}

/// Computes the exact byte size of a tensor given its dimensions and `GgmlType`.
[[nodiscard]] constexpr std::uint64_t compute_tensor_bytes(
    GgmlType type, std::uint32_t n_dims, const std::uint64_t dims[kGgufMaxDims]) noexcept {
    if (n_dims == 0 || n_dims > kGgufMaxDims) {
        return 0;
    }
    const GgmlTypeTraits traits = get_ggml_type_traits(type);
    if (traits.block_elements == 0 || traits.block_bytes == 0 ||
        (dims[0] % traits.block_elements) != 0) {
        return 0;
    }

    std::uint64_t total_elements = 1;
    for (std::uint32_t i = 0; i < n_dims; ++i) {
        total_elements *= dims[i];
    }

    return (total_elements / traits.block_elements) * traits.block_bytes;
}

/// ZeroStride storage tier classification for each tensor in a GGUF model.
enum class TensorStorageClass : std::uint8_t {
    kResidentDense = 0,       ///< Embeddings, RMSNorm, Self-Attention, Shared Experts (always in RAM)
    kResidentRouter = 1,      ///< MoE Router Gate matrices (`ffn_gate_inp`) kept in RAM for lookahead
    kStreamableMoEExpert = 2, ///< MoE Expert FFN slices streamed from SSD into Staging Buffers A/B
};

} // namespace zerostride::format
