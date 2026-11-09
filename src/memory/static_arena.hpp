#pragma once

#include "zerostride/config.hpp"
#include "zerostride/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace zerostride::memory {

/// Default tensor alignment (128 bytes satisfies both Apple Silicon 128B cache lines and x86 AVX2/512).
inline constexpr std::size_t kDefaultTensorAlignment = 128;

/// Configuration for partitioning the single contiguous StaticArena memory block.
struct ArenaLayoutConfig {
    std::size_t persistent_weights_bytes = 0; ///< Resident dense weights + router gates
    std::size_t kv_cache_bytes = 0;           ///< Bounded KV-cache ring buffer
    std::size_t staging_buffer_bytes = 0;     ///< Size of EACH MoE expert staging buffer (A and B)
    std::size_t scratch_bytes = 0;            ///< Per-token intermediate activation scratch space
    std::size_t memory_ceiling_bytes = kDefaultMemoryCeilingBytes; ///< Hard total memory cap
    bool prefault_on_init = true;             ///< Touch all pages at init to eliminate minor faults
    bool pin_staging_buffers = false;         ///< Lock Staging Buffers A/B in physical RAM via mlock
};

/// Contiguous, pre-allocated memory arena backed by a single anonymous mmap reservation.
/// Guarantees zero OS memory allocations during the inference hot loop.
class StaticArena {
public:
    StaticArena() noexcept = default;
    ~StaticArena() noexcept;

    StaticArena(const StaticArena&) = delete;
    StaticArena& operator=(const StaticArena&) = delete;

    StaticArena(StaticArena&& other) noexcept;
    StaticArena& operator=(StaticArena&& other) noexcept;

    /// Reserves and partitions the contiguous memory block according to `cfg`.
    [[nodiscard]] Status initialize(const ArenaLayoutConfig& cfg) noexcept;

    /// Releases the underlying virtual memory reservation.
    void release() noexcept;

    /// Returns true if the arena is initialized and ready.
    [[nodiscard]] bool is_initialized() const noexcept { return base_ptr_ != nullptr; }

    /// Sub-allocates `size_bytes` from the Persistent Weights region with `alignment` (power of 2).
    [[nodiscard]] Status allocate_persistent(std::size_t size_bytes, void** out_ptr,
                                             std::size_t alignment = kDefaultTensorAlignment) noexcept;

    /// Sub-allocates `size_bytes` from the Bounded KV-Cache region with `alignment` (power of 2).
    [[nodiscard]] Status allocate_kv_cache(std::size_t size_bytes, void** out_ptr,
                                           std::size_t alignment = kDefaultTensorAlignment) noexcept;

    /// Sub-allocates `size_bytes` from the per-token Scratch region with `alignment` (power of 2).
    [[nodiscard]] Status allocate_scratch(std::size_t size_bytes, void** out_ptr,
                                          std::size_t alignment = kDefaultTensorAlignment) noexcept;

    /// Resets the per-token Scratch region bump pointer to 0 in O(1) time.
    void reset_scratch() noexcept { scratch_used_ = 0; }

    /// Returns a page-aligned span for MoE Expert Staging Buffer A ( index 0 ) or B ( index 1 ).
    [[nodiscard]] std::span<std::uint8_t> staging_buffer(std::size_t buffer_index) noexcept;
    [[nodiscard]] std::span<const std::uint8_t> staging_buffer(std::size_t buffer_index) const noexcept;

    [[nodiscard]] std::span<std::uint8_t> staging_buffer_a() noexcept { return staging_buffer(0); }
    [[nodiscard]] std::span<std::uint8_t> staging_buffer_b() noexcept { return staging_buffer(1); }

    /// Introspection getters.
    [[nodiscard]] std::size_t total_reserved_bytes() const noexcept { return total_bytes_; }
    [[nodiscard]] std::size_t persistent_capacity_bytes() const noexcept { return persistent_cap_; }
    [[nodiscard]] std::size_t persistent_used_bytes() const noexcept { return persistent_used_; }
    [[nodiscard]] std::size_t kv_cache_capacity_bytes() const noexcept { return kv_cache_cap_; }
    [[nodiscard]] std::size_t kv_cache_used_bytes() const noexcept { return kv_cache_used_; }
    [[nodiscard]] std::size_t staging_buffer_capacity_bytes() const noexcept { return staging_cap_; }
    [[nodiscard]] std::size_t scratch_capacity_bytes() const noexcept { return scratch_cap_; }
    [[nodiscard]] std::size_t scratch_used_bytes() const noexcept { return scratch_used_; }
    [[nodiscard]] bool is_staging_pinned() const noexcept { return staging_pinned_; }

private:
    std::uint8_t* base_ptr_ = nullptr;
    std::size_t total_bytes_ = 0;

    // Region base offsets from base_ptr_ and capacities (all page-aligned)
    std::size_t persistent_offset_ = 0;
    std::size_t persistent_cap_ = 0;
    std::size_t persistent_used_ = 0;

    std::size_t kv_cache_offset_ = 0;
    std::size_t kv_cache_cap_ = 0;
    std::size_t kv_cache_used_ = 0;

    std::size_t staging_a_offset_ = 0;
    std::size_t staging_b_offset_ = 0;
    std::size_t staging_cap_ = 0;
    bool staging_pinned_ = false;

    std::size_t scratch_offset_ = 0;
    std::size_t scratch_cap_ = 0;
    std::size_t scratch_used_ = 0;
};

} // namespace zerostride::memory
