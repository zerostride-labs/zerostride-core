#pragma once

#include "zerostride/status.hpp"
#include "zerostride/telemetry.hpp"

#include <cstddef>

namespace zerostride::platform {

/// Captures OS-level process resource counters (page faults, RSS, CPU time, context switches).
[[nodiscard]] ResourceSnapshot query_os_resources() noexcept;

/// Queries the operating system virtual memory page size in bytes (e.g., 4096 or 16384).
[[nodiscard]] std::size_t system_page_size() noexcept;

/// Rounds `size_bytes` up to the nearest multiple of `alignment` (must be a power of two).
[[nodiscard]] constexpr std::size_t align_up(std::size_t size_bytes,
                                             std::size_t alignment) noexcept {
    return (size_bytes + alignment - 1U) & ~(alignment - 1U);
}

/// Maps a contiguous anonymous virtual memory region of `size_bytes`.
[[nodiscard]] Status map_anonymous_pages(std::size_t size_bytes, void** out_ptr) noexcept;

/// Releases a previously mapped virtual memory region of `size_bytes`.
[[nodiscard]] Status unmap_pages(void* ptr, std::size_t size_bytes) noexcept;

/// Deterministically touches every page in `[ptr, ptr + size_bytes)` to fault it into physical RAM
/// during initialization, preventing minor page faults during the inference hot loop.
void prefault_pages(void* ptr, std::size_t size_bytes) noexcept;

/// Pins `[ptr, ptr + size_bytes)` in physical RAM via POSIX `mlock` so the OS kernel cannot swap
/// these pages out to disk (used for MoE Staging Buffers A & B).
[[nodiscard]] Status lock_pages(void* ptr, std::size_t size_bytes) noexcept;

/// Unpins previously locked pages via POSIX `munlock`.
[[nodiscard]] Status unlock_pages(void* ptr, std::size_t size_bytes) noexcept;

} // namespace zerostride::platform
