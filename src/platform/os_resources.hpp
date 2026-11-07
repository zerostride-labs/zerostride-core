#pragma once

#include "zerostride/status.hpp"
#include "zerostride/telemetry.hpp"

#include <cstddef>

namespace zerostride::platform {

/// Captures OS-level process resource counters (page faults, RSS, CPU time, context switches).
[[nodiscard]] ResourceSnapshot query_os_resources() noexcept;

/// Queries the operating system virtual memory page size in bytes (e.g., 4096 or 16384).
[[nodiscard]] std::size_t system_page_size() noexcept;

/// Maps a contiguous anonymous virtual memory region of `size_bytes`.
[[nodiscard]] Status map_anonymous_pages(std::size_t size_bytes, void** out_ptr) noexcept;

/// Releases a previously mapped virtual memory region of `size_bytes`.
[[nodiscard]] Status unmap_pages(void* ptr, std::size_t size_bytes) noexcept;

} // namespace zerostride::platform
