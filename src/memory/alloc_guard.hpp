#pragma once

#include "zerostride/status.hpp"

#include <cstddef>
#include <cstdint>

namespace zerostride::memory {

/// RAII scope guard that detects dynamic heap allocations occurring on the calling thread
/// during latency-critical hot-loop sections.
class AllocGuard {
public:
    AllocGuard() noexcept;
    ~AllocGuard() noexcept;

    AllocGuard(const AllocGuard&) = delete;
    AllocGuard& operator=(const AllocGuard&) = delete;

    /// Returns the number of dynamic heap allocations detected since this guard was constructed.
    [[nodiscard]] std::size_t allocations_in_scope() const noexcept;

    /// Returns kOk if zero allocations occurred in scope, or kOutOfMemory / violation status.
    [[nodiscard]] Status verify_zero_allocations() const noexcept;

    /// Returns true if an AllocGuard is currently active on the calling thread.
    [[nodiscard]] static bool is_active_on_thread() noexcept;

    /// Increments the thread-local allocation counter (invoked by operator new hooks).
    static void notify_allocation(std::size_t bytes) noexcept;

private:
    std::size_t start_alloc_count_ = 0;
    std::size_t start_zone_blocks_ = 0;
};

} // namespace zerostride::memory
