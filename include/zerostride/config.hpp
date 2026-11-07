#pragma once

#include <cstddef>
#include <cstdint>

namespace zerostride {

/// Default memory ceiling for constrained 8 GB systems (3.5 GiB).
inline constexpr std::size_t kDefaultMemoryCeilingBytes = 3ULL * 1024 * 1024 * 1024 + 512ULL * 1024 * 1024;

/// Maximum number of per-token latency samples recorded in the fixed telemetry buffer
/// without triggering dynamic heap allocations during the hot loop.
inline constexpr std::size_t kMaxTelemetrySamples = 8192;

/// Core configuration for the ZeroStride deterministic MoE runtime.
struct RuntimeConfig {
    const char* model_path = nullptr;
    std::size_t memory_ceiling_bytes = kDefaultMemoryCeilingBytes;
    std::uint32_t context_window = 2048;
    std::uint32_t active_experts_k = 4;
    bool enable_async_io = true;
    bool enable_router_lookahead = true;
};

} // namespace zerostride
