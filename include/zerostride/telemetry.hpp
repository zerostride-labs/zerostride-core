#pragma once

#include "zerostride/config.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace zerostride {

/// Snapshot of OS-level process resource counters obtained via getrusage() and Mach task_info.
struct ResourceSnapshot {
    std::int64_t major_page_faults = 0;        ///< Hard page faults requiring disk/swap I/O (ru_majflt)
    std::int64_t minor_page_faults = 0;        ///< Soft page faults resolved in RAM (ru_minflt)
    std::int64_t current_rss_bytes = 0;        ///< Current resident set size in bytes (Mach / procfs)
    std::int64_t max_rss_bytes = 0;            ///< Peak resident set size in bytes (ru_maxrss)
    std::int64_t voluntary_ctx_switches = 0;   ///< Context switches due to I/O wait (ru_nvcsw)
    std::int64_t involuntary_ctx_switches = 0; ///< Context switches due to preemption (ru_nivcsw)
    double user_cpu_ms = 0.0;                  ///< User CPU time in milliseconds
    double system_cpu_ms = 0.0;                ///< System/Kernel CPU time in milliseconds

    [[nodiscard]] static ResourceSnapshot capture() noexcept;
};

/// Delta between two ResourceSnapshots.
struct ResourceDelta {
    std::int64_t major_page_faults = 0;
    std::int64_t minor_page_faults = 0;
    std::int64_t current_rss_bytes = 0;
    std::int64_t peak_rss_bytes = 0;
    std::int64_t rss_growth_bytes = 0;
    std::int64_t voluntary_ctx_switches = 0;
    std::int64_t involuntary_ctx_switches = 0;
    double user_cpu_ms = 0.0;
    double system_cpu_ms = 0.0;

    [[nodiscard]] static ResourceDelta compute(const ResourceSnapshot& start,
                                               const ResourceSnapshot& end) noexcept;
};

/// Aggregated latency and OS telemetry summary for a benchmark or generation session.
struct TelemetrySummary {
    std::size_t token_count = 0;
    double ttft_ms = 0.0;             ///< Time-To-First-Token in milliseconds
    double total_decode_ms = 0.0;     ///< Total autoregressive decode time in milliseconds
    double tokens_per_sec = 0.0;      ///< Decode throughput (tokens / second)
    double mean_ms = 0.0;             ///< Mean per-token latency (ms)
    double stddev_ms = 0.0;           ///< Standard deviation (sigma) of per-token latency (ms)
    double coeff_variation_pct = 0.0; ///< Relative jitter: (stddev / mean) * 100
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
    ResourceDelta resources{};
};

/// Zero-allocation telemetry recorder for the inference hot loop.
/// Pre-allocates sample storage inline; recording steps performs zero heap or large stack allocations.
class TelemetryRecorder {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    TelemetryRecorder() noexcept;

    /// Resets all counters and captures the baseline OS resource snapshot.
    void reset() noexcept;

    /// Records the Time-To-First-Token (prefill latency) in milliseconds.
    void record_ttft_ms(double ms) noexcept;

    /// Marks the start of a single token decode step (zero allocation).
    void begin_step() noexcept;

    /// Marks the end of a single token decode step and updates online statistics (zero allocation).
    double end_step() noexcept;

    /// Directly records a token decode latency sample in milliseconds (zero allocation).
    void record_step_ms(double ms) noexcept;

    /// Computes the final summary statistics and captures the closing OS resource snapshot.
    /// Sorts the internal sample buffer in-place to avoid large stack frames or heap allocations.
    [[nodiscard]] TelemetrySummary summarize() noexcept;

    /// Number of decode steps recorded so far.
    [[nodiscard]] std::size_t step_count() const noexcept { return count_; }

    /// Formatting helpers (used outside the hot loop).
    static void print_table(const TelemetrySummary& summary, const char* label, std::ostream& os);
    static void print_json(const TelemetrySummary& summary, const char* label, std::ostream& os);
    static void print_csv_header(std::ostream& os);
    static void print_csv_row(const TelemetrySummary& summary, const char* label, std::ostream& os);

private:
    ResourceSnapshot start_snapshot_{};
    TimePoint step_start_{};
    double ttft_ms_ = 0.0;

    // Welford's online algorithm state for exact mean & variance over arbitrary token counts
    std::size_t count_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
    double min_ms_ = 0.0;
    double max_ms_ = 0.0;
    double total_ms_ = 0.0;

    // Fixed-capacity buffer for exact percentile calculation (zero heap allocation, sorted in-place)
    std::size_t stored_samples_ = 0;
    std::array<double, kMaxTelemetrySamples> samples_{};
};

} // namespace zerostride
