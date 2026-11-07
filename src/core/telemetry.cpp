#include "zerostride/telemetry.hpp"

#include "platform/os_resources.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>

namespace zerostride {

namespace {

double percentile_from_sorted(const double* sorted_data, std::size_t n, double p) noexcept {
    if (n == 0) {
        return 0.0;
    }
    if (n == 1) {
        return sorted_data[0];
    }
    const double rank = (p / 100.0) * static_cast<double>(n - 1);
    const auto lower = static_cast<std::size_t>(rank);
    const std::size_t upper = std::min(lower + 1, n - 1);
    const double weight = rank - static_cast<double>(lower);
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

} // namespace

ResourceSnapshot ResourceSnapshot::capture() noexcept {
    return platform::query_os_resources();
}

ResourceDelta ResourceDelta::compute(const ResourceSnapshot& start,
                                     const ResourceSnapshot& end) noexcept {
    ResourceDelta delta{};
    delta.major_page_faults = end.major_page_faults - start.major_page_faults;
    delta.minor_page_faults = end.minor_page_faults - start.minor_page_faults;
    delta.current_rss_bytes = end.current_rss_bytes;
    delta.peak_rss_bytes = end.max_rss_bytes;
    delta.rss_growth_bytes = end.current_rss_bytes - start.current_rss_bytes;
    delta.voluntary_ctx_switches = end.voluntary_ctx_switches - start.voluntary_ctx_switches;
    delta.involuntary_ctx_switches = end.involuntary_ctx_switches - start.involuntary_ctx_switches;
    delta.user_cpu_ms = end.user_cpu_ms - start.user_cpu_ms;
    delta.system_cpu_ms = end.system_cpu_ms - start.system_cpu_ms;
    return delta;
}

TelemetryRecorder::TelemetryRecorder() noexcept {
    reset();
}

void TelemetryRecorder::reset() noexcept {
    start_snapshot_ = ResourceSnapshot::capture();
    step_start_ = Clock::now();
    ttft_ms_ = 0.0;
    count_ = 0;
    mean_ = 0.0;
    m2_ = 0.0;
    min_ms_ = 0.0;
    max_ms_ = 0.0;
    total_ms_ = 0.0;
    stored_samples_ = 0;
}

void TelemetryRecorder::record_ttft_ms(double ms) noexcept {
    ttft_ms_ = ms;
}

void TelemetryRecorder::begin_step() noexcept {
    step_start_ = Clock::now();
}

double TelemetryRecorder::end_step() noexcept {
    const TimePoint now = Clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(now - step_start_).count();
    record_step_ms(elapsed_ms);
    return elapsed_ms;
}

void TelemetryRecorder::record_step_ms(double ms) noexcept {
    if (count_ == 0) {
        min_ms_ = ms;
        max_ms_ = ms;
    } else {
        if (ms < min_ms_) {
            min_ms_ = ms;
        }
        if (ms > max_ms_) {
            max_ms_ = ms;
        }
    }

    ++count_;
    total_ms_ += ms;

    // Welford's online algorithm for numerical stability
    const double delta = ms - mean_;
    mean_ += delta / static_cast<double>(count_);
    const double delta2 = ms - mean_;
    m2_ += delta * delta2;

    if (stored_samples_ < kMaxTelemetrySamples) {
        samples_[stored_samples_++] = ms;
    }
}

TelemetrySummary TelemetryRecorder::summarize() noexcept {
    const ResourceSnapshot end_snapshot = ResourceSnapshot::capture();

    TelemetrySummary summary{};
    summary.token_count = count_;
    summary.ttft_ms = ttft_ms_;
    summary.total_decode_ms = total_ms_;
    summary.tokens_per_sec =
        (total_ms_ > 0.0) ? (static_cast<double>(count_) * 1000.0 / total_ms_) : 0.0;
    summary.mean_ms = mean_;
    summary.stddev_ms =
        (count_ > 1) ? std::sqrt(m2_ / static_cast<double>(count_ - 1)) : 0.0;
    summary.coeff_variation_pct =
        (mean_ > 0.0) ? (summary.stddev_ms / mean_) * 100.0 : 0.0;
    summary.min_ms = min_ms_;
    summary.max_ms = max_ms_;
    summary.resources = ResourceDelta::compute(start_snapshot_, end_snapshot);

    if (stored_samples_ > 0) {
        std::sort(samples_.begin(),
                  samples_.begin() + static_cast<std::ptrdiff_t>(stored_samples_));
        summary.p50_ms = percentile_from_sorted(samples_.data(), stored_samples_, 50.0);
        summary.p95_ms = percentile_from_sorted(samples_.data(), stored_samples_, 95.0);
        summary.p99_ms = percentile_from_sorted(samples_.data(), stored_samples_, 99.0);
    }

    return summary;
}

void TelemetryRecorder::print_table(const TelemetrySummary& s, const char* label,
                                    std::ostream& os) {
    const double current_rss_mb =
        static_cast<double>(s.resources.current_rss_bytes) / (1024.0 * 1024.0);
    const double peak_rss_mb =
        static_cast<double>(s.resources.peak_rss_bytes) / (1024.0 * 1024.0);
    const double rss_growth_mb =
        static_cast<double>(s.resources.rss_growth_bytes) / (1024.0 * 1024.0);

    os << std::fixed << std::setprecision(2);
    os << "====================================================================\n";
    os << " ZeroStride Telemetry Report: " << (label ? label : "unnamed") << "\n";
    os << "====================================================================\n";
    os << " Tokens Decoded          : " << s.token_count << "\n";
    os << " TTFT                    : " << s.ttft_ms << " ms\n";
    os << " Decode Throughput       : " << s.tokens_per_sec << " tok/s\n";
    os << "--------------------------------------------------------------------\n";
    os << " Per-Token Latency (ms)\n";
    os << "   Mean                  : " << s.mean_ms << " ms\n";
    os << "   StdDev (Jitter sigma) : " << s.stddev_ms << " ms (" << s.coeff_variation_pct
       << "% CV)\n";
    os << "   Min / P50 / P95 / P99 : " << s.min_ms << " / " << s.p50_ms << " / " << s.p95_ms
       << " / " << s.p99_ms << " ms\n";
    os << "   Max                   : " << s.max_ms << " ms\n";
    os << "--------------------------------------------------------------------\n";
    os << " OS Resource & Paging Metrics\n";
    os << "   Major Page Faults     : " << s.resources.major_page_faults
       << " (disk/swap stalls)\n";
    os << "   Minor Page Faults     : " << s.resources.minor_page_faults << "\n";
    os << "   Current RSS           : " << current_rss_mb << " MiB (" << std::showpos
       << rss_growth_mb << std::noshowpos << " MiB delta)\n";
    os << "   Peak RSS              : " << peak_rss_mb << " MiB\n";
    os << "   CPU Time (User / Sys) : " << s.resources.user_cpu_ms << " ms / "
       << s.resources.system_cpu_ms << " ms\n";
    os << "   Ctx Switches (V / IV) : " << s.resources.voluntary_ctx_switches << " / "
       << s.resources.involuntary_ctx_switches << "\n";
    os << "====================================================================\n";
}

void TelemetryRecorder::print_json(const TelemetrySummary& s, const char* label,
                                   std::ostream& os) {
    os << std::fixed << std::setprecision(4);
    os << "{\n"
       << "  \"label\": \"" << (label ? label : "unnamed") << "\",\n"
       << "  \"token_count\": " << s.token_count << ",\n"
       << "  \"ttft_ms\": " << s.ttft_ms << ",\n"
       << "  \"total_decode_ms\": " << s.total_decode_ms << ",\n"
       << "  \"tokens_per_sec\": " << s.tokens_per_sec << ",\n"
       << "  \"latency_ms\": {\n"
       << "    \"mean\": " << s.mean_ms << ",\n"
       << "    \"stddev\": " << s.stddev_ms << ",\n"
       << "    \"coeff_variation_pct\": " << s.coeff_variation_pct << ",\n"
       << "    \"min\": " << s.min_ms << ",\n"
       << "    \"p50\": " << s.p50_ms << ",\n"
       << "    \"p95\": " << s.p95_ms << ",\n"
       << "    \"p99\": " << s.p99_ms << ",\n"
       << "    \"max\": " << s.max_ms << "\n"
       << "  },\n"
       << "  \"resources\": {\n"
       << "    \"major_page_faults\": " << s.resources.major_page_faults << ",\n"
       << "    \"minor_page_faults\": " << s.resources.minor_page_faults << ",\n"
       << "    \"current_rss_bytes\": " << s.resources.current_rss_bytes << ",\n"
       << "    \"peak_rss_bytes\": " << s.resources.peak_rss_bytes << ",\n"
       << "    \"rss_growth_bytes\": " << s.resources.rss_growth_bytes << ",\n"
       << "    \"user_cpu_ms\": " << s.resources.user_cpu_ms << ",\n"
       << "    \"system_cpu_ms\": " << s.resources.system_cpu_ms << ",\n"
       << "    \"voluntary_ctx_switches\": " << s.resources.voluntary_ctx_switches << ",\n"
       << "    \"involuntary_ctx_switches\": " << s.resources.involuntary_ctx_switches << "\n"
       << "  }\n"
       << "}\n";
}

void TelemetryRecorder::print_csv_header(std::ostream& os) {
    os << "label,token_count,ttft_ms,tokens_per_sec,mean_ms,stddev_ms,cv_pct,"
          "min_ms,p50_ms,p95_ms,p99_ms,max_ms,major_faults,minor_faults,"
          "current_rss_bytes,peak_rss_bytes,rss_growth_bytes,user_cpu_ms,sys_cpu_ms\n";
}

void TelemetryRecorder::print_csv_row(const TelemetrySummary& s, const char* label,
                                      std::ostream& os) {
    os << std::fixed << std::setprecision(4)
       << (label ? label : "unnamed") << ","
       << s.token_count << ","
       << s.ttft_ms << ","
       << s.tokens_per_sec << ","
       << s.mean_ms << ","
       << s.stddev_ms << ","
       << s.coeff_variation_pct << ","
       << s.min_ms << ","
       << s.p50_ms << ","
       << s.p95_ms << ","
       << s.p99_ms << ","
       << s.max_ms << ","
       << s.resources.major_page_faults << ","
       << s.resources.minor_page_faults << ","
       << s.resources.current_rss_bytes << ","
       << s.resources.peak_rss_bytes << ","
       << s.resources.rss_growth_bytes << ","
       << s.resources.user_cpu_ms << ","
       << s.resources.system_cpu_ms << "\n";
}

} // namespace zerostride
