#include "platform/os_resources.hpp"
#include "zerostride/status.hpp"
#include "zerostride/telemetry.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

namespace {

bool approx_equal(double a, double b, double eps = 1e-4) {
    return std::fabs(a - b) <= eps;
}

int test_statistics_calculation() {
    zerostride::TelemetryRecorder rec;
    rec.reset();
    rec.record_ttft_ms(42.5);

    // Sample dataset: {10.0, 20.0, 30.0, 40.0, 50.0}
    // Mean = 30.0
    // Sample variance = ((400 + 100 + 0 + 100 + 400) / 4) = 250.0 -> StdDev = sqrt(250) = 15.8113883
    rec.record_step_ms(10.0);
    rec.record_step_ms(20.0);
    rec.record_step_ms(30.0);
    rec.record_step_ms(40.0);
    rec.record_step_ms(50.0);

    const auto summary = rec.summarize();

    if (summary.token_count != 5) {
        std::cerr << "[FAIL] Expected token_count == 5, got " << summary.token_count << "\n";
        return 1;
    }
    if (!approx_equal(summary.ttft_ms, 42.5)) {
        std::cerr << "[FAIL] Expected ttft_ms == 42.5, got " << summary.ttft_ms << "\n";
        return 1;
    }
    if (!approx_equal(summary.mean_ms, 30.0)) {
        std::cerr << "[FAIL] Expected mean_ms == 30.0, got " << summary.mean_ms << "\n";
        return 1;
    }
    if (!approx_equal(summary.stddev_ms, std::sqrt(250.0))) {
        std::cerr << "[FAIL] Expected stddev_ms == " << std::sqrt(250.0) << ", got "
                  << summary.stddev_ms << "\n";
        return 1;
    }
    if (!approx_equal(summary.min_ms, 10.0) || !approx_equal(summary.max_ms, 50.0)) {
        std::cerr << "[FAIL] Unexpected min/max: " << summary.min_ms << " / " << summary.max_ms
                  << "\n";
        return 1;
    }
    if (!approx_equal(summary.p50_ms, 30.0)) {
        std::cerr << "[FAIL] Expected p50_ms == 30.0, got " << summary.p50_ms << "\n";
        return 1;
    }

    return 0;
}

int test_resource_page_fault_detection() {
    const std::size_t buf_size = 4ULL * 1024 * 1024; // 4 MiB

    // Case 1: Touching freshly mapped pages must register minor page faults
    const auto before_fault = zerostride::ResourceSnapshot::capture();
    void* ptr = nullptr;
    const zerostride::Status map_st =
        zerostride::platform::map_anonymous_pages(buf_size, &ptr);
    if (!map_st.ok()) {
        std::cerr << "[FAIL] map_anonymous_pages failed: " << map_st.message << "\n";
        return 1;
    }
    std::memset(ptr, 0xAB, buf_size);
    const auto after_fault = zerostride::ResourceSnapshot::capture();
    const auto fault_delta = zerostride::ResourceDelta::compute(before_fault, after_fault);

    if (fault_delta.minor_page_faults <= 0) {
        std::cerr << "[FAIL] Expected minor_page_faults > 0 when faulting fresh anonymous pages, got "
                  << fault_delta.minor_page_faults << "\n";
        (void)zerostride::platform::unmap_pages(ptr, buf_size);
        return 1;
    }

    // Case 2: Re-writing already faulted pages must register 0 major page faults
    const auto before_reuse = zerostride::ResourceSnapshot::capture();
    std::memset(ptr, 0xCD, buf_size);
    const auto after_reuse = zerostride::ResourceSnapshot::capture();
    const auto reuse_delta = zerostride::ResourceDelta::compute(before_reuse, after_reuse);

    (void)zerostride::platform::unmap_pages(ptr, buf_size);

    if (reuse_delta.major_page_faults != 0) {
        std::cerr << "[FAIL] Expected 0 major_page_faults on prefaulted buffer reuse, got "
                  << reuse_delta.major_page_faults << "\n";
        return 1;
    }

    return 0;
}

int test_serialization_formats() {
    zerostride::TelemetryRecorder rec;
    rec.record_step_ms(12.34);
    const auto summary = rec.summarize();

    std::ostringstream json_os;
    zerostride::TelemetryRecorder::print_json(summary, "unit-test", json_os);
    if (json_os.str().find("\"label\": \"unit-test\"") == std::string::npos) {
        std::cerr << "[FAIL] JSON output missing label field\n";
        return 1;
    }

    std::ostringstream csv_os;
    zerostride::TelemetryRecorder::print_csv_header(csv_os);
    zerostride::TelemetryRecorder::print_csv_row(summary, "unit-test", csv_os);
    if (csv_os.str().find("unit-test,1,") == std::string::npos) {
        std::cerr << "[FAIL] CSV output missing expected row\n";
        return 1;
    }

    return 0;
}

} // namespace

int main() {
    if (test_statistics_calculation() != 0) {
        return 1;
    }
    if (test_resource_page_fault_detection() != 0) {
        return 1;
    }
    if (test_serialization_formats() != 0) {
        return 1;
    }

    std::cout << "[PASS] All ZeroStride Phase 0 telemetry unit tests succeeded.\n";
    return 0;
}
