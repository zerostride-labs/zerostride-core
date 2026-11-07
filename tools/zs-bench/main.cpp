#include "platform/os_resources.hpp"
#include "zerostride/config.hpp"
#include "zerostride/status.hpp"
#include "zerostride/telemetry.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {

struct BenchOptions {
    std::size_t steps = 128;
    std::size_t buffer_bytes = 16ULL * 1024 * 1024; // 16 MiB per simulated expert working set
    std::string_view mode = "arena";                // "arena" | "fault"
    std::string_view format = "table";              // "table" | "json" | "csv"
    const char* label = "zs-synthetic-baseline";
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --steps <N>              Number of simulated token steps (default: 128)\n"
              << "  --buffer-mb <MiB>        Working buffer size in MiB (default: 16)\n"
              << "  --mode <arena|fault>     'arena' reuses pinned buffer; 'fault' maps fresh pages per step\n"
              << "  --format <table|json|csv> Output telemetry format (default: table)\n"
              << "  --label <str>            Benchmark run label\n"
              << "  --help                   Show this help message\n";
}

// Performs a deterministic strided reduction + update over a buffer to prevent dead-code elimination.
std::uint64_t compute_simulated_expert(volatile std::uint8_t* buf, std::size_t bytes,
                                       std::size_t page_stride, std::size_t step) noexcept {
    std::uint64_t acc = static_cast<std::uint64_t>(step);
    for (std::size_t offset = 0; offset < bytes; offset += page_stride) {
        std::uint8_t val = buf[offset];
        val = static_cast<std::uint8_t>((val + step + (offset >> 12)) & 0xFFU);
        buf[offset] = val;
        acc += val;
    }
    return acc;
}

} // namespace

int main(int argc, char** argv) {
    BenchOptions opts{};

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--steps" && i + 1 < argc) {
            opts.steps = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--buffer-mb" && i + 1 < argc) {
            opts.buffer_bytes =
                static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10)) * 1024ULL * 1024ULL;
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            opts.format = argv[++i];
        } else if (arg == "--label" && i + 1 < argc) {
            opts.label = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::size_t page_stride = zerostride::platform::system_page_size();
    std::uint8_t* static_buffer = nullptr;

    if (opts.mode == "arena") {
        void* raw_ptr = nullptr;
        const zerostride::Status st =
            zerostride::platform::map_anonymous_pages(opts.buffer_bytes, &raw_ptr);
        if (!st.ok()) {
            std::cerr << "Failed to allocate arena buffer: " << st.message << "\n";
            return 1;
        }
        static_buffer = static_cast<std::uint8_t*>(raw_ptr);
        std::memset(static_buffer, 0x2A, opts.buffer_bytes);
    } else if (opts.mode != "fault") {
        std::cerr << "Invalid --mode '" << opts.mode << "'. Expected 'arena' or 'fault'.\n";
        return 1;
    }

    zerostride::TelemetryRecorder recorder;
    recorder.reset();

    // Record simulated prefill (TTFT)
    recorder.begin_step();
    volatile std::uint64_t sink = 0;
    if (static_buffer != nullptr) {
        sink ^= compute_simulated_expert(static_buffer, opts.buffer_bytes, page_stride, 0);
    }
    const double prefill_summary_ms = recorder.end_step();
    recorder.reset();
    recorder.record_ttft_ms(prefill_summary_ms);

    // Hot decode loop
    for (std::size_t step = 0; step < opts.steps; ++step) {
        recorder.begin_step();

        if (opts.mode == "arena") {
            sink ^= compute_simulated_expert(static_buffer, opts.buffer_bytes, page_stride, step);
        } else {
            void* temp = nullptr;
            if (zerostride::platform::map_anonymous_pages(opts.buffer_bytes, &temp).ok()) {
                sink ^= compute_simulated_expert(static_cast<std::uint8_t*>(temp),
                                                 opts.buffer_bytes, page_stride, step);
                (void)zerostride::platform::unmap_pages(temp, opts.buffer_bytes);
            }
        }

        recorder.end_step();
    }

    const zerostride::TelemetrySummary summary = recorder.summarize();

    if (static_buffer != nullptr) {
        (void)zerostride::platform::unmap_pages(static_buffer, opts.buffer_bytes);
    }

    if (opts.format == "json") {
        zerostride::TelemetryRecorder::print_json(summary, opts.label, std::cout);
    } else if (opts.format == "csv") {
        zerostride::TelemetryRecorder::print_csv_header(std::cout);
        zerostride::TelemetryRecorder::print_csv_row(summary, opts.label, std::cout);
    } else {
        zerostride::TelemetryRecorder::print_table(summary, opts.label, std::cout);
    }

    return (sink == 0xFFFFFFFFFFFFFFFFULL) ? 2 : 0;
}
