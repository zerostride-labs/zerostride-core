#include "platform/os_resources.hpp"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace zerostride::platform {

namespace {

double timeval_to_ms(const timeval& tv) noexcept {
    return static_cast<double>(tv.tv_sec) * 1000.0 + static_cast<double>(tv.tv_usec) / 1000.0;
}

} // namespace

ResourceSnapshot query_os_resources() noexcept {
    ResourceSnapshot snap{};
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) == 0) {
        snap.major_page_faults = usage.ru_majflt;
        snap.minor_page_faults = usage.ru_minflt;
#if defined(__APPLE__)
        // On macOS/Darwin, ru_maxrss is reported in bytes.
        snap.max_rss_bytes = usage.ru_maxrss;
#else
        // On Linux and other POSIX systems, ru_maxrss is reported in kilobytes.
        snap.max_rss_bytes = static_cast<std::int64_t>(usage.ru_maxrss) * 1024;
#endif
        snap.voluntary_ctx_switches = usage.ru_nvcsw;
        snap.involuntary_ctx_switches = usage.ru_nivcsw;
        snap.user_cpu_ms = timeval_to_ms(usage.ru_utime);
        snap.system_cpu_ms = timeval_to_ms(usage.ru_stime);
    }

#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (::task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                    reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        snap.current_rss_bytes = static_cast<std::int64_t>(info.resident_size);
    } else {
        snap.current_rss_bytes = snap.max_rss_bytes;
    }
#else
    snap.current_rss_bytes = snap.max_rss_bytes;
#endif

    return snap;
}

std::size_t system_page_size() noexcept {
    const long page_sz = ::sysconf(_SC_PAGESIZE);
    return (page_sz > 0) ? static_cast<std::size_t>(page_sz) : 4096U;
}

Status map_anonymous_pages(std::size_t size_bytes, void** out_ptr) noexcept {
    if (out_ptr == nullptr || size_bytes == 0) {
        return Status::error(StatusCode::kInvalidArgument,
                             "Invalid arguments to map_anonymous_pages");
    }
    void* ptr = ::mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        *out_ptr = nullptr;
        return Status::error(StatusCode::kOutOfMemory, "mmap(MAP_ANONYMOUS) failed");
    }
    *out_ptr = ptr;
    return Status::success();
}

Status unmap_pages(void* ptr, std::size_t size_bytes) noexcept {
    if (ptr == nullptr || size_bytes == 0) {
        return Status::error(StatusCode::kInvalidArgument, "Invalid arguments to unmap_pages");
    }
    if (::munmap(ptr, size_bytes) != 0) {
        return Status::error(StatusCode::kInternalError, "munmap() failed");
    }
    return Status::success();
}

} // namespace zerostride::platform
