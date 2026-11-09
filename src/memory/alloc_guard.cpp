#include "memory/alloc_guard.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

#if defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

namespace zerostride::memory {

namespace {

// Stored in static .bss (not dynamic TLS) so __sanitizer_malloc_hook never triggers
// Darwin's _tlv_bootstrap -> calloc recursion on first touch.
std::atomic<std::size_t> g_active_guards{0};
std::atomic<std::size_t> g_alloc_count{0};

[[nodiscard]] std::size_t query_zone_blocks_in_use() noexcept {
#if defined(__APPLE__)
    malloc_statistics_t stats{};
    ::malloc_zone_statistics(nullptr, &stats);
    return stats.blocks_in_use;
#else
    return 0;
#endif
}

} // namespace

AllocGuard::AllocGuard() noexcept
    : start_alloc_count_(g_alloc_count.load(std::memory_order_relaxed)),
      start_zone_blocks_(query_zone_blocks_in_use()) {
    g_active_guards.fetch_add(1, std::memory_order_relaxed);
}

AllocGuard::~AllocGuard() noexcept {
    g_active_guards.fetch_sub(1, std::memory_order_relaxed);
}

std::size_t AllocGuard::allocations_in_scope() const noexcept {
    const std::size_t current_count = g_alloc_count.load(std::memory_order_relaxed);
    const std::size_t hook_delta = current_count - start_alloc_count_;
    const std::size_t current_blocks = query_zone_blocks_in_use();
    const std::size_t zone_delta =
        (current_blocks > start_zone_blocks_) ? (current_blocks - start_zone_blocks_) : 0;
    return (hook_delta > zone_delta) ? hook_delta : zone_delta;
}

Status AllocGuard::verify_zero_allocations() const noexcept {
    if (allocations_in_scope() != 0) {
        return Status::error(StatusCode::kInternalError,
                             "Zero-malloc hot-loop violation: dynamic allocation detected");
    }
    return Status::success();
}

bool AllocGuard::is_active_on_thread() noexcept {
    return g_active_guards.load(std::memory_order_relaxed) > 0;
}

void AllocGuard::notify_allocation(std::size_t /*bytes*/) noexcept {
    if (g_active_guards.load(std::memory_order_relaxed) > 0) {
        g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace zerostride::memory

#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)

extern "C" void __sanitizer_malloc_hook(const volatile void* /*ptr*/, std::size_t size) {
    zerostride::memory::AllocGuard::notify_allocation(size);
}

#else

void* operator new(std::size_t size) {
    zerostride::memory::AllocGuard::notify_allocation(size);
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}

void* operator new[](std::size_t size) {
    zerostride::memory::AllocGuard::notify_allocation(size);
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p == nullptr) {
        std::abort();
    }
    return p;
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    zerostride::memory::AllocGuard::notify_allocation(size);
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    zerostride::memory::AllocGuard::notify_allocation(size);
    return std::malloc(size == 0 ? 1 : size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t /*size*/) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t /*size*/) noexcept {
    std::free(ptr);
}

#endif
