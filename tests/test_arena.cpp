#include "memory/alloc_guard.hpp"
#include "memory/static_arena.hpp"
#include "platform/os_resources.hpp"
#include "zerostride/telemetry.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

int test_arena_partition_and_alignment() {
    zerostride::memory::ArenaLayoutConfig cfg{};
    cfg.persistent_weights_bytes = 64 * 1024;
    cfg.kv_cache_bytes = 32 * 1024;
    cfg.staging_buffer_bytes = 64 * 1024;
    cfg.scratch_bytes = 32 * 1024;
    cfg.memory_ceiling_bytes = 16 * 1024 * 1024;
    cfg.prefault_on_init = true;
    cfg.pin_staging_buffers = true;

    zerostride::memory::StaticArena arena;
    const zerostride::Status st = arena.initialize(cfg);
    if (!st.ok()) {
        std::cerr << "[FAIL] StaticArena::initialize failed: " << st.message << "\n";
        return 1;
    }

    const std::size_t page_sz = zerostride::platform::system_page_size();
    const auto stg_a = arena.staging_buffer_a();
    const auto stg_b = arena.staging_buffer_b();

    if (stg_a.empty() || stg_b.empty() || stg_a.data() == stg_b.data()) {
        std::cerr << "[FAIL] Invalid staging buffers A/B\n";
        return 1;
    }

    // Verify both staging buffers are strictly OS page-aligned
    if ((reinterpret_cast< std::uintptr_t >(stg_a.data()) % page_sz) != 0 ||
        (reinterpret_cast< std::uintptr_t >(stg_b.data()) % page_sz) != 0) {
        std::cerr << "[FAIL] Staging buffers are not OS page-aligned\n";
        return 1;
    }

    // Verify sub-allocation alignment (128-byte cache-line alignment after an unaligned 17-byte request)
    void* p1 = nullptr;
    void* p2 = nullptr;
    if (!arena.allocate_persistent(17, &p1, 64).ok() ||
        !arena.allocate_persistent(256, &p2, 128).ok()) {
        std::cerr << "[FAIL] Persistent sub-allocation failed\n";
        return 1;
    }
    if ((reinterpret_cast< std::uintptr_t >(p2) % 128U) != 0) {
        std::cerr << "[FAIL] Expected 128-byte alignment on second persistent allocation\n";
        return 1;
    }

    // Verify region exhaustion returns kArenaExhausted without crashing
    void* p_overflow = nullptr;
    const zerostride::Status ovf =
        arena.allocate_persistent(arena.persistent_capacity_bytes() + 1, &p_overflow, 64);
    if (ovf.code != zerostride::StatusCode::kArenaExhausted) {
        std::cerr << "[FAIL] Expected kArenaExhausted on region overflow\n";
        return 1;
    }

    return 0;
}

int test_arena_ceiling_enforcement() {
    zerostride::memory::ArenaLayoutConfig cfg{};
    cfg.persistent_weights_bytes = 8 * 1024 * 1024;
    cfg.staging_buffer_bytes = 4 * 1024 * 1024;
    cfg.memory_ceiling_bytes = 4 * 1024 * 1024; // Ceiling smaller than requested regions

    zerostride::memory::StaticArena arena;
    const zerostride::Status st = arena.initialize(cfg);
    if (st.code != zerostride::StatusCode::kArenaExhausted) {
        std::cerr << "[FAIL] Expected kArenaExhausted when exceeding memory_ceiling_bytes\n";
        return 1;
    }
    return 0;
}

int test_zero_malloc_hot_loop_and_scratch_reuse() {
    zerostride::memory::ArenaLayoutConfig cfg{};
    cfg.persistent_weights_bytes = 64 * 1024;
    cfg.kv_cache_bytes = 64 * 1024;
    cfg.staging_buffer_bytes = 64 * 1024;
    cfg.scratch_bytes = 64 * 1024;
    cfg.prefault_on_init = true;

    zerostride::memory::StaticArena arena;
    if (!arena.initialize(cfg).ok()) {
        std::cerr << "[FAIL] Arena init failed in hot-loop test\n";
        return 1;
    }

    zerostride::memory::AllocGuard guard;
    zerostride::ResourceSnapshot before_res{};

    // Simulate 1,000 autoregressive token decode steps reusing scratch memory
    for (std::size_t step = 0; step <= 1000; ++step) {
        if (step == 1) {
            // Capture baseline after step 0 warmup so steady-state decode steps are measured
            before_res = zerostride::ResourceSnapshot::capture();
        }
        arena.reset_scratch();

        void* act1 = nullptr;
        void* act2 = nullptr;
        if (!arena.allocate_scratch(4096, &act1, 128).ok() ||
            !arena.allocate_scratch(8192, &act2, 128).ok()) {
            std::cerr << "[FAIL] Scratch allocation failed on step " << step << "\n";
            return 1;
        }

        std::memset(act1, static_cast<int>(step & 0xFFU), 4096);
        std::memset(act2, static_cast<int>((step + 1) & 0xFFU), 8192);
    }

    const auto after_res = zerostride::ResourceSnapshot::capture();
    const zerostride::Status zero_st = guard.verify_zero_allocations();
    if (!zero_st.ok()) {
        std::cerr << "[FAIL] Heap allocation detected in hot loop: "
                  << guard.allocations_in_scope() << " allocations\n";
        return 1;
    }
    const auto delta = zerostride::ResourceDelta::compute(before_res, after_res);

    if (delta.major_page_faults != 0 || delta.minor_page_faults != 0) {
        std::cerr << "[FAIL] Expected 0 major and 0 minor page faults in prefaulted hot loop, got "
                  << delta.major_page_faults << " major / " << delta.minor_page_faults
                  << " minor\n";
        return 1;
    }

    return 0;
}

int test_alloc_guard_catches_violation() {
    zerostride::memory::AllocGuard guard;
    volatile std::size_t dynamic_sz = 128;
    void* raw_heap = ::operator new(dynamic_sz);
    static volatile void* escape_sink = raw_heap;
    (void)escape_sink;
    const std::size_t detected = guard.allocations_in_scope();
    ::operator delete(raw_heap);

    if (detected == 0) {
        std::cerr << "[FAIL] AllocGuard failed to detect operator new allocation\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (test_arena_partition_and_alignment() != 0) {
        return 1;
    }
    if (test_arena_ceiling_enforcement() != 0) {
        return 1;
    }
    if (test_zero_malloc_hot_loop_and_scratch_reuse() != 0) {
        return 1;
    }
    if (test_alloc_guard_catches_violation() != 0) {
        return 1;
    }

    std::cout << "[PASS] All ZeroStride Phase 1 StaticArena & AllocGuard unit tests succeeded.\n";
    return 0;
}
