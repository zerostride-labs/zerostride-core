#include "memory/static_arena.hpp"

#include "platform/os_resources.hpp"

namespace zerostride::memory {

namespace {

[[nodiscard]] constexpr bool is_power_of_two(std::size_t x) noexcept {
    return x != 0 && ((x & (x - 1U)) == 0);
}

[[nodiscard]] Status bump_allocate(std::uint8_t* region_base, std::size_t region_cap,
                                   std::size_t& region_used, std::size_t size_bytes,
                                   std::size_t alignment, void** out_ptr) noexcept {
    if (out_ptr == nullptr || size_bytes == 0 || !is_power_of_two(alignment)) {
        return Status::error(StatusCode::kInvalidArgument,
                             "Invalid arguments to StaticArena sub-allocator");
    }

    const std::size_t aligned_offset = platform::align_up(region_used, alignment);
    if (aligned_offset > region_cap || size_bytes > (region_cap - aligned_offset)) {
        *out_ptr = nullptr;
        return Status::error(StatusCode::kArenaExhausted,
                             "StaticArena region capacity exhausted");
    }

    *out_ptr = region_base + aligned_offset;
    region_used = aligned_offset + size_bytes;
    return Status::success();
}

} // namespace

StaticArena::~StaticArena() noexcept {
    release();
}

StaticArena::StaticArena(StaticArena&& other) noexcept
    : base_ptr_(other.base_ptr_),
      total_bytes_(other.total_bytes_),
      persistent_offset_(other.persistent_offset_),
      persistent_cap_(other.persistent_cap_),
      persistent_used_(other.persistent_used_),
      kv_cache_offset_(other.kv_cache_offset_),
      kv_cache_cap_(other.kv_cache_cap_),
      kv_cache_used_(other.kv_cache_used_),
      staging_a_offset_(other.staging_a_offset_),
      staging_b_offset_(other.staging_b_offset_),
      staging_cap_(other.staging_cap_),
      staging_pinned_(other.staging_pinned_),
      scratch_offset_(other.scratch_offset_),
      scratch_cap_(other.scratch_cap_),
      scratch_used_(other.scratch_used_) {
    other.base_ptr_ = nullptr;
    other.total_bytes_ = 0;
    other.staging_pinned_ = false;
}

StaticArena& StaticArena::operator=(StaticArena&& other) noexcept {
    if (this != &other) {
        release();
        base_ptr_ = other.base_ptr_;
        total_bytes_ = other.total_bytes_;
        persistent_offset_ = other.persistent_offset_;
        persistent_cap_ = other.persistent_cap_;
        persistent_used_ = other.persistent_used_;
        kv_cache_offset_ = other.kv_cache_offset_;
        kv_cache_cap_ = other.kv_cache_cap_;
        kv_cache_used_ = other.kv_cache_used_;
        staging_a_offset_ = other.staging_a_offset_;
        staging_b_offset_ = other.staging_b_offset_;
        staging_cap_ = other.staging_cap_;
        staging_pinned_ = other.staging_pinned_;
        scratch_offset_ = other.scratch_offset_;
        scratch_cap_ = other.scratch_cap_;
        scratch_used_ = other.scratch_used_;

        other.base_ptr_ = nullptr;
        other.total_bytes_ = 0;
        other.staging_pinned_ = false;
    }
    return *this;
}

Status StaticArena::initialize(const ArenaLayoutConfig& cfg) noexcept {
    release();

    const std::size_t page_sz = platform::system_page_size();
    const std::size_t p_cap = platform::align_up(cfg.persistent_weights_bytes, page_sz);
    const std::size_t kv_cap = platform::align_up(cfg.kv_cache_bytes, page_sz);
    const std::size_t stg_cap = platform::align_up(cfg.staging_buffer_bytes, page_sz);
    const std::size_t scr_cap = platform::align_up(cfg.scratch_bytes, page_sz);

    const std::size_t total_required = p_cap + kv_cap + (2U * stg_cap) + scr_cap;
    if (total_required == 0) {
        return Status::error(StatusCode::kInvalidArgument,
                             "StaticArena total requested capacity cannot be zero");
    }

    if (cfg.memory_ceiling_bytes > 0 && total_required > cfg.memory_ceiling_bytes) {
        return Status::error(StatusCode::kArenaExhausted,
                             "Requested StaticArena layout exceeds configured memory ceiling");
    }

    void* raw_ptr = nullptr;
    const Status map_st = platform::map_anonymous_pages(total_required, &raw_ptr);
    if (!map_st.ok()) {
        return map_st;
    }

    base_ptr_ = static_cast<std::uint8_t*>(raw_ptr);
    total_bytes_ = total_required;

    persistent_offset_ = 0;
    persistent_cap_ = p_cap;
    persistent_used_ = 0;

    kv_cache_offset_ = persistent_offset_ + p_cap;
    kv_cache_cap_ = kv_cap;
    kv_cache_used_ = 0;

    staging_a_offset_ = kv_cache_offset_ + kv_cap;
    staging_b_offset_ = staging_a_offset_ + stg_cap;
    staging_cap_ = stg_cap;
    staging_pinned_ = false;

    scratch_offset_ = staging_b_offset_ + stg_cap;
    scratch_cap_ = scr_cap;
    scratch_used_ = 0;

    if (cfg.prefault_on_init) {
        platform::prefault_pages(base_ptr_, total_bytes_);
    }

    if (cfg.pin_staging_buffers && stg_cap > 0) {
        const Status lock_st =
            platform::lock_pages(base_ptr_ + staging_a_offset_, 2U * stg_cap);
        staging_pinned_ = lock_st.ok();
    }

    return Status::success();
}

void StaticArena::release() noexcept {
    if (base_ptr_ != nullptr) {
        if (staging_pinned_ && staging_cap_ > 0) {
            (void)platform::unlock_pages(base_ptr_ + staging_a_offset_, 2U * staging_cap_);
            staging_pinned_ = false;
        }
        (void)platform::unmap_pages(base_ptr_, total_bytes_);
        base_ptr_ = nullptr;
        total_bytes_ = 0;
        persistent_cap_ = 0;
        persistent_used_ = 0;
        kv_cache_cap_ = 0;
        kv_cache_used_ = 0;
        staging_cap_ = 0;
        scratch_cap_ = 0;
        scratch_used_ = 0;
    }
}

Status StaticArena::allocate_persistent(std::size_t size_bytes, void** out_ptr,
                                        std::size_t alignment) noexcept {
    if (base_ptr_ == nullptr) {
        return Status::error(StatusCode::kInternalError, "StaticArena not initialized");
    }
    return bump_allocate(base_ptr_ + persistent_offset_, persistent_cap_, persistent_used_,
                         size_bytes, alignment, out_ptr);
}

Status StaticArena::allocate_kv_cache(std::size_t size_bytes, void** out_ptr,
                                      std::size_t alignment) noexcept {
    if (base_ptr_ == nullptr) {
        return Status::error(StatusCode::kInternalError, "StaticArena not initialized");
    }
    return bump_allocate(base_ptr_ + kv_cache_offset_, kv_cache_cap_, kv_cache_used_,
                         size_bytes, alignment, out_ptr);
}

Status StaticArena::allocate_scratch(std::size_t size_bytes, void** out_ptr,
                                     std::size_t alignment) noexcept {
    if (base_ptr_ == nullptr) {
        return Status::error(StatusCode::kInternalError, "StaticArena not initialized");
    }
    return bump_allocate(base_ptr_ + scratch_offset_, scratch_cap_, scratch_used_,
                         size_bytes, alignment, out_ptr);
}

std::span<std::uint8_t> StaticArena::staging_buffer(std::size_t buffer_index) noexcept {
    if (base_ptr_ == nullptr || staging_cap_ == 0 || buffer_index > 1) {
        return {};
    }
    const std::size_t offset = (buffer_index == 0) ? staging_a_offset_ : staging_b_offset_;
    return std::span<std::uint8_t>(base_ptr_ + offset, staging_cap_);
}

std::span<const std::uint8_t> StaticArena::staging_buffer(std::size_t buffer_index) const noexcept {
    if (base_ptr_ == nullptr || staging_cap_ == 0 || buffer_index > 1) {
        return {};
    }
    const std::size_t offset = (buffer_index == 0) ? staging_a_offset_ : staging_b_offset_;
    return std::span<const std::uint8_t>(base_ptr_ + offset, staging_cap_);
}

} // namespace zerostride::memory
