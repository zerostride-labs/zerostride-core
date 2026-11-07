#pragma once

#include <cstdint>
#include <string_view>

namespace zerostride {

/// Deterministic status codes used across all subsystems (replaces C++ exceptions).
enum class StatusCode : std::uint8_t {
    kOk = 0,
    kInvalidArgument = 1,
    kOutOfMemory = 2,
    kArenaExhausted = 3,
    kIoError = 4,
    kCorruptedFormat = 5,
    kUnsupportedFeature = 6,
    kInternalError = 7,
};

/// Lightweight, zero-allocation status result for deterministic error handling.
struct [[nodiscard]] Status {
    StatusCode code = StatusCode::kOk;
    std::string_view message{};

    [[nodiscard]] constexpr bool ok() const noexcept { return code == StatusCode::kOk; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] static constexpr Status success() noexcept { return Status{StatusCode::kOk, {}}; }

    [[nodiscard]] static constexpr Status error(StatusCode c, std::string_view msg) noexcept {
        return Status{c, msg};
    }
};

} // namespace zerostride
