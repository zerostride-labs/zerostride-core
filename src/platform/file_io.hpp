#pragma once

#include "zerostride/status.hpp"

#include <cstddef>
#include <cstdint>

namespace zerostride::platform {

/// Move-only RAII wrapper around a POSIX file descriptor for deterministic positional I/O.
class FileHandle {
public:
    FileHandle() noexcept = default;
    ~FileHandle() noexcept;

    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FileHandle(FileHandle&& other) noexcept;
    FileHandle& operator=(FileHandle&& other) noexcept;

    /// Opens a file for read-only positional access (`pread`).
    /// If `disable_os_cache` is true on Darwin, sets `F_NOCACHE` to avoid polluting OS page cache.
    [[nodiscard]] Status open_readonly(const char* path, bool disable_os_cache = false) noexcept;

    /// Opens or creates a file for writing (truncating any existing file).
    [[nodiscard]] Status open_write_trunc(const char* path) noexcept;

    /// Closes the underlying file descriptor if open.
    void close() noexcept;

    /// Returns true if the file descriptor is open.
    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }

    /// Returns the total size of the open file in bytes.
    [[nodiscard]] std::uint64_t size_bytes() const noexcept { return size_bytes_; }

    /// Reads exactly `bytes` from `offset` into `dst` without modifying file descriptor position.
    [[nodiscard]] Status pread_exact(void* dst, std::size_t bytes,
                                     std::uint64_t offset) const noexcept;

    /// Writes exactly `bytes` at `offset` from `src`.
    [[nodiscard]] Status pwrite_exact(const void* src, std::size_t bytes,
                                      std::uint64_t offset) noexcept;

    /// Hints to the OS kernel that `[offset, offset + bytes)` will be read soon (non-blocking).
    [[nodiscard]] Status advise_willneed(std::uint64_t offset, std::size_t bytes) const noexcept;

private:
    int fd_ = -1;
    std::uint64_t size_bytes_ = 0;
};

} // namespace zerostride::platform
