#include "platform/file_io.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace zerostride::platform {

FileHandle::~FileHandle() noexcept {
    close();
}

FileHandle::FileHandle(FileHandle&& other) noexcept
    : fd_(other.fd_), size_bytes_(other.size_bytes_) {
    other.fd_ = -1;
    other.size_bytes_ = 0;
}

FileHandle& FileHandle::operator=(FileHandle&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        size_bytes_ = other.size_bytes_;
        other.fd_ = -1;
        other.size_bytes_ = 0;
    }
    return *this;
}

Status FileHandle::open_readonly(const char* path, bool disable_os_cache) noexcept {
    close();
    if (path == nullptr) {
        return Status::error(StatusCode::kInvalidArgument, "Null path passed to open_readonly");
    }

    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return Status::error(StatusCode::kIoError, "Failed to open file for reading");
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        ::close(fd);
        return Status::error(StatusCode::kIoError, "fstat() failed on open file");
    }

#if defined(__APPLE__)
    if (disable_os_cache) {
        (void)::fcntl(fd, F_NOCACHE, 1);
    }
#else
    (void)disable_os_cache;
#endif

    fd_ = fd;
    size_bytes_ = static_cast<std::uint64_t>(st.st_size);
    return Status::success();
}

Status FileHandle::open_write_trunc(const char* path) noexcept {
    close();
    if (path == nullptr) {
        return Status::error(StatusCode::kInvalidArgument, "Null path passed to open_write_trunc");
    }

    const int fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return Status::error(StatusCode::kIoError, "Failed to open file for writing");
    }

    fd_ = fd;
    size_bytes_ = 0;
    return Status::success();
}

void FileHandle::close() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
        size_bytes_ = 0;
    }
}

Status FileHandle::pread_exact(void* dst, std::size_t bytes,
                               std::uint64_t offset) const noexcept {
    if (fd_ < 0) {
        return Status::error(StatusCode::kIoError, "pread_exact called on closed FileHandle");
    }
    if (bytes == 0) {
        return Status::success();
    }
    if (dst == nullptr) {
        return Status::error(StatusCode::kInvalidArgument, "Null destination buffer in pread_exact");
    }
    if (offset > size_bytes_ || bytes > (size_bytes_ - offset)) {
        return Status::error(StatusCode::kCorruptedFormat,
                             "pread_exact range exceeds file size");
    }

    auto* out = static_cast<std::uint8_t*>(dst);
    std::size_t total_read = 0;

    while (total_read < bytes) {
        const std::size_t remaining = bytes - total_read;
        const auto current_off = static_cast<off_t>(offset + total_read);
        const ssize_t n = ::pread(fd_, out + total_read, remaining, current_off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Status::error(StatusCode::kIoError, "pread() syscall failed");
        }
        if (n == 0) {
            return Status::error(StatusCode::kCorruptedFormat, "Unexpected EOF during pread_exact");
        }
        total_read += static_cast<std::size_t>(n);
    }

    return Status::success();
}

Status FileHandle::pwrite_exact(const void* src, std::size_t bytes,
                                std::uint64_t offset) noexcept {
    if (fd_ < 0) {
        return Status::error(StatusCode::kIoError, "pwrite_exact called on closed FileHandle");
    }
    if (bytes == 0) {
        return Status::success();
    }
    if (src == nullptr) {
        return Status::error(StatusCode::kInvalidArgument, "Null source buffer in pwrite_exact");
    }

    const auto* in = static_cast<const std::uint8_t*>(src);
    std::size_t total_written = 0;

    while (total_written < bytes) {
        const std::size_t remaining = bytes - total_written;
        const auto current_off = static_cast<off_t>(offset + total_written);
        const ssize_t n = ::pwrite(fd_, in + total_written, remaining, current_off);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Status::error(StatusCode::kIoError, "pwrite() syscall failed");
        }
        total_written += static_cast<std::size_t>(n);
    }

    const std::uint64_t end_pos = offset + bytes;
    if (end_pos > size_bytes_) {
        size_bytes_ = end_pos;
    }

    return Status::success();
}

Status FileHandle::advise_willneed(std::uint64_t offset, std::size_t bytes) const noexcept {
    if (fd_ < 0) {
        return Status::error(StatusCode::kIoError, "advise_willneed called on closed FileHandle");
    }
    if (bytes == 0) {
        return Status::success();
    }

#if defined(__APPLE__)
    struct radvisory adv {};
    adv.ra_offset = static_cast<off_t>(offset);
    adv.ra_count = static_cast<int>(bytes);
    if (::fcntl(fd_, F_RDADVISE, &adv) != 0) {
        return Status::error(StatusCode::kIoError, "fcntl(F_RDADVISE) failed");
    }
#elif defined(POSIX_FADV_WILLNEED)
    if (::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(bytes),
                        POSIX_FADV_WILLNEED) != 0) {
        return Status::error(StatusCode::kIoError, "posix_fadvise(POSIX_FADV_WILLNEED) failed");
    }
#else
    (void)offset;
    (void)bytes;
#endif
    return Status::success();
}

} // namespace zerostride::platform
