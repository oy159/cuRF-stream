#pragma once

// Minimal CUDA error checking and RAII buffers. Host-only header (plain C++
// translation units may include it when linked against CUDA::cudart).

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace curf::cuda {

    inline void check(cudaError_t e, const char *what, const char *file, int line) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(what) + " failed at " + file + ":" + std::to_string(line) + ": " +
                                     cudaGetErrorString(e));
    }

#define CURF_CUDA_CHECK(expr) ::curf::cuda::check((expr), #expr, __FILE__, __LINE__)

    // Device allocation of `count` elements of T.
    template <typename T>
    class DeviceBuffer {
    public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(std::size_t count) : count_(count) {
            if (count) CURF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
        }
        ~DeviceBuffer() { reset(); }
        DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
        DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
            if (this != &o) {
                reset();
                ptr_ = std::exchange(o.ptr_, nullptr);
                count_ = std::exchange(o.count_, 0);
            }
            return *this;
        }
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        T *get() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return count_; }
        std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    private:
        void reset() noexcept {
            if (ptr_) cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
        T *ptr_ = nullptr;
        std::size_t count_ = 0;
    };

    // Page-locked host allocation (cudaHostAlloc), DMA-able by the GPU.
    template <typename T>
    class PinnedBuffer {
    public:
        PinnedBuffer() = default;
        explicit PinnedBuffer(std::size_t count, unsigned flags = cudaHostAllocDefault) : count_(count) {
            if (count) CURF_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T), flags));
        }
        ~PinnedBuffer() { reset(); }
        PinnedBuffer(PinnedBuffer &&o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
        PinnedBuffer &operator=(PinnedBuffer &&o) noexcept {
            if (this != &o) {
                reset();
                ptr_ = std::exchange(o.ptr_, nullptr);
                count_ = std::exchange(o.count_, 0);
            }
            return *this;
        }
        PinnedBuffer(const PinnedBuffer &) = delete;
        PinnedBuffer &operator=(const PinnedBuffer &) = delete;

        T *get() const noexcept { return ptr_; }
        T &operator[](std::size_t i) const noexcept { return ptr_[i]; }
        std::size_t size() const noexcept { return count_; }
        std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    private:
        void reset() noexcept {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
        T *ptr_ = nullptr;
        std::size_t count_ = 0;
    };

} // namespace curf::cuda
