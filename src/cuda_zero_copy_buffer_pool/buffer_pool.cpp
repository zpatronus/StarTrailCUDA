#include "buffer_pool.h"
#include "utils/cuda_utils.h"
#include <iostream>

BufferPool::BufferPool(int width, int height, int pool_size)
    : width_(width), height_(height), pool_size_(pool_size), initialized_(false) {
    buffers_.resize(pool_size);
}

BufferPool::~BufferPool() {
    for (auto& buffer : buffers_) {
        if (buffer.d_y_data) {
            cudaFree(buffer.d_y_data);
            buffer.d_y_data = nullptr;
        }
        if (buffer.d_uv_data) {
            cudaFree(buffer.d_uv_data);
            buffer.d_uv_data = nullptr;
        }
    }
}

void BufferPool::initialize() {
    if (initialized_) {
        return;
    }

    std::cout << "Initializing buffer pool with " << pool_size_ << " buffers...\n";

    for (int i = 0; i < pool_size_; i++) {
        FrameBuffer& buffer = buffers_[i];
        buffer.buffer_id = i;
        buffer.width = width_;
        buffer.height = height_;
        buffer.y_pitch = width_;
        buffer.uv_pitch = width_;
        buffer.pts = 0;
        buffer.is_last = false;

        CUDA_CHECK(cudaMalloc(&buffer.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&buffer.d_uv_data, width_ * (height_ / 2)));

        CUDA_CHECK(cudaMemset(buffer.d_y_data, 0, width_ * height_));
        CUDA_CHECK(cudaMemset(buffer.d_uv_data, 128, width_ * (height_ / 2)));

        available_ids_.push(i);
    }

    initialized_ = true;
    std::cout << "Buffer pool initialized successfully\n";
}

FrameBuffer* BufferPool::acquire_buffer() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_available_.wait(lock, [this] { return !available_ids_.empty(); });

    int buffer_id = available_ids_.front();
    available_ids_.pop();

    return &buffers_[buffer_id];
}

void BufferPool::release_buffer(int buffer_id) {
    if (buffer_id < 0 || buffer_id >= pool_size_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    available_ids_.push(buffer_id);
    cv_available_.notify_one();
}

int BufferPool::available_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_ids_.size();
}
