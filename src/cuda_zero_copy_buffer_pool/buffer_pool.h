#pragma once
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

struct FrameBuffer {
    int buffer_id;

    uint8_t* d_y_data;
    uint8_t* d_uv_data;
    int width;
    int height;
    int y_pitch;
    int uv_pitch;

    int64_t pts;
    bool is_last;

    FrameBuffer()
        : buffer_id(-1), d_y_data(nullptr), d_uv_data(nullptr), width(0), height(0), y_pitch(0),
          uv_pitch(0), pts(0), is_last(false) {}
};

class BufferPool {
  public:
    std::vector<FrameBuffer> buffers_;

  private:
    std::queue<int> available_ids_;
    mutable std::mutex mutex_;
    std::condition_variable cv_available_;

    int width_;
    int height_;
    int pool_size_;
    bool initialized_;

  public:
    BufferPool(int width, int height, int pool_size);
    ~BufferPool();

    void initialize();

    FrameBuffer* acquire_buffer();

    void release_buffer(int buffer_id);

    int available_count() const;

    int total_size() const { return pool_size_; }
};
