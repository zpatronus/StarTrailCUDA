#pragma once
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <queue>

struct Frame {
    uint8_t* d_y_data;
    uint8_t* d_uv_data;
    int width;
    int height;
    int y_pitch;
    int uv_pitch;
    int64_t pts;
    bool is_last;

    Frame()
        : d_y_data(nullptr), d_uv_data(nullptr), width(0), height(0), y_pitch(0), uv_pitch(0),
          pts(0), is_last(false) {}
};

class FrameQueue {
  private:
    std::queue<Frame> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool finished_;

  public:
    explicit FrameQueue(size_t max_size = 10);
    ~FrameQueue();

    void push(const Frame& frame);
    Frame pop();
    bool try_pop(Frame& frame);
    void finish();
    bool is_finished() const;
    size_t size() const;
};
