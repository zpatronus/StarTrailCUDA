#pragma once
#include "buffer_pool.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

struct FrameRef {
    int buffer_id;
    int64_t pts;
    bool is_last;

    FrameRef() : buffer_id(-1), pts(0), is_last(false) {}
    FrameRef(int id, int64_t p) : buffer_id(id), pts(p), is_last(false) {}
};

class FrameRefQueue {
  private:
    std::queue<FrameRef> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool finished_;

  public:
    explicit FrameRefQueue(size_t max_size = 10);
    ~FrameRefQueue();

    void push(const FrameRef& ref);
    FrameRef pop();
    bool try_pop(FrameRef& ref);
    void finish();
    bool is_finished() const;
    size_t size() const;
};
