#include "frame_queue.h"

FrameQueue::FrameQueue(size_t max_size) : max_size_(max_size), finished_(false) {}

FrameQueue::~FrameQueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

void FrameQueue::push(const Frame& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this] { return queue_.size() < max_size_ || finished_; });
    if (finished_)
        return;
    queue_.push(frame);
    cv_not_empty_.notify_one();
}

Frame FrameQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
    if (queue_.empty() && finished_) {
        Frame frame;
        frame.is_last = true;
        return frame;
    }
    Frame frame = queue_.front();
    queue_.pop();
    cv_not_full_.notify_one();
    return frame;
}

bool FrameQueue::try_pop(Frame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return false;
    }
    frame = queue_.front();
    queue_.pop();
    cv_not_full_.notify_one();
    return true;
}

void FrameQueue::finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    finished_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
}

bool FrameQueue::is_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return finished_;
}

size_t FrameQueue::size() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
}
