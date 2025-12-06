#include "frame_ref_queue.h"

FrameRefQueue::FrameRefQueue(size_t max_size) : max_size_(max_size), finished_(false) {}

FrameRefQueue::~FrameRefQueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

void FrameRefQueue::push(const FrameRef& ref) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [this] { return queue_.size() < max_size_ || finished_; });
    if (finished_)
        return;
    queue_.push(ref);
    cv_not_empty_.notify_one();
}

FrameRef FrameRefQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
    if (queue_.empty() && finished_) {
        FrameRef ref;
        ref.is_last = true;
        return ref;
    }
    FrameRef ref = queue_.front();
    queue_.pop();
    cv_not_full_.notify_one();
    return ref;
}

bool FrameRefQueue::try_pop(FrameRef& ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return false;
    }
    ref = queue_.front();
    queue_.pop();
    cv_not_full_.notify_one();
    return true;
}

void FrameRefQueue::finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    finished_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
}

bool FrameRefQueue::is_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return finished_;
}

size_t FrameRefQueue::size() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
}
