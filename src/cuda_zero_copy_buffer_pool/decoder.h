#pragma once
#include "buffer_pool.h"
#include "frame_ref_queue.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

class VideoDecoder {
  private:
    std::string input_path_;
    int frame_step_;
    std::shared_ptr<BufferPool> buffer_pool_;
    std::shared_ptr<FrameRefQueue> output_queue_;

    AVFormatContext* fmt_ctx_;
    AVCodecContext* codec_ctx_;
    AVBufferRef* hw_device_ctx_;
    int video_stream_idx_;
    int width_;
    int height_;
    int fps_;
    int total_frames_;

    std::thread decode_thread_;
    std::thread packet_reader_thread_;
    bool running_;
    bool packet_reader_running_;
    bool end_of_stream_;

    cudaStream_t stream_;

    std::queue<AVPacket*> packet_queue_;
    std::mutex packet_mutex_;
    std::condition_variable packet_cv_;
    static const size_t MAX_PACKET_QUEUE_SIZE = 50;

    std::atomic<long long> total_read_time_us_{0};
    std::atomic<long long> total_decode_time_us_{0};
    std::atomic<long long> total_gpu_transfer_time_us_{0};
    std::atomic<long long> total_output_queue_push_time_us_{0};
    std::atomic<int> packets_read_{0};
    std::atomic<int> frames_decoded_{0};
    std::atomic<int> frames_pushed_{0};

    void decode_loop();
    void packet_reader_loop();

    int get_expected_output_frames() const {
        return (total_frames_ + frame_step_ - 1) / frame_step_;
    }

  public:
    VideoDecoder(const std::string& input_path, int frame_step,
                 std::shared_ptr<BufferPool> buffer_pool,
                 std::shared_ptr<FrameRefQueue> output_queue);
    ~VideoDecoder();

    void start();
    void wait();
    void print_stats() const;
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    int get_fps() const { return fps_; }
    int get_total_frames() const { return total_frames_; }
};
