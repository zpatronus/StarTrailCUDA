#pragma once
#include "../utils/frame_queue.h"
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
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

class VideoEncoder {
  private:
    std::string output_path_;
    int width_;
    int height_;
    int fps_;
    std::shared_ptr<FrameQueue> input_queue_;

    AVFormatContext* fmt_ctx_;
    AVCodecContext* codec_ctx_;
    AVStream* av_stream_;
    AVFrame* av_frame_;
    AVBufferRef* hw_device_ctx_;
    AVBufferRef* hw_frames_ctx_;

    std::thread encode_thread_;
    bool running_;
    int64_t frame_count_;

    cudaStream_t cuda_stream_;

    std::thread write_thread_;
    std::queue<AVPacket*> write_queue_;
    std::mutex write_mutex_;
    std::condition_variable write_cv_;
    bool write_running_;

    std::atomic<long long> total_queue_wait_time_us_{0};
    std::atomic<long long> total_gpu_upload_time_us_{0};
    std::atomic<long long> total_encode_time_us_{0};
    std::atomic<long long> total_packet_queue_time_us_{0};
    std::atomic<long long> total_write_time_us_{0};
    std::atomic<long long> total_iteration_time_us_{0};
    std::atomic<int> frames_encoded_{0};
    std::atomic<int> packets_written_{0};

    void encode_loop();
    void write_loop();

  public:
    VideoEncoder(const std::string& output_path, int width, int height, int fps,
                 std::shared_ptr<FrameQueue> input_queue);
    ~VideoEncoder();

    void start();
    void wait();
    void print_stats() const;
};
