#pragma once
#include "buffer_pool.h"
#include "frame_ref_queue.h"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <thread>

enum class RenderAlgorithm { MAX, LINEARAPPROX };

class VideoRenderer {
  private:
    int width_;
    int height_;
    int window_size_;
    RenderAlgorithm algorithm_;
    std::shared_ptr<BufferPool> buffer_pool_;
    std::shared_ptr<FrameRefQueue> input_queue_;
    std::shared_ptr<FrameRefQueue> output_queue_;

    uint8_t* d_accum_y_;
    uint8_t* d_accum_uv_;
    float* d_accum_y_float_;
    float* d_accum_uv_float_;

    cudaStream_t stream_;
    std::thread render_thread_;
    bool running_;

    std::atomic<long long> total_input_queue_pop_time_us_{0};
    std::atomic<long long> total_buffer_acquire_time_us_{0};
    std::atomic<long long> total_render_time_us_{0};
    std::atomic<long long> total_buffer_release_time_us_{0};
    std::atomic<long long> total_output_queue_push_time_us_{0};
    std::atomic<int> frames_rendered_{0};

    void max_renderer();
    void linear_approx_renderer();
    void render_loop();

  public:
    VideoRenderer(int width, int height, int window_size, RenderAlgorithm algorithm,
                  std::shared_ptr<BufferPool> buffer_pool,
                  std::shared_ptr<FrameRefQueue> input_queue,
                  std::shared_ptr<FrameRefQueue> output_queue);
    ~VideoRenderer();

    void start();
    void wait();
    void print_stats() const;
};

void launch_max_kernel(uint8_t* d_output, const uint8_t* d_input, uint8_t* d_max_frame,
                       float decay_factor, int width, int height, int pitch, cudaStream_t stream);

void launch_linear_approx_kernel(uint8_t* d_output, const uint8_t* d_input, float* d_y_frame,
                                 float factor, float L, int width, int height, int pitch,
                                 cudaStream_t stream);
