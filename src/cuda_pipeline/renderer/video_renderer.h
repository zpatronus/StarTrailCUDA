#pragma once
#include "../utils/frame_queue.h"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <thread>

enum class RenderAlgorithm { MAX, AVERAGE, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY };

class VideoRenderer {
  private:
    int width_;
    int height_;
    int window_size_;
    RenderAlgorithm algorithm_;
    std::shared_ptr<FrameQueue> input_queue_;
    std::shared_ptr<FrameQueue> output_queue_;

    uint8_t* d_accumulator_;
    cudaStream_t stream_;
    std::thread render_thread_;

    std::atomic<long long> total_input_queue_pop_time_us_{0};
    std::atomic<long long> total_memory_alloc_time_us_{0};
    std::atomic<long long> total_render_time_us_{0};
    std::atomic<long long> total_memory_free_time_us_{0};
    std::atomic<long long> total_output_queue_push_time_us_{0};
    std::atomic<int> frames_rendered_{0};

    void max_renderer();
    void average_renderer();
    void exponential_renderer();
    void linear_renderer();
    void linear_approx_renderer();
    void dummy_renderer();
    void render_loop();

  public:
    VideoRenderer(int width, int height, int window_size, RenderAlgorithm algorithm,
                  std::shared_ptr<FrameQueue> input_queue,
                  std::shared_ptr<FrameQueue> output_queue);
    ~VideoRenderer();

    void start();
    void wait();
    void print_stats() const;
};

void launch_max_kernel(uint8_t* d_output, const uint8_t* d_input, uint8_t* d_max_frame,
                       float decay_factor, int width, int height, int pitch, cudaStream_t stream);

void launch_average_kernel(uint8_t* d_output, const uint8_t** d_window, int window_count, int width,
                           int height, int pitch, cudaStream_t stream);

void launch_exponential_kernel(uint8_t* d_output, const uint8_t* d_input, uint8_t* d_acc_frame,
                               float exp_factor, int width, int height, int pitch,
                               cudaStream_t stream);

void launch_linear_kernel(uint8_t* d_output, const uint8_t** d_window, const float* d_weights,
                          int window_count, int width, int height, int pitch, cudaStream_t stream);

void launch_linear_approx_kernel(uint8_t* d_output, const uint8_t* d_input, uint8_t* d_y_frame,
                                 float factor, float L, int width, int height, int pitch,
                                 cudaStream_t stream);

void launch_copy_kernel(uint8_t* d_output, const uint8_t* d_input, int width, int height, int pitch,
                        cudaStream_t stream);
