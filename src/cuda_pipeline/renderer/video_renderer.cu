#include "../utils/cuda_utils.h"
#include "../utils/progress.h"
#include "video_renderer.h"
#include <deque>
#include <iostream>
#include <thread>

__global__ void max_kernel_single(uint8_t* output, const uint8_t* input, uint8_t* max_frame,
                                  float decay_factor, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float decayed = max_frame[idx] * decay_factor;
        float current = input[idx];
        max_frame[idx] = fmaxf(decayed, current);
        output[idx] = max_frame[idx];
    }
}

__global__ void max_kernel_dual(uint8_t* output, const uint8_t* input, uint8_t* max_frame,
                                float decay_factor, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float decayed = max_frame[idx + c] * decay_factor;
            float current = input[idx + c];
            max_frame[idx + c] = fmaxf(decayed, current);
            output[idx + c] = max_frame[idx + c];
        }
    }
}

__global__ void average_kernel_single(uint8_t* output, const uint8_t** window, int window_count,
                                      int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float sum = 0.0f;
        for (int i = 0; i < window_count; i++) {
            sum += window[i][idx];
        }
        output[idx] = (uint8_t)(sum / window_count);
    }
}

__global__ void average_kernel_dual(uint8_t* output, const uint8_t** window, int window_count,
                                    int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float sum = 0.0f;
            for (int i = 0; i < window_count; i++) {
                sum += window[i][idx + c];
            }
            output[idx + c] = (uint8_t)(sum / window_count);
        }
    }
}

__global__ void exponential_kernel_single(uint8_t* output, const uint8_t* input, uint8_t* acc_frame,
                                          float exp_factor, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float acc_val = acc_frame[idx];
        float input_val = input[idx];
        acc_frame[idx] = (1.0f - exp_factor) * acc_val + exp_factor * input_val;
        output[idx] = acc_frame[idx];
    }
}

__global__ void exponential_kernel_dual(uint8_t* output, const uint8_t* input, uint8_t* acc_frame,
                                        float exp_factor, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float acc_val = acc_frame[idx + c];
            float input_val = input[idx + c];
            acc_frame[idx + c] = (1.0f - exp_factor) * acc_val + exp_factor * input_val;
            output[idx + c] = acc_frame[idx + c];
        }
    }
}

__global__ void linear_kernel_single(uint8_t* output, const uint8_t** window, const float* weights,
                                     int window_count, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float max_val = 0.0f;
        for (int i = 0; i < window_count; i++) {
            float weighted = window[i][idx] * weights[i] / 255.0f;
            max_val = fmaxf(max_val, weighted);
        }
        output[idx] = (uint8_t)(max_val * 255.0f);
    }
}

__global__ void linear_kernel_dual(uint8_t* output, const uint8_t** window, const float* weights,
                                   int window_count, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float max_val = 0.0f;
            for (int i = 0; i < window_count; i++) {
                float weighted = window[i][idx + c] * weights[i] / 255.0f;
                max_val = fmaxf(max_val, weighted);
            }
            output[idx + c] = (uint8_t)(max_val * 255.0f);
        }
    }
}

__global__ void linear_approx_kernel_single(uint8_t* output, const uint8_t* input, uint8_t* y_frame,
                                            float factor, float L, int width, int height,
                                            int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float y_val = y_frame[idx] / 255.0f;
        float input_val = input[idx] / 255.0f;
        float temp = (L + 1.0f) - ((1.0f + L) - y_val) * factor;
        temp = fmaxf(temp, 0.0f);
        y_val = fmaxf(temp, input_val);
        y_frame[idx] = (uint8_t)(y_val * 255.0f);
        output[idx] = y_frame[idx];
    }
}

__global__ void linear_approx_kernel_dual(uint8_t* output, const uint8_t* input, uint8_t* y_frame,
                                          float factor, float L, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float y_val = y_frame[idx + c] / 255.0f;
            float input_val = input[idx + c] / 255.0f;
            float temp = (L + 1.0f) - ((1.0f + L) - y_val) * factor;
            temp = fmaxf(temp, 0.0f);
            y_val = fmaxf(temp, input_val);
            y_frame[idx + c] = (uint8_t)(y_val * 255.0f);
            output[idx + c] = y_frame[idx + c];
        }
    }
}

VideoRenderer::VideoRenderer(int width, int height, int window_size, RenderAlgorithm algorithm,
                             std::shared_ptr<FrameQueue> input_queue,
                             std::shared_ptr<FrameQueue> output_queue)
    : width_(width), height_(height), window_size_(window_size), algorithm_(algorithm),
      input_queue_(input_queue), output_queue_(output_queue), d_accumulator_(nullptr) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

VideoRenderer::~VideoRenderer() {
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
    if (d_accumulator_) {
        cudaFree(d_accumulator_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void VideoRenderer::render_loop() {
    switch (algorithm_) {
    case RenderAlgorithm::MAX:
        max_renderer();
        break;
    case RenderAlgorithm::AVERAGE:
        average_renderer();
        break;
    case RenderAlgorithm::EXPONENTIAL:
        exponential_renderer();
        break;
    case RenderAlgorithm::LINEAR:
        linear_renderer();
        break;
    case RenderAlgorithm::LINEARAPPROX:
        linear_approx_renderer();
        break;
    case RenderAlgorithm::DUMMY:
        dummy_renderer();
        break;
    }
}

void VideoRenderer::start() { render_thread_ = std::thread(&VideoRenderer::render_loop, this); }

void VideoRenderer::wait() {
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
}

void VideoRenderer::print_stats() const {
    std::cout << "\n=== Renderer Statistics ===\n";
    std::cout << "Frames rendered: " << frames_rendered_ << "\n";
    if (frames_rendered_ > 0) {
        std::cout << "Avg input queue pop time: "
                  << (total_input_queue_pop_time_us_ / frames_rendered_) << " us\n";
        std::cout << "Avg memory alloc time per frame: "
                  << (total_memory_alloc_time_us_ / frames_rendered_) << " us\n";
        std::cout << "Avg render time per frame: " << (total_render_time_us_ / frames_rendered_)
                  << " us\n";
        std::cout << "Avg memory free time per frame: "
                  << (total_memory_free_time_us_ / frames_rendered_) << " us\n";
        std::cout << "Avg output queue push time: "
                  << (total_output_queue_push_time_us_ / frames_rendered_) << " us\n";
    }
}

void VideoRenderer::max_renderer() {
    const float DECAY_FACTOR = 0.95f;
    uint8_t* d_max_y = nullptr;
    uint8_t* d_max_uv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_max_y, width_ * height_));
    CUDA_CHECK(cudaMalloc(&d_max_uv, width_ * (height_ / 2)));
    CUDA_CHECK(cudaMemset(d_max_y, 0, width_ * height_));
    CUDA_CHECK(cudaMemset(d_max_uv, 0, width_ * (height_ / 2)));

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        auto alloc_start = std::chrono::high_resolution_clock::now();
        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = width_;
        output_frame.uv_pitch = width_;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, width_ * (height_ / 2)));
        auto alloc_end = std::chrono::high_resolution_clock::now();
        total_memory_alloc_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

        auto render_start = std::chrono::high_resolution_clock::now();
        max_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_frame.d_y_data, input_frame.d_y_data, d_max_y, DECAY_FACTOR, width_, height_,
            output_frame.y_pitch);
        max_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_frame.d_uv_data, input_frame.d_uv_data, d_max_uv, DECAY_FACTOR, width_ / 2,
            height_ / 2, output_frame.uv_pitch);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();

        auto free_start = std::chrono::high_resolution_clock::now();
        cudaFree(input_frame.d_y_data);
        cudaFree(input_frame.d_uv_data);
        auto free_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start).count();
    }

    cudaFree(d_max_y);
    cudaFree(d_max_uv);
    output_queue_->finish();
}

void VideoRenderer::average_renderer() {
    struct FramePair {
        uint8_t* d_y_data;
        uint8_t* d_uv_data;
    };
    std::deque<FramePair> window;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        auto free_start = std::chrono::high_resolution_clock::now();
        FramePair pair = {input_frame.d_y_data, input_frame.d_uv_data};
        window.push_back(pair);
        if ((int)window.size() > window_size_) {
            cudaFree(window.front().d_y_data);
            cudaFree(window.front().d_uv_data);
            window.pop_front();
        }
        auto free_end = std::chrono::high_resolution_clock::now();
        if ((int)window.size() > window_size_) {
            total_memory_free_time_us_ +=
                std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start)
                    .count();
        }

        auto alloc_start = std::chrono::high_resolution_clock::now();
        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = width_;
        output_frame.uv_pitch = width_;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, width_ * (height_ / 2)));

        std::vector<uint8_t*> y_ptrs, uv_ptrs;
        for (const auto& fp : window) {
            y_ptrs.push_back(fp.d_y_data);
            uv_ptrs.push_back(fp.d_uv_data);
        }
        uint8_t** d_y_ptrs = nullptr;
        uint8_t** d_uv_ptrs = nullptr;
        CUDA_CHECK(cudaMalloc(&d_y_ptrs, y_ptrs.size() * sizeof(uint8_t*)));
        CUDA_CHECK(cudaMalloc(&d_uv_ptrs, uv_ptrs.size() * sizeof(uint8_t*)));
        CUDA_CHECK(cudaMemcpyAsync(d_y_ptrs, y_ptrs.data(), y_ptrs.size() * sizeof(uint8_t*),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_uv_ptrs, uv_ptrs.data(), uv_ptrs.size() * sizeof(uint8_t*),
                                   cudaMemcpyHostToDevice, stream_));
        auto alloc_end = std::chrono::high_resolution_clock::now();
        total_memory_alloc_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

        auto render_start = std::chrono::high_resolution_clock::now();
        average_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_frame.d_y_data, (const uint8_t**)d_y_ptrs, window.size(), width_, height_,
            output_frame.y_pitch);
        average_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_frame.d_uv_data, (const uint8_t**)d_uv_ptrs, window.size(), width_ / 2,
            height_ / 2, output_frame.uv_pitch);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        auto free2_start = std::chrono::high_resolution_clock::now();
        cudaFree(d_y_ptrs);
        cudaFree(d_uv_ptrs);
        auto free2_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free2_end - free2_start).count();

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();
    }

    for (const auto& fp : window) {
        cudaFree(fp.d_y_data);
        cudaFree(fp.d_uv_data);
    }
    output_queue_->finish();
}

void VideoRenderer::exponential_renderer() {
    const float EXP_FACTOR = 0.05f;
    uint8_t* d_acc_y = nullptr;
    uint8_t* d_acc_uv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_acc_y, width_ * height_));
    CUDA_CHECK(cudaMalloc(&d_acc_uv, width_ * (height_ / 2)));
    CUDA_CHECK(cudaMemset(d_acc_y, 0, width_ * height_));
    CUDA_CHECK(cudaMemset(d_acc_uv, 0, width_ * (height_ / 2)));
    bool first_frame = true;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        if (first_frame) {
            CUDA_CHECK(cudaMemcpy(d_acc_y, input_frame.d_y_data, width_ * height_,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_acc_uv, input_frame.d_uv_data, width_ * (height_ / 2),
                                  cudaMemcpyDeviceToDevice));
            first_frame = false;
        }

        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = width_;
        output_frame.uv_pitch = width_;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, width_ * (height_ / 2)));

        auto render_start = std::chrono::high_resolution_clock::now();
        exponential_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_frame.d_y_data, input_frame.d_y_data, d_acc_y, EXP_FACTOR, width_, height_,
            output_frame.y_pitch);
        exponential_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_frame.d_uv_data, input_frame.d_uv_data, d_acc_uv, EXP_FACTOR, width_ / 2,
            height_ / 2, output_frame.uv_pitch);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();

        auto free_start = std::chrono::high_resolution_clock::now();
        cudaFree(input_frame.d_y_data);
        cudaFree(input_frame.d_uv_data);
        auto free_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start).count();
    }

    cudaFree(d_acc_y);
    cudaFree(d_acc_uv);
    output_queue_->finish();
}

void VideoRenderer::linear_renderer() {
    struct FramePair {
        uint8_t* d_y_data;
        uint8_t* d_uv_data;
    };
    std::deque<FramePair> window;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        auto free_start = std::chrono::high_resolution_clock::now();
        FramePair pair = {input_frame.d_y_data, input_frame.d_uv_data};
        window.push_front(pair);
        if ((int)window.size() > window_size_) {
            cudaFree(window.back().d_y_data);
            cudaFree(window.back().d_uv_data);
            window.pop_back();
        }
        auto free_end = std::chrono::high_resolution_clock::now();
        if ((int)window.size() > window_size_) {
            total_memory_free_time_us_ +=
                std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start)
                    .count();
        }

        auto alloc_start = std::chrono::high_resolution_clock::now();
        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = width_;
        output_frame.uv_pitch = width_;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, width_ * (height_ / 2)));

        std::vector<float> weights(window.size());
        for (size_t i = 0; i < window.size(); i++) {
            weights[i] = (float)(window_size_ - i) / window_size_;
        }

        std::vector<uint8_t*> y_ptrs, uv_ptrs;
        for (const auto& fp : window) {
            y_ptrs.push_back(fp.d_y_data);
            uv_ptrs.push_back(fp.d_uv_data);
        }
        uint8_t** d_y_ptrs = nullptr;
        uint8_t** d_uv_ptrs = nullptr;
        float* d_weights = nullptr;
        CUDA_CHECK(cudaMalloc(&d_y_ptrs, y_ptrs.size() * sizeof(uint8_t*)));
        CUDA_CHECK(cudaMalloc(&d_uv_ptrs, uv_ptrs.size() * sizeof(uint8_t*)));
        CUDA_CHECK(cudaMalloc(&d_weights, weights.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_y_ptrs, y_ptrs.data(), y_ptrs.size() * sizeof(uint8_t*),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_uv_ptrs, uv_ptrs.data(), uv_ptrs.size() * sizeof(uint8_t*),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_weights, weights.data(), window.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        auto alloc_end = std::chrono::high_resolution_clock::now();
        total_memory_alloc_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

        auto render_start = std::chrono::high_resolution_clock::now();
        linear_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_frame.d_y_data, (const uint8_t**)d_y_ptrs, d_weights, window.size(), width_,
            height_, output_frame.y_pitch);
        linear_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_frame.d_uv_data, (const uint8_t**)d_uv_ptrs, d_weights, window.size(),
            width_ / 2, height_ / 2, output_frame.uv_pitch);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        auto free2_start = std::chrono::high_resolution_clock::now();
        cudaFree(d_y_ptrs);
        cudaFree(d_uv_ptrs);
        cudaFree(d_weights);
        auto free2_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free2_end - free2_start).count();

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();
    }

    for (const auto& fp : window) {
        cudaFree(fp.d_y_data);
        cudaFree(fp.d_uv_data);
    }
    output_queue_->finish();
}

void VideoRenderer::linear_approx_renderer() {
    const float L = 10.0f;
    const float factor = expf(1.0f / (L * window_size_));
    uint8_t* d_acc_y = nullptr;
    uint8_t* d_acc_uv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_acc_y, width_ * height_));
    CUDA_CHECK(cudaMalloc(&d_acc_uv, width_ * (height_ / 2)));
    CUDA_CHECK(cudaMemset(d_acc_y, 0, width_ * height_));
    CUDA_CHECK(cudaMemset(d_acc_uv, 0, width_ * (height_ / 2)));
    bool first_frame = true;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        if (first_frame) {
            CUDA_CHECK(cudaMemcpy(d_acc_y, input_frame.d_y_data, width_ * height_,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_acc_uv, input_frame.d_uv_data, width_ * (height_ / 2),
                                  cudaMemcpyDeviceToDevice));
            first_frame = false;
        }

        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = width_;
        output_frame.uv_pitch = width_;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, width_ * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, width_ * (height_ / 2)));

        auto render_start = std::chrono::high_resolution_clock::now();
        linear_approx_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_frame.d_y_data, input_frame.d_y_data, d_acc_y, factor, L, width_, height_,
            output_frame.y_pitch);
        linear_approx_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_frame.d_uv_data, input_frame.d_uv_data, d_acc_uv, factor, L, width_ / 2,
            height_ / 2, output_frame.uv_pitch);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();

        auto free_start = std::chrono::high_resolution_clock::now();
        cudaFree(input_frame.d_y_data);
        cudaFree(input_frame.d_uv_data);
        auto free_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start).count();
    }

    cudaFree(d_acc_y);
    cudaFree(d_acc_uv);
    output_queue_->finish();
}

void VideoRenderer::dummy_renderer() {
    while (true) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        Frame input_frame = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_frame.is_last)
            break;

        auto alloc_start = std::chrono::high_resolution_clock::now();
        Frame output_frame;
        output_frame.width = width_;
        output_frame.height = height_;
        output_frame.y_pitch = input_frame.y_pitch;
        output_frame.uv_pitch = input_frame.uv_pitch;
        output_frame.pts = input_frame.pts;
        CUDA_CHECK(cudaMalloc(&output_frame.d_y_data, output_frame.y_pitch * height_));
        CUDA_CHECK(cudaMalloc(&output_frame.d_uv_data, output_frame.uv_pitch * (height_ / 2)));

        CUDA_CHECK(cudaMemcpyAsync(output_frame.d_y_data, input_frame.d_y_data,
                                   output_frame.y_pitch * height_, cudaMemcpyDeviceToDevice,
                                   stream_));
        CUDA_CHECK(cudaMemcpyAsync(output_frame.d_uv_data, input_frame.d_uv_data,
                                   output_frame.uv_pitch * (height_ / 2), cudaMemcpyDeviceToDevice,
                                   stream_));
        auto alloc_end = std::chrono::high_resolution_clock::now();
        total_memory_alloc_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

        auto output_push_start = std::chrono::high_resolution_clock::now();
        output_queue_->push(output_frame);
        auto output_push_end = std::chrono::high_resolution_clock::now();
        total_output_queue_push_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                                                output_push_end - output_push_start)
                                                .count();

        auto free_start = std::chrono::high_resolution_clock::now();
        cudaFree(input_frame.d_y_data);
        cudaFree(input_frame.d_uv_data);
        auto free_end = std::chrono::high_resolution_clock::now();
        total_memory_free_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(free_end - free_start).count();
    }

    output_queue_->finish();
}
