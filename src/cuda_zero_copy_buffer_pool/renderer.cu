#include "renderer.h"
#include "utils/cuda_utils.h"
#include <iostream>

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

__global__ void linear_approx_kernel_single(uint8_t* output, const uint8_t* input, float* y_frame,
                                            float factor, float L, int width, int height,
                                            int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        float y_val = y_frame[idx];
        float input_val = input[idx] / 255.0f;
        float temp = (L + 1.0f) - ((1.0f + L) - y_val) * factor;
        temp = fmaxf(temp, 0.0f);
        y_val = fmaxf(temp, input_val);
        y_frame[idx] = y_val;
        output[idx] = (uint8_t)(y_val * 255.0f);
    }
}

__global__ void linear_approx_kernel_dual(uint8_t* output, const uint8_t* input, float* y_frame,
                                          float factor, float L, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x * 2;
        for (int c = 0; c < 2; c++) {
            float y_val = y_frame[idx + c];
            float input_val = input[idx + c] / 255.0f;
            float temp = (L + 1.0f) - ((1.0f + L) - y_val) * factor;
            temp = fmaxf(temp, 0.0f);
            y_val = fmaxf(temp, input_val);
            y_frame[idx + c] = y_val;
            output[idx + c] = (uint8_t)(y_val * 255.0f);
        }
    }
}

void launch_max_kernel(uint8_t* d_output, const uint8_t* d_input, uint8_t* d_max_frame,
                       float decay_factor, int width, int height, int pitch, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    max_kernel_single<<<grid, block, 0, stream>>>(d_output, d_input, d_max_frame, decay_factor,
                                                  width, height, pitch);
}

void launch_linear_approx_kernel(uint8_t* d_output, const uint8_t* d_input, float* d_y_frame,
                                 float factor, float L, int width, int height, int pitch,
                                 cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    linear_approx_kernel_single<<<grid, block, 0, stream>>>(d_output, d_input, d_y_frame, factor, L,
                                                            width, height, pitch);
}

VideoRenderer::VideoRenderer(int width, int height, int window_size, RenderAlgorithm algorithm,
                             std::shared_ptr<BufferPool> buffer_pool,
                             std::shared_ptr<FrameRefQueue> input_queue,
                             std::shared_ptr<FrameRefQueue> output_queue)
    : width_(width), height_(height), window_size_(window_size), algorithm_(algorithm),
      buffer_pool_(buffer_pool), input_queue_(input_queue), output_queue_(output_queue),
      d_accum_y_(nullptr), d_accum_uv_(nullptr), d_accum_y_float_(nullptr),
      d_accum_uv_float_(nullptr), running_(false) {
    CUDA_CHECK(cudaStreamCreate(&stream_));

    CUDA_CHECK(cudaMalloc(&d_accum_y_, width_ * height_));
    CUDA_CHECK(cudaMalloc(&d_accum_uv_, width_ * (height_ / 2)));
    CUDA_CHECK(cudaMemset(d_accum_y_, 0, width_ * height_));
    CUDA_CHECK(cudaMemset(d_accum_uv_, 128, width_ * (height_ / 2)));

    CUDA_CHECK(cudaMalloc(&d_accum_y_float_, width_ * height_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum_uv_float_, width_ * (height_ / 2) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_accum_y_float_, 0, width_ * height_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_accum_uv_float_, 0, width_ * (height_ / 2) * sizeof(float)));
}

VideoRenderer::~VideoRenderer() {
    running_ = false;
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
    if (d_accum_y_) {
        cudaFree(d_accum_y_);
    }
    if (d_accum_uv_) {
        cudaFree(d_accum_uv_);
    }
    if (d_accum_y_float_) {
        cudaFree(d_accum_y_float_);
    }
    if (d_accum_uv_float_) {
        cudaFree(d_accum_uv_float_);
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
    case RenderAlgorithm::LINEARAPPROX:
        linear_approx_renderer();
        break;
    }
}

void VideoRenderer::start() {
    running_ = true;
    render_thread_ = std::thread(&VideoRenderer::render_loop, this);
}

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
        std::cout << "Avg render time per frame: " << (total_render_time_us_ / frames_rendered_)
                  << " us\n";
        std::cout << "Avg output queue push time: "
                  << (total_output_queue_push_time_us_ / frames_rendered_) << " us\n";
    }
}

void VideoRenderer::max_renderer() {
    const float DECAY_FACTOR = 0.95f;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (running_) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        FrameRef input_ref = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_ref.is_last)
            break;

        FrameBuffer* input_buf = &buffer_pool_->buffers_[input_ref.buffer_id];

        FrameBuffer* output_buf = buffer_pool_->acquire_buffer();
        output_buf->pts = input_ref.pts;
        output_buf->is_last = false;

        auto render_start = std::chrono::high_resolution_clock::now();

        max_kernel_single<<<grid_y, block, 0, stream_>>>(output_buf->d_y_data, input_buf->d_y_data,
                                                         d_accum_y_, DECAY_FACTOR, width_, height_,
                                                         output_buf->y_pitch);

        max_kernel_dual<<<grid_uv, block, 0, stream_>>>(output_buf->d_uv_data, input_buf->d_uv_data,
                                                        d_accum_uv_, DECAY_FACTOR, width_ / 2,
                                                        height_ / 2, output_buf->uv_pitch);

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        buffer_pool_->release_buffer(input_ref.buffer_id);

        FrameRef output_ref(output_buf->buffer_id, output_buf->pts);
        output_queue_->push(output_ref);
    }

    output_queue_->finish();
}

void VideoRenderer::linear_approx_renderer() {
    const float L = 10.0f;
    const float factor = expf(1.0f / (L * window_size_));
    bool first_frame = true;

    dim3 block(16, 16);
    dim3 grid_y((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    dim3 grid_uv((width_ / 2 + block.x - 1) / block.x, (height_ / 2 + block.y - 1) / block.y);

    while (running_) {
        auto input_pop_start = std::chrono::high_resolution_clock::now();
        FrameRef input_ref = input_queue_->pop();
        auto input_pop_end = std::chrono::high_resolution_clock::now();
        total_input_queue_pop_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(input_pop_end - input_pop_start)
                .count();

        if (input_ref.is_last)
            break;

        FrameBuffer* input_buf = &buffer_pool_->buffers_[input_ref.buffer_id];

        if (first_frame) {
            cudaMemsetAsync(d_accum_y_float_, 0, width_ * height_ * sizeof(float), stream_);
            cudaMemsetAsync(d_accum_uv_float_, 0, width_ * (height_ / 2) * sizeof(float), stream_);
            first_frame = false;
        }

        FrameBuffer* output_buf = buffer_pool_->acquire_buffer();
        output_buf->pts = input_ref.pts;
        output_buf->is_last = false;

        auto render_start = std::chrono::high_resolution_clock::now();

        linear_approx_kernel_single<<<grid_y, block, 0, stream_>>>(
            output_buf->d_y_data, input_buf->d_y_data, d_accum_y_float_, factor, L, width_, height_,
            output_buf->y_pitch);

        linear_approx_kernel_dual<<<grid_uv, block, 0, stream_>>>(
            output_buf->d_uv_data, input_buf->d_uv_data, d_accum_uv_float_, factor, L, width_ / 2,
            height_ / 2, output_buf->uv_pitch);

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        auto render_end = std::chrono::high_resolution_clock::now();
        total_render_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start)
                .count();
        frames_rendered_++;

        buffer_pool_->release_buffer(input_ref.buffer_id);

        FrameRef output_ref(output_buf->buffer_id, output_buf->pts);
        output_queue_->push(output_ref);
    }

    output_queue_->finish();
}
