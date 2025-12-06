#include "encoder.h"
#include "utils/cuda_utils.h"
#include <iomanip>
#include <iostream>

VideoEncoder::VideoEncoder(const std::string& output_path, int width, int height, int fps,
                           std::shared_ptr<BufferPool> buffer_pool,
                           std::shared_ptr<FrameRefQueue> input_queue)
    : output_path_(output_path), width_(width), height_(height), fps_(fps),
      buffer_pool_(buffer_pool), input_queue_(input_queue), fmt_ctx_(nullptr), codec_ctx_(nullptr),
      av_stream_(nullptr), av_frame_(nullptr), hw_device_ctx_(nullptr), hw_frames_ctx_(nullptr),
      running_(false), write_running_(false), frame_count_(0) {
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
}

VideoEncoder::~VideoEncoder() {
    running_ = false;
    write_running_ = false;
    write_cv_.notify_all();

    if (write_thread_.joinable()) {
        write_thread_.join();
    }
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(write_mutex_);
        while (!write_queue_.empty()) {
            AVPacket* packet = write_queue_.front();
            write_queue_.pop();
            if (packet) {
                av_packet_free(&packet);
            }
        }
    }

    if (av_frame_) {
        av_frame_free(&av_frame_);
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    if (hw_frames_ctx_) {
        av_buffer_unref(&hw_frames_ctx_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        if (fmt_ctx_->pb) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
}

void VideoEncoder::start() {
    avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, output_path_.c_str());
    if (!fmt_ctx_) {
        throw std::runtime_error("Could not create output context");
    }

    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) {
        std::cerr << "ERROR: H.264 NVENC encoder not found.\n";
        throw std::runtime_error("Hardware encoder required but not available");
    }

    std::cout << "Using hardware encoder: h264_nvenc" << std::endl;

    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context for encoding");
    }

    av_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!av_stream_) {
        throw std::runtime_error("Could not create stream");
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        throw std::runtime_error("Could not allocate codec context");
    }

    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->time_base = AVRational{1, fps_};
    codec_ctx_->framerate = AVRational{fps_, 1};
    codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;
    codec_ctx_->sw_pix_fmt = AV_PIX_FMT_NV12;

    hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ctx_) {
        throw std::runtime_error("Failed to create hardware frames context");
    }

    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx_->data;
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width_;
    frames_ctx->height = height_;
    frames_ctx->initial_pool_size = 20;

    if (av_hwframe_ctx_init(hw_frames_ctx_) < 0) {
        throw std::runtime_error("Failed to initialize hardware frames context");
    }

    codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
    if (!codec_ctx_->hw_frames_ctx) {
        throw std::runtime_error("Failed to reference hardware frames context");
    }

    int64_t reference_pixels = 1920 * 1080;
    int64_t actual_pixels = static_cast<int64_t>(width_) * height_;
    codec_ctx_->bit_rate = (actual_pixels * 6000000) / reference_pixels;

    codec_ctx_->gop_size = 12;
    codec_ctx_->max_b_frames = 0;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    av_opt_set(codec_ctx_->priv_data, "preset", "p1", 0);
    av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0);
    av_opt_set(codec_ctx_->priv_data, "rc", "vbr", 0);

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    if (avcodec_parameters_from_context(av_stream_->codecpar, codec_ctx_) < 0) {
        throw std::runtime_error("Could not copy codec parameters");
    }

    av_stream_->time_base = codec_ctx_->time_base;

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx_->pb, output_path_.c_str(), AVIO_FLAG_WRITE) < 0) {
            throw std::runtime_error("Could not open output file");
        }
    }

    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
        throw std::runtime_error("Could not write header");
    }

    av_frame_ = av_frame_alloc();
    if (!av_frame_) {
        throw std::runtime_error("Could not allocate AVFrame");
    }

    running_ = true;
    write_running_ = true;
    write_thread_ = std::thread(&VideoEncoder::write_loop, this);
    encode_thread_ = std::thread(&VideoEncoder::encode_loop, this);
}

void VideoEncoder::write_loop() {
    while (write_running_) {
        AVPacket* packet = nullptr;
        {
            std::unique_lock<std::mutex> lock(write_mutex_);
            write_cv_.wait(lock, [this] { return !write_queue_.empty() || !write_running_; });

            if (!write_running_ && write_queue_.empty())
                break;

            if (!write_queue_.empty()) {
                packet = write_queue_.front();
                write_queue_.pop();
            }
        }

        if (packet) {
            if (packet->data) {
                auto write_start = std::chrono::high_resolution_clock::now();
                av_packet_rescale_ts(packet, codec_ctx_->time_base, av_stream_->time_base);
                packet->stream_index = 0;
                av_interleaved_write_frame(fmt_ctx_, packet);
                auto write_end = std::chrono::high_resolution_clock::now();
                total_write_time_us_ +=
                    std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start)
                        .count();
                packets_written_++;
            }
            av_packet_unref(packet);
            av_packet_free(&packet);
        }
    }
}

void VideoEncoder::encode_loop() {
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

        auto upload_start = std::chrono::high_resolution_clock::now();

        if (av_hwframe_get_buffer(hw_frames_ctx_, av_frame_, 0) < 0) {
            std::cerr << "Failed to allocate hardware frame" << std::endl;
            buffer_pool_->release_buffer(input_ref.buffer_id);
            buffers_released_++;
            continue;
        }

        av_frame_->format = AV_PIX_FMT_CUDA;
        av_frame_->width = width_;
        av_frame_->height = height_;
        av_frame_->pts = frame_count_++;

        uint8_t* dst_nv12 = (uint8_t*)av_frame_->data[0];
        size_t dst_pitch = av_frame_->linesize[0];

        CUDA_CHECK(cudaMemcpy2DAsync(dst_nv12, dst_pitch, input_buf->d_y_data, input_buf->y_pitch,
                                     width_, height_, cudaMemcpyDeviceToDevice, cuda_stream_));

        CUDA_CHECK(cudaMemcpy2DAsync(dst_nv12 + dst_pitch * height_, dst_pitch,
                                     input_buf->d_uv_data, input_buf->uv_pitch, width_, height_ / 2,
                                     cudaMemcpyDeviceToDevice, cuda_stream_));

        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

        auto upload_end = std::chrono::high_resolution_clock::now();
        total_gpu_upload_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(upload_end - upload_start)
                .count();

        buffer_pool_->release_buffer(input_ref.buffer_id);
        buffers_released_++;

        auto encode_start = std::chrono::high_resolution_clock::now();
        int send_ret = avcodec_send_frame(codec_ctx_, av_frame_);
        auto encode_end = std::chrono::high_resolution_clock::now();

        if (send_ret >= 0) {
            total_encode_time_us_ +=
                std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start)
                    .count();

            while (true) {
                AVPacket* packet = av_packet_alloc();
                int ret = avcodec_receive_packet(codec_ctx_, packet);
                if (ret >= 0) {
                    auto packet_queue_start = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(write_mutex_);
                        write_queue_.push(packet);
                        write_cv_.notify_one();
                    }
                    auto packet_queue_end = std::chrono::high_resolution_clock::now();
                    total_packet_queue_push_time_us_ +=
                        std::chrono::duration_cast<std::chrono::microseconds>(packet_queue_end -
                                                                              packet_queue_start)
                            .count();
                } else {
                    av_packet_free(&packet);
                    break;
                }
            }
        }

        frames_encoded_++;
        av_frame_unref(av_frame_);
    }

    avcodec_send_frame(codec_ctx_, nullptr);
    while (true) {
        AVPacket* packet = av_packet_alloc();
        if (avcodec_receive_packet(codec_ctx_, packet) >= 0) {
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                write_queue_.push(packet);
                write_cv_.notify_one();
            }
        } else {
            av_packet_free(&packet);
            break;
        }
    }

    write_running_ = false;
    write_cv_.notify_all();
    if (write_thread_.joinable()) {
        write_thread_.join();
    }

    av_write_trailer(fmt_ctx_);
}

void VideoEncoder::wait() {
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
}

void VideoEncoder::print_stats() const {
    std::cout << "\n=== Encoder Statistics ===\n";
    std::cout << "Frames encoded: " << frames_encoded_ << "\n";
    std::cout << "Packets written: " << packets_written_ << "\n";
    std::cout << "Buffers released: " << buffers_released_ << "\n";
    if (frames_encoded_ > 0) {
        std::cout << "Avg input queue pop time: "
                  << (total_input_queue_pop_time_us_ / frames_encoded_) << " us\n";
        std::cout << "Avg GPU upload time per frame: "
                  << (total_gpu_upload_time_us_ / frames_encoded_) << " us\n";
        std::cout << "Avg encoding time per frame: " << (total_encode_time_us_ / frames_encoded_)
                  << " us\n";
        std::cout << "Avg packet queue push time: "
                  << (total_packet_queue_push_time_us_ / frames_encoded_) << " us\n";
    }
    if (packets_written_ > 0) {
        std::cout << "Avg write time per packet: " << (total_write_time_us_ / packets_written_)
                  << " us\n";
    }
}
