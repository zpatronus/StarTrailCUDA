#include "decoder.h"
#include "utils/cuda_utils.h"
#include "utils/progress.h"
#include <iostream>

VideoDecoder::VideoDecoder(const std::string& input_path, int frame_step,
                           std::shared_ptr<BufferPool> buffer_pool,
                           std::shared_ptr<FrameRefQueue> output_queue)
    : input_path_(input_path), frame_step_(frame_step), buffer_pool_(buffer_pool),
      output_queue_(output_queue), fmt_ctx_(nullptr), codec_ctx_(nullptr), hw_device_ctx_(nullptr),
      video_stream_idx_(-1), width_(0), height_(0), fps_(30), total_frames_(0), running_(false),
      packet_reader_running_(false), end_of_stream_(false) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

VideoDecoder::~VideoDecoder() {
    running_ = false;
    packet_reader_running_ = false;
    packet_cv_.notify_all();

    if (packet_reader_thread_.joinable()) {
        packet_reader_thread_.join();
    }
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

    std::lock_guard<std::mutex> lock(packet_mutex_);
    while (!packet_queue_.empty()) {
        AVPacket* packet = packet_queue_.front();
        packet_queue_.pop();
        av_packet_free(&packet);
    }

    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void VideoDecoder::start() {
    if (avformat_open_input(&fmt_ctx_, input_path_.c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("Could not open input file: " + input_path_);
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        throw std::runtime_error("Could not find stream information");
    }

    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        throw std::runtime_error("Could not find video stream");
    }

    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
    width_ = video_stream->codecpar->width;
    height_ = video_stream->codecpar->height;
    total_frames_ = video_stream->nb_frames;

    if (video_stream->avg_frame_rate.den != 0) {
        fps_ = video_stream->avg_frame_rate.num / video_stream->avg_frame_rate.den;
    }

    const AVCodec* decoder = nullptr;
    const char* hw_decoder_name = nullptr;

    switch (video_stream->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
        hw_decoder_name = "h264_cuvid";
        break;
    case AV_CODEC_ID_HEVC:
        hw_decoder_name = "hevc_cuvid";
        break;
    case AV_CODEC_ID_VP9:
        hw_decoder_name = "vp9_cuvid";
        break;
    case AV_CODEC_ID_AV1:
        hw_decoder_name = "av1_cuvid";
        break;
    default:
        hw_decoder_name = nullptr;
        break;
    }

    if (!hw_decoder_name) {
        std::cerr << "ERROR: No hardware decoder available for this codec.\n";
        throw std::runtime_error("Hardware decoder required but codec not supported");
    }

    decoder = avcodec_find_decoder_by_name(hw_decoder_name);
    if (!decoder) {
        std::cerr << "ERROR: Hardware decoder " << hw_decoder_name << " not found.\n";
        throw std::runtime_error("Hardware decoder required but not available");
    }

    std::cout << "Using hardware decoder: " << hw_decoder_name << std::endl;

    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        throw std::runtime_error("Failed to create CUDA device context");
    }

    codec_ctx_ = avcodec_alloc_context3(decoder);
    if (!codec_ctx_) {
        throw std::runtime_error("Could not allocate codec context");
    }

    if (avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar) < 0) {
        throw std::runtime_error("Could not copy codec parameters");
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    if (!codec_ctx_->hw_device_ctx) {
        throw std::runtime_error("Failed to reference hardware device context");
    }

    if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    std::cout << "Video Information:\n";
    std::cout << "  Resolution: " << width_ << "x" << height_ << "\n";
    std::cout << "  FPS: " << fps_ << "\n";
    std::cout << "  Frame Step: " << frame_step_ << "\n\n";

    packet_reader_running_ = true;
    packet_reader_thread_ = std::thread(&VideoDecoder::packet_reader_loop, this);

    running_ = true;
    decode_thread_ = std::thread(&VideoDecoder::decode_loop, this);
}

void VideoDecoder::packet_reader_loop() {
    AVPacket* packet = av_packet_alloc();

    while (packet_reader_running_) {
        auto read_start = std::chrono::high_resolution_clock::now();
        int ret = av_read_frame(fmt_ctx_, packet);
        auto read_end = std::chrono::high_resolution_clock::now();
        total_read_time_us_ +=
            std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();

        if (ret < 0) {
            av_packet_free(&packet);
            std::lock_guard<std::mutex> lock(packet_mutex_);
            end_of_stream_ = true;
            packet_cv_.notify_all();
            break;
        }

        if (packet->stream_index == video_stream_idx_) {
            packets_read_++;
            AVPacket* packet_copy = av_packet_clone(packet);

            {
                std::unique_lock<std::mutex> lock(packet_mutex_);
                packet_cv_.wait(lock, [this] {
                    return packet_queue_.size() < MAX_PACKET_QUEUE_SIZE || !packet_reader_running_;
                });

                if (!packet_reader_running_) {
                    av_packet_free(&packet_copy);
                    break;
                }

                packet_queue_.push(packet_copy);
                packet_cv_.notify_one();
            }
        }
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
}

void VideoDecoder::decode_loop() {
    int frame_count = 0;
    AVFrame* decoded_frame = av_frame_alloc();

    while (running_) {
        AVPacket* packet = nullptr;

        {
            std::unique_lock<std::mutex> lock(packet_mutex_);
            packet_cv_.wait(
                lock, [this] { return !packet_queue_.empty() || end_of_stream_ || !running_; });

            if (!running_) {
                break;
            }

            if (packet_queue_.empty()) {
                if (end_of_stream_) {
                    break;
                }
                continue;
            }

            packet = packet_queue_.front();
            packet_queue_.pop();
            packet_cv_.notify_one();
        }

        auto decode_start = std::chrono::high_resolution_clock::now();
        if (avcodec_send_packet(codec_ctx_, packet) >= 0) {
            while (avcodec_receive_frame(codec_ctx_, decoded_frame) >= 0) {
                auto decode_end = std::chrono::high_resolution_clock::now();
                total_decode_time_us_ +=
                    std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start)
                        .count();
                frames_decoded_++;

                bool should_process = (frame_count % frame_step_ == 0);

                if (should_process) {
                    AVFrame* gpu_frame = decoded_frame;

                    FrameBuffer* frame_buf = buffer_pool_->acquire_buffer();
                    frame_buf->pts = gpu_frame->pts;
                    frame_buf->is_last = false;

                    auto gpu_copy_start = std::chrono::high_resolution_clock::now();

                    uint8_t* src_nv12 = (uint8_t*)gpu_frame->data[0];
                    size_t src_pitch = gpu_frame->linesize[0];

                    CUDA_CHECK(cudaMemcpy2DAsync(frame_buf->d_y_data, frame_buf->y_pitch, src_nv12,
                                                 src_pitch, width_, height_,
                                                 cudaMemcpyDeviceToDevice, stream_));

                    CUDA_CHECK(cudaMemcpy2DAsync(frame_buf->d_uv_data, frame_buf->uv_pitch,
                                                 src_nv12 + src_pitch * height_, src_pitch, width_,
                                                 height_ / 2, cudaMemcpyDeviceToDevice, stream_));

                    auto gpu_copy_end = std::chrono::high_resolution_clock::now();
                    total_gpu_transfer_time_us_ +=
                        std::chrono::duration_cast<std::chrono::microseconds>(gpu_copy_end -
                                                                              gpu_copy_start)
                            .count();

                    auto queue_push_start = std::chrono::high_resolution_clock::now();
                    FrameRef ref(frame_buf->buffer_id, frame_buf->pts);
                    output_queue_->push(ref);
                    auto queue_push_end = std::chrono::high_resolution_clock::now();
                    total_output_queue_push_time_us_ +=
                        std::chrono::duration_cast<std::chrono::microseconds>(queue_push_end -
                                                                              queue_push_start)
                            .count();
                    frames_pushed_++;

                    int expected_total = get_expected_output_frames();
                    print_progress(frames_pushed_, expected_total);
                }

                frame_count++;
                av_frame_unref(decoded_frame);
                decode_start = std::chrono::high_resolution_clock::now();
            }
        }

        av_packet_free(&packet);
    }

    av_frame_free(&decoded_frame);
    output_queue_->finish();
}

void VideoDecoder::wait() {
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }
}

void VideoDecoder::print_stats() const {
    std::cout << "\n=== Decoder Statistics ===\n";
    std::cout << "Packets read: " << packets_read_ << "\n";
    std::cout << "Frames decoded: " << frames_decoded_ << "\n";
    std::cout << "Frames pushed: " << frames_pushed_ << "\n";
    if (packets_read_ > 0) {
        std::cout << "Avg read time per packet: " << (total_read_time_us_ / packets_read_)
                  << " us\n";
    }
    if (frames_decoded_ > 0) {
        std::cout << "Avg decode time per frame: " << (total_decode_time_us_ / frames_decoded_)
                  << " us\n";
    }
    if (frames_pushed_ > 0) {
        std::cout << "Avg GPU copy time per frame: "
                  << (total_gpu_transfer_time_us_ / frames_pushed_) << " us\n";
        std::cout << "Avg output queue push time: "
                  << (total_output_queue_push_time_us_ / frames_pushed_) << " us\n";
    }
}
