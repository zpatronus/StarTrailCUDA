#include "video_frame_reader.h"
#include "../utils/utils.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void VideoFrameReader::print_video_metadata() {
    if (fmt_ctx_ && video_stream_idx_ >= 0) {
        AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
        AVRational frame_rate = video_stream->avg_frame_rate;
        double fps = (double)frame_rate.num / frame_rate.den;
        int64_t frame_count = video_stream->nb_frames;

        std::cout << "  Resolution : " << width_ << " x " << height_ << "\n"
                  << "  FPS        : " << fps << "\n"
                  << "  FrameCount : " << frame_count << "\n"
                  << "  Duration   : " << (fps > 0 ? frame_count / fps : 0) << " seconds\n";
    }
}

VideoFrameReader::VideoFrameReader(const std::string& videoPath, const unsigned int frameStep)
    : fmt_ctx_(nullptr), codec_ctx_(nullptr), video_stream_idx_(-1), sws_ctx_(nullptr), width_(0),
      height_(0), current_frame_index_(0), current_read_index_(0), total_frames_(0),
      frame_step_(frameStep) {

    if (avformat_open_input(&fmt_ctx_, videoPath.c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("Failed to open video: " + videoPath);
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not find stream information");
    }

    for (unsigned int i = 0; i < fmt_ctx_->nb_streams; i++) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            break;
        }
    }
    if (video_stream_idx_ == -1) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not find video stream");
    }

    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codecpar = video_stream->codecpar;

    const AVCodec* decoder = nullptr;
    std::string hw_decoder_name;

    switch (codecpar->codec_id) {
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
        break;
    }

    bool use_hardware = false;
    if (!hw_decoder_name.empty()) {
        decoder = avcodec_find_decoder_by_name(hw_decoder_name.c_str());
        if (decoder) {
            AVCodecContext* test_ctx = avcodec_alloc_context3(decoder);
            if (test_ctx) {
                if (avcodec_parameters_to_context(test_ctx, codecpar) >= 0) {
                    if (avcodec_open2(test_ctx, decoder, nullptr) >= 0) {
                        use_hardware = true;
                        std::cout << "  Using hardware decoder: " << hw_decoder_name << std::endl;
                        avcodec_free_context(&test_ctx);
                    } else {
                        std::cerr << "  ERROR: Hardware decoder " << hw_decoder_name
                                  << " not supported." << std::endl;
                        std::cerr << "  This tool requires NVENC hardware support." << std::endl;
                        std::cerr << "  Please use the baseline version for software decoding."
                                  << std::endl;
                        avcodec_free_context(&test_ctx);
                        avformat_close_input(&fmt_ctx_);
                        throw std::runtime_error("Hardware decoder required but not available");
                    }
                } else {
                    avcodec_free_context(&test_ctx);
                    decoder = nullptr;
                }
            }
        } else {
            std::cerr << "  ERROR: Hardware decoder " << hw_decoder_name << " not found."
                      << std::endl;
            std::cerr << "  This tool requires NVENC hardware support." << std::endl;
            std::cerr << "  Please use the baseline version for software decoding." << std::endl;
            avformat_close_input(&fmt_ctx_);
            throw std::runtime_error("Hardware decoder required but not available");
        }
    } else {
        std::cerr << "  ERROR: No hardware decoder available for this codec." << std::endl;
        std::cerr << "  This tool requires NVENC hardware support." << std::endl;
        std::cerr << "  Please use the baseline version for software decoding." << std::endl;
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Hardware decoder required but not available");
    }

    if (!use_hardware || !decoder) {
        std::cerr << "  ERROR: Hardware decoder initialization failed." << std::endl;
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Hardware decoder required but not available");
    }

    if (!decoder) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not find decoder");
    }

    codec_ctx_ = avcodec_alloc_context3(decoder);
    if (!codec_ctx_) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not allocate decoder context");
    }

    if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not copy codec parameters");
    }

    if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not open decoder");
    }

    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;

    sws_ctx_ = sws_getContext(width_, height_, codec_ctx_->pix_fmt, width_, height_,
                              AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("Could not create SwsContext");
    }

    total_frames_ = static_cast<int>(std::ceil(video_stream->nb_frames / (double)frame_step_));

    print_video_metadata();
}

VideoFrameReader::~VideoFrameReader() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
    }
}

FrameSize VideoFrameReader::getFrameSize() { return FrameSize(width_, height_); }

std::optional<FrameData> VideoFrameReader::nextFrame() {
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    if (!packet || !frame) {
        if (packet)
            av_packet_free(&packet);
        if (frame)
            av_frame_free(&frame);
        return std::nullopt;
    }

    while (av_read_frame(fmt_ctx_, packet) >= 0) {
        if (packet->stream_index == video_stream_idx_) {
            current_frame_index_++;
            if ((current_frame_index_ - 1) % frame_step_ == 0) {
                int ret = avcodec_send_packet(codec_ctx_, packet);
                if (ret < 0) {
                    av_packet_unref(packet);
                    continue;
                }

                ret = avcodec_receive_frame(codec_ctx_, frame);
                if (ret >= 0) {
                    FrameData rgb_frame(width_, height_, 3);
                    uint8_t* dest[1] = {rgb_frame.getRawData()};
                    int dest_linesize[1] = {width_ * 3};

                    sws_scale(sws_ctx_, frame->data, frame->linesize, 0, height_, dest,
                              dest_linesize);

                    current_read_index_++;
                    print_progress(current_read_index_, total_frames_);

                    av_frame_unref(frame);
                    av_packet_unref(packet);
                    av_frame_free(&frame);
                    av_packet_free(&packet);

                    return rgb_frame;
                }
            }
            av_packet_unref(packet);
        } else {
            av_packet_unref(packet);
        }
    }

    av_frame_free(&frame);
    av_packet_free(&packet);
    return std::nullopt;
}

bool VideoFrameReader::hasNextFrame() { return current_read_index_ < total_frames_; }

void VideoFrameReader::reset() {
    av_seek_frame(fmt_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(codec_ctx_);
    current_frame_index_ = 0;
    current_read_index_ = 0;
}

int VideoFrameReader::getTotalFrames() const { return total_frames_; }

int VideoFrameReader::getCurrentFrameIndex() const { return current_read_index_; }
