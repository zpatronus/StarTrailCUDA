#include "video_renderer.h"
#include "../frame_reader/frame_reader.h"
#include <deque>
#include <iostream>
#include <queue>

VideoRenderer::VideoRenderer(std::unique_ptr<FrameReader> reader, std::string output_path, int fps,
                             RenderAlgo algorithm, int window_size)
    : frame_reader(std::move(reader)), output_path(output_path), fps(fps), algo(algorithm),
      window_size(window_size), fmt_ctx_(nullptr), codec_ctx_(nullptr), stream_(nullptr),
      sws_ctx_(nullptr), pts_(0) {
    init_encoder();
}

VideoRenderer::~VideoRenderer() { cleanup_encoder(); }

void VideoRenderer::init_encoder() {
    cv::Size frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;

    avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, output_path.c_str());
    if (!fmt_ctx_) {
        throw std::runtime_error("Could not create output context");
    }

    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) {
        std::cerr << "ERROR: H.264 NVENC encoder not found.\n"
                  << "This tool requires NVENC hardware support.\n"
                  << "Please use the baseline version for software encoding." << std::endl;
        throw std::runtime_error("Hardware encoder required but not available");
    }

    AVCodecContext* test_ctx = avcodec_alloc_context3(codec);
    test_ctx->width = width;
    test_ctx->height = height;
    test_ctx->time_base = AVRational{1, fps};
    test_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (avcodec_open2(test_ctx, codec, nullptr) < 0) {
        std::cerr << "ERROR: NVENC encoder not supported on this system.\n"
                  << "This tool requires NVENC hardware support.\n"
                  << "Please use the baseline version for software encoding." << std::endl;
        avcodec_free_context(&test_ctx);
        throw std::runtime_error("Hardware encoder required but not available");
    }
    avcodec_free_context(&test_ctx);
    std::cout << "Using hardware encoder: h264_nvenc" << std::endl;

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
        throw std::runtime_error("Could not create output stream");
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        throw std::runtime_error("Could not allocate codec context");
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = AVRational{1, fps};
    codec_ctx_->framerate = AVRational{fps, 1};
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;

    int64_t reference_pixels = 1920 * 1080;
    int64_t actual_pixels = static_cast<int64_t>(width) * height;
    codec_ctx_->bit_rate = (actual_pixels * 6000000) / reference_pixels;

    codec_ctx_->gop_size = 12;
    codec_ctx_->max_b_frames = 2;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);
    av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0);
    av_opt_set(codec_ctx_->priv_data, "rc", "vbr", 0);

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
        throw std::runtime_error("Could not copy codec parameters");
    }

    stream_->time_base = codec_ctx_->time_base;

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx_->pb, output_path.c_str(), AVIO_FLAG_WRITE) < 0) {
            throw std::runtime_error("Could not open output file");
        }
    }

    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
        throw std::runtime_error("Could not write header");
    }

    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height, AV_PIX_FMT_YUV420P,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        throw std::runtime_error("Could not create SwsContext for encoding");
    }
}

void VideoRenderer::encode_frame(const cv::Mat& frame) {
    AVFrame* av_frame = av_frame_alloc();
    av_frame->format = codec_ctx_->pix_fmt;
    av_frame->width = codec_ctx_->width;
    av_frame->height = codec_ctx_->height;

    if (av_frame_get_buffer(av_frame, 0) < 0) {
        av_frame_free(&av_frame);
        throw std::runtime_error("Could not allocate frame buffer");
    }

    const uint8_t* src[1] = {frame.data};
    int src_linesize[1] = {static_cast<int>(frame.step[0])};
    sws_scale(sws_ctx_, src, src_linesize, 0, codec_ctx_->height, av_frame->data,
              av_frame->linesize);

    av_frame->pts = pts_++;

    if (avcodec_send_frame(codec_ctx_, av_frame) >= 0) {
        AVPacket* packet = av_packet_alloc();
        while (avcodec_receive_packet(codec_ctx_, packet) >= 0) {
            av_packet_rescale_ts(packet, codec_ctx_->time_base, stream_->time_base);
            packet->stream_index = 0;
            av_interleaved_write_frame(fmt_ctx_, packet);
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
    }

    av_frame_free(&av_frame);
}

void VideoRenderer::flush_encoder() {
    avcodec_send_frame(codec_ctx_, nullptr);
    AVPacket* packet = av_packet_alloc();
    while (avcodec_receive_packet(codec_ctx_, packet) >= 0) {
        av_packet_rescale_ts(packet, codec_ctx_->time_base, stream_->time_base);
        packet->stream_index = 0;
        av_interleaved_write_frame(fmt_ctx_, packet);
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    av_write_trailer(fmt_ctx_);
}

void VideoRenderer::cleanup_encoder() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
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
}

void VideoRenderer::averageRenderer() {
    cv::Mat acc;
    std::queue<cv::Mat> render_window;
    const unsigned int RENDER_WINDOW_SIZE = window_size;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            render_window.push(frame.value());
            if (acc.empty()) {
                frame.value().convertTo(acc, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                acc += frame_f;
                if (render_window.size() > RENDER_WINDOW_SIZE) {
                    auto frame_rm = render_window.front();
                    render_window.pop();
                    cv::Mat frame_rm_f;
                    frame_rm.convertTo(frame_rm_f, CV_32FC3);
                    acc -= frame_rm_f;
                }
            }
            cv::Mat acc_write = acc / render_window.size();
            acc_write.convertTo(acc_write, CV_8UC3);
            encode_frame(acc_write);
        } else {
            break;
        }
    }
}

void VideoRenderer::maxRenderer() {
    cv::Mat maxFrame;
    const double DECAY_FACTOR = 0.95;
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            if (maxFrame.empty()) {
                frame.value().convertTo(maxFrame, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                maxFrame = cv::max(maxFrame * DECAY_FACTOR, frame_f);
            }
            cv::Mat output_frame;
            maxFrame.convertTo(output_frame, CV_8UC3);
            encode_frame(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::exponentialRenderer() {
    cv::Mat accFrame;
    const double EXP_FACTOR = 0.05;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            if (accFrame.empty()) {
                frame.value().convertTo(accFrame, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                accFrame = (1 - EXP_FACTOR) * accFrame + EXP_FACTOR * frame_f;
            }
            cv::Mat output_frame;
            accFrame.convertTo(output_frame, CV_8UC3);
            encode_frame(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::linearRenderer() {
    std::deque<cv::Mat> render_window;
    const int RENDER_WINDOW_SIZE = window_size;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            const cv::Mat& frame8u = frame.value(); // CV_8UC3
            render_window.push_front(frame8u);
            if (static_cast<int>(render_window.size()) > RENDER_WINDOW_SIZE) {
                render_window.pop_back();
            }

            cv::Mat accFrame(frame8u.rows, frame8u.cols, CV_32FC3, cv::Scalar(0, 0, 0));

            int windowSize = static_cast<int>(render_window.size());

            for (int i = 0; i < windowSize; ++i) {
                double weight = double(RENDER_WINDOW_SIZE - i) / double(RENDER_WINDOW_SIZE);

                cv::Mat curr32f;
                render_window[i].convertTo(curr32f, CV_32FC3, weight / 255.0);

                cv::max(accFrame, curr32f, accFrame);
            }

            cv::Mat output_frame;
            accFrame.convertTo(output_frame, CV_8UC3, 255.0); // [0,1] -> [0,255]
            encode_frame(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::linearApproxRenderer() {
    const int N = window_size;
    const double L = 10.0;
    const double factor = exp(1.0 / (L * N));

    cv::Mat yFrame;
    while (true) {
        auto frameOpt = frame_reader->nextFrame();
        if (frameOpt.has_value()) {
            cv::Mat frame32f;
            frameOpt.value().convertTo(frame32f, CV_32FC3);
            frame32f /= 255.0;

            if (yFrame.empty()) {
                yFrame = frame32f.clone();
            } else {
                cv::Mat temp = (L + 1) - ((1 + L) - yFrame) * factor;
                yFrame = cv::max(temp, cv::Mat::zeros(temp.size(), temp.type()));
                yFrame = cv::max(yFrame, frame32f);
            }

            cv::Mat output_frame;
            yFrame.convertTo(output_frame, CV_8UC3, 255.0);
            encode_frame(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::dummyRenderer() {
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            encode_frame(frame.value());
        } else {
            break;
        }
    }
}

void VideoRenderer::render() {
    switch (algo) {
    case AVGRAGE:
        averageRenderer();
        break;
    case MAX:
        maxRenderer();
        break;
    case EXPONENTIAL:
        exponentialRenderer();
        break;
    case LINEAR:
        linearRenderer();
        break;
    case LINEARAPPROX:
        linearApproxRenderer();
        break;
    case DUMMY:
        dummyRenderer();
        break;
    default:
        throw std::runtime_error("Unimplemented render algorithm");
        break;
    }

    flush_encoder();
}
