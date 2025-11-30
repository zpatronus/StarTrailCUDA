#include "video_renderer.h"
#include "../frame_reader/frame_reader.h"
#include <deque>
#include <iostream>
#include <queue>

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void max_kernel(uint8_t* output, const uint8_t* input, uint8_t* max_frame, float decay_factor, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			float decayed = max_frame[idx + c] * decay_factor;
			float current = input[idx + c];
			max_frame[idx + c] = fmaxf(decayed, current);
			output[idx + c] = max_frame[idx + c];
		}
	}
}

__global__ void average_kernel(uint8_t* output, const uint8_t** window, int window_count, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			float sum = 0.0f;
			for (int i = 0; i < window_count; i++) {
				sum += window[i][idx + c];
			}
			output[idx + c] = (uint8_t)(sum / window_count);
		}
	}
}

__global__ void exponential_kernel(uint8_t* output, const uint8_t* input, uint8_t* acc_frame, float exp_factor, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			float acc_val = acc_frame[idx + c];
			float input_val = input[idx + c];
			acc_frame[idx + c] = (1.0f - exp_factor) * acc_val + exp_factor * input_val;
			output[idx + c] = acc_frame[idx + c];
		}
	}
}

__global__ void linear_kernel(uint8_t* output, const uint8_t** window, const float* weights, int window_count, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			float max_val = 0.0f;
			for (int i = 0; i < window_count; i++) {
				float weighted_val = fmaxf(window[i][idx + c] * weights[i], max_val);
				max_val = weighted_val;
			}
			output[idx + c] = (uint8_t)fminf(255.0f, max_val);
		}
	}
}

__global__ void linear_approx_kernel(uint8_t* output, const uint8_t* input, uint8_t* y_frame, float factor, float L, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			float input_val = input[idx + c] / 255.0f;
			float y_val = y_frame[idx + c] / 255.0f;

			float temp = (L + 1) - ((1 + L) - y_val) * factor;
			float new_y = fmaxf(temp, 0.0f);
			new_y = fmaxf(new_y, input_val);

			y_frame[idx + c] = (uint8_t)(new_y * 255.0f);
			output[idx + c] = y_frame[idx + c];
		}
	}
}

__global__ void copy_kernel(uint8_t* output, const uint8_t* input, int width, int height, int pitch) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		int idx = y * pitch + x * 3;
		for (int c = 0; c < 3; c++) {
			output[idx + c] = input[idx + c];
		}
	}
}

VideoRenderer::VideoRenderer(std::unique_ptr<FrameReader> reader, std::string output_path, int fps,
                             RenderAlgo algorithm, int window_size)
    : frame_reader(std::move(reader)), output_path(output_path), fps(fps), algo(algorithm),
      window_size(window_size), fmt_ctx_(nullptr), codec_ctx_(nullptr), stream_(nullptr),
      sws_ctx_(nullptr), pts_(0) {
    init_encoder();
}

VideoRenderer::~VideoRenderer() { cleanup_encoder(); }

void VideoRenderer::init_encoder() {
    FrameSize frame_size = frame_reader->getFrameSize();
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

    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        throw std::runtime_error("Could not create SwsContext for encoding");
    }
}

void VideoRenderer::encode_frame(const FrameData& frame) {
    AVFrame* av_frame = av_frame_alloc();
    av_frame->format = codec_ctx_->pix_fmt;
    av_frame->width = codec_ctx_->width;
    av_frame->height = codec_ctx_->height;

    if (av_frame_get_buffer(av_frame, 0) < 0) {
        av_frame_free(&av_frame);
        throw std::runtime_error("Could not allocate frame buffer");
    }

    const uint8_t* src[1] = {frame.getRawData()};
    int src_linesize[1] = {static_cast<int>(frame.width * 3)};
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
    const unsigned int RENDER_WINDOW_SIZE = window_size;
    FrameSize frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    int pitch = width * channels;

    uint8_t **d_window = nullptr;
    cudaMalloc(&d_window, RENDER_WINDOW_SIZE * sizeof(uint8_t*));
    uint8_t *d_output = nullptr;
    cudaMalloc(&d_output, width * height * channels);

    std::vector<uint8_t*> d_frames(RENDER_WINDOW_SIZE, nullptr);
    std::vector<FrameData> frame_window;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            frame_window.push_back(frame.value());
            if (frame_window.size() > RENDER_WINDOW_SIZE) {
                frame_window.erase(frame_window.begin());
            }

            int actual_window_size = frame_window.size();
            for (int i = 0; i < actual_window_size; ++i) {
                if (!d_frames[i]) cudaMalloc(&d_frames[i], width * height * channels);
                cudaMemcpy(d_frames[i], frame_window[i].getRawData(), width * height * channels, cudaMemcpyHostToDevice);
            }

            cudaMemcpy(d_window, d_frames.data(), actual_window_size * sizeof(uint8_t*), cudaMemcpyHostToDevice);

            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            average_kernel<<<grid, block>>>(d_output, (const uint8_t**)d_window, actual_window_size, width, height, pitch);
            cudaDeviceSynchronize();

            FrameData output_frame(width, height, channels);
            cudaMemcpy(output_frame.getRawData(), d_output, width * height * channels, cudaMemcpyDeviceToHost);
            encode_frame(output_frame);
        } else {
            break;
        }
    }

    for (auto ptr : d_frames) if (ptr) cudaFree(ptr);
    cudaFree(d_window);
    cudaFree(d_output);
}

void VideoRenderer::maxRenderer() {
    FrameSize frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    int pitch = width * channels;
    const float DECAY_FACTOR = 0.95f;

    uint8_t *d_max = nullptr, *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_output, width * height * channels);
    cudaMalloc(&d_input, width * height * channels);
    cudaMalloc(&d_max, width * height * channels);
    cudaMemset(d_max, 0, width * height * channels);

    bool first = true;
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            cudaMemcpy(d_input, frame.value().getRawData(), width * height * channels, cudaMemcpyHostToDevice);
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            max_kernel<<<grid, block>>>(d_output, d_input, d_max, DECAY_FACTOR, width, height, pitch);
            cudaDeviceSynchronize();

            FrameData output_frame(width, height, channels);
            cudaMemcpy(output_frame.getRawData(), d_output, width * height * channels, cudaMemcpyDeviceToHost);
            encode_frame(output_frame);
            cudaMemcpy(d_max, d_output, width * height * channels, cudaMemcpyDeviceToDevice);
            first = false;
        } else {
            break;
        }
    }
    cudaFree(d_max);
    cudaFree(d_input);
    cudaFree(d_output);
}

void VideoRenderer::exponentialRenderer() {
    FrameSize frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    int pitch = width * channels;
    const float EXP_FACTOR = 0.05f;

    uint8_t *d_acc = nullptr, *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_output, width * height * channels);
    cudaMalloc(&d_input, width * height * channels);
    cudaMalloc(&d_acc, width * height * channels);

    bool first = true;
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            cudaMemcpy(d_input, frame.value().getRawData(), width * height * channels, cudaMemcpyHostToDevice);

            if (first) {
                cudaMemcpy(d_acc, d_input, width * height * channels, cudaMemcpyDeviceToDevice);
                first = false;
            }

            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            exponential_kernel<<<grid, block>>>(d_output, d_input, d_acc, EXP_FACTOR, width, height, pitch);
            cudaDeviceSynchronize();

            FrameData output_frame(width, height, channels);
            cudaMemcpy(output_frame.getRawData(), d_output, width * height * channels, cudaMemcpyDeviceToHost);
            encode_frame(output_frame);
            cudaMemcpy(d_acc, d_output, width * height * channels, cudaMemcpyDeviceToDevice);
        } else {
            break;
        }
    }
    cudaFree(d_acc);
    cudaFree(d_input);
    cudaFree(d_output);
}

void VideoRenderer::linearRenderer() {
    FrameSize frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    int pitch = width * channels;
    const int RENDER_WINDOW_SIZE = window_size;

    uint8_t **d_window = nullptr;
    cudaMalloc(&d_window, RENDER_WINDOW_SIZE * sizeof(uint8_t*));
    uint8_t *d_output = nullptr;
    cudaMalloc(&d_output, width * height * channels);
    float *d_weights = nullptr;
    cudaMalloc(&d_weights, RENDER_WINDOW_SIZE * sizeof(float));

    std::vector<uint8_t*> d_frames(RENDER_WINDOW_SIZE, nullptr);
    std::vector<FrameData> frame_window;

    std::vector<float> h_weights(RENDER_WINDOW_SIZE);
    for (int i = 0; i < RENDER_WINDOW_SIZE; ++i) {
        h_weights[i] = static_cast<float>(RENDER_WINDOW_SIZE - i) / static_cast<float>(RENDER_WINDOW_SIZE);
    }
    cudaMemcpy(d_weights, h_weights.data(), RENDER_WINDOW_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            frame_window.push_back(frame.value());
            if (frame_window.size() > RENDER_WINDOW_SIZE) {
                frame_window.erase(frame_window.begin());
            }

            int actual_window_size = frame_window.size();
            for (int i = 0; i < actual_window_size; ++i) {
                if (!d_frames[i]) cudaMalloc(&d_frames[i], width * height * channels);
                cudaMemcpy(d_frames[i], frame_window[i].getRawData(), width * height * channels, cudaMemcpyHostToDevice);
            }

            cudaMemcpy(d_window, d_frames.data(), actual_window_size * sizeof(uint8_t*), cudaMemcpyHostToDevice);

            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            linear_kernel<<<grid, block>>>(d_output, (const uint8_t**)d_window, d_weights, actual_window_size, width, height, pitch);
            cudaDeviceSynchronize();

            FrameData output_frame(width, height, channels);
            cudaMemcpy(output_frame.getRawData(), d_output, width * height * channels, cudaMemcpyDeviceToHost);
            encode_frame(output_frame);
        } else {
            break;
        }
    }

    for (auto ptr : d_frames) if (ptr) cudaFree(ptr);
    cudaFree(d_window);
    cudaFree(d_output);
    cudaFree(d_weights);
}

void VideoRenderer::linearApproxRenderer() {
    FrameSize frame_size = frame_reader->getFrameSize();
    int width = frame_size.width;
    int height = frame_size.height;
    int channels = 3;
    int pitch = width * channels;
    const int N = window_size;
    const float L = 10.0f;
    const float factor = expf(1.0f / (L * N));

    uint8_t *d_y_frame = nullptr, *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_output, width * height * channels);
    cudaMalloc(&d_input, width * height * channels);
    cudaMalloc(&d_y_frame, width * height * channels);
    cudaMemset(d_y_frame, 0, width * height * channels);

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            cudaMemcpy(d_input, frame.value().getRawData(), width * height * channels, cudaMemcpyHostToDevice);

            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            linear_approx_kernel<<<grid, block>>>(d_output, d_input, d_y_frame, factor, L, width, height, pitch);
            cudaDeviceSynchronize();

            FrameData output_frame(width, height, channels);
            cudaMemcpy(output_frame.getRawData(), d_output, width * height * channels, cudaMemcpyDeviceToHost);
            encode_frame(output_frame);
        } else {
            break;
        }
    }
    cudaFree(d_y_frame);
    cudaFree(d_input);
    cudaFree(d_output);
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
