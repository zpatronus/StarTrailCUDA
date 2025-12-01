#pragma once
#include "../frame_reader/frame_reader.h"
#include <chrono>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <stdint.h>

enum RenderAlgo { AVGRAGE, MAX, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY };

class VideoRenderer {
  private:
    const RenderAlgo algo;
    std::unique_ptr<FrameReader> frame_reader;
    std::string output_path;
    int fps;
    int window_size;

    AVFormatContext* fmt_ctx_;
    AVCodecContext* codec_ctx_;
    AVStream* stream_;
    SwsContext* sws_ctx_;
    int64_t pts_;

    double decode_time_ms = 0.0;
    double render_time_ms = 0.0;
    double encode_time_ms = 0.0;

    void init_encoder();
    void encode_frame(const FrameData& frame);
    void flush_encoder();
    void cleanup_encoder();

    void averageRenderer();
    void maxRenderer();
    void exponentialRenderer();
    void linearRenderer();
    void linearApproxRenderer();
    void dummyRenderer();

  public:
    explicit VideoRenderer(std::unique_ptr<FrameReader> reader, std::string output_path, int fps,
                           RenderAlgo algorithm, int window_size = 0);
    ~VideoRenderer();
    void render();
    void printTimingStats();
};
