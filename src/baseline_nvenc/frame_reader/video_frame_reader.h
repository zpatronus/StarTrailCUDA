#pragma once
#include "frame_reader.h"
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class VideoFrameReader : public FrameReader {
  private:
    AVFormatContext* fmt_ctx_;
    AVCodecContext* codec_ctx_;
    int video_stream_idx_;
    SwsContext* sws_ctx_;
    int width_;
    int height_;
    int current_frame_index_;
    int current_read_index_;
    int total_frames_;
    const unsigned int frame_step_;

    void print_video_metadata();

  public:
    explicit VideoFrameReader(const std::string& videoPath, const unsigned int frameStep);
    ~VideoFrameReader() override;

    cv::Size getFrameSize() override;

    std::optional<cv::Mat> nextFrame() override;
    bool hasNextFrame() override;
    void reset() override;
    int getTotalFrames() const override;
    int getCurrentFrameIndex() const override;
};
