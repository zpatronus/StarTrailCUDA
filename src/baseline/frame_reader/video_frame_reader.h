#pragma once
#include "frame_reader.h"
#include <string>

class VideoFrameReader : public FrameReader {
  private:
    cv::VideoCapture cap;
    int currentFrameIndex = 0;
    int currentReadIndex = 0;
    int totalFrames = 0;
    const unsigned int frameStep;
    void print_video_metadata();

  public:
    explicit VideoFrameReader(const std::string& videoPath, const unsigned int frameStep);
    ~VideoFrameReader() override = default;

    cv::Size getFrameSize() override;

    std::optional<cv::Mat> nextFrame() override;
    bool hasNextFrame() override;
    void reset() override;
    int getTotalFrames() const override;
    int getCurrentFrameIndex() const override;
};