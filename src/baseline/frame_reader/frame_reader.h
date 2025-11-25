#pragma once
#include <opencv2/opencv.hpp>
#include <optional>

class FrameReader {
  private:
    /* data */
  public:
    FrameReader(){};
    virtual ~FrameReader() = default;

    virtual cv::Size getFrameSize() = 0;

    virtual std::optional<cv::Mat> nextFrame() = 0;

    virtual bool hasNextFrame() = 0;

    virtual void reset() {}

    virtual int getTotalFrames() const { return -1; }

    virtual int getCurrentFrameIndex() const { return -1; }
};
