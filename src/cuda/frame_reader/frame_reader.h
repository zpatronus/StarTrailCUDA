#pragma once
#include <memory>
#include <optional>
#include <vector>

struct FrameSize {
    int width;
    int height;
    int channels = 3;

    FrameSize() : width(0), height(0) {}
    FrameSize(int w, int h) : width(w), height(h) {}
};

struct FrameData {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;

    FrameData() : width(0), height(0), channels(3) {}
    FrameData(int w, int h, int c = 3) : width(w), height(h), channels(c), data(w * h * c) {}

    uint8_t* getRawData() { return data.data(); }
    const uint8_t* getRawData() const { return data.data(); }
    size_t getSizeInBytes() const { return data.size(); }
};

class FrameReader {
  private:
    /* data */
  public:
    FrameReader(){};
    virtual ~FrameReader() = default;

    virtual FrameSize getFrameSize() = 0;

    virtual std::optional<FrameData> nextFrame() = 0;

    virtual bool hasNextFrame() = 0;

    virtual void reset() {}

    virtual int getTotalFrames() const { return -1; }

    virtual int getCurrentFrameIndex() const { return -1; }
};
