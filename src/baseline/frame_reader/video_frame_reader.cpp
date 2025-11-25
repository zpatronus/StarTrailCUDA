#include "video_frame_reader.h"
#include "../utils/utils.h"
#include <stdexcept>

void VideoFrameReader::print_video_metadata() {
    if (cap.isOpened()) {
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        double frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
        double durationSec = (fps > 0 && frameCount > 0) ? (frameCount / fps) : -1.0;

        std::cout << "  Resolution : " << width << " x " << height << "\n"
                  << "  FPS        : " << fps << "\n"
                  << "  FrameCount : " << frameCount << "\n"
                  << "  Duration   : " << durationSec << " seconds\n";
    }
}

VideoFrameReader::VideoFrameReader(const std::string& videoPath, const unsigned int frameStep)
    : currentFrameIndex(0), frameStep(frameStep) {
    cap.open(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + videoPath);
    }
    print_video_metadata();
    totalFrames = static_cast<int>(ceil(ceil(cap.get(cv::CAP_PROP_FRAME_COUNT)) / frameStep));
}

cv::Size VideoFrameReader::getFrameSize() {
    if (!cap.isOpened()) {
        throw std::runtime_error("Video unopened");
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return cv::Size(width, height);
}

std::optional<cv::Mat> VideoFrameReader::nextFrame() {
    cv::Mat frame;
    while (currentReadIndex < totalFrames) {
        if (!cap.read(frame)) {
            return std::nullopt;
        }
        currentFrameIndex++;
        if ((currentFrameIndex - 1) % frameStep == 0) {
            currentReadIndex++;
            print_progress(currentReadIndex, totalFrames);
            return frame;
        }
    }
    return std::nullopt;
}

bool VideoFrameReader::hasNextFrame() { return currentReadIndex < totalFrames; }

void VideoFrameReader::reset() {
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    currentFrameIndex = 0;
    currentReadIndex = 0;
}

int VideoFrameReader::getTotalFrames() const { return totalFrames; }

int VideoFrameReader::getCurrentFrameIndex() const { return currentReadIndex; }