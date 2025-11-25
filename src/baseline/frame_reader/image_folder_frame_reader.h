#pragma once
#include "frame_reader.h"
#include <filesystem>
#include <string>
#include <vector>

class ImageFolderFrameReader : public FrameReader {
  private:
    std::vector<std::filesystem::path> imagePaths;
    size_t currentIndex;

    void loadImagePaths(const std::string& folderPath);

  public:
    explicit ImageFolderFrameReader(const std::string& folderPath);
    ~ImageFolderFrameReader() override = default;

    std::optional<cv::Mat> nextFrame() override;
    bool hasNextFrame() override;
    void reset() override;
    int getTotalFrames() const override;
    int getCurrentFrameIndex() const override;
};