#include "image_folder_frame_reader.h"
#include <algorithm>
#include <stdexcept>

ImageFolderFrameReader::ImageFolderFrameReader(const std::string& folderPath) : currentIndex(0) {
    loadImagePaths(folderPath);
    if (imagePaths.empty()) {
        throw std::runtime_error("No valid images found in folder: " + folderPath);
    }
}

void ImageFolderFrameReader::loadImagePaths(const std::string& folderPath) {
    if (!std::filesystem::exists(folderPath)) {
        throw std::runtime_error("Folder does not exist: " + folderPath);
    }

    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                imagePaths.push_back(entry.path());
            }
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
}

std::optional<cv::Mat> ImageFolderFrameReader::nextFrame() {
    if (currentIndex >= imagePaths.size()) {
        return std::nullopt;
    }

    cv::Mat frame = cv::imread(imagePaths[currentIndex].string());
    if (frame.empty()) {
        currentIndex++;
        return nextFrame();
    }

    currentIndex++;
    return frame;
}

bool ImageFolderFrameReader::hasNextFrame() { return currentIndex < imagePaths.size(); }

void ImageFolderFrameReader::reset() { currentIndex = 0; }

int ImageFolderFrameReader::getTotalFrames() const { return static_cast<int>(imagePaths.size()); }

int ImageFolderFrameReader::getCurrentFrameIndex() const { return static_cast<int>(currentIndex); }