#pragma once
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using std::string;

void print_video_metadata(cv::VideoCapture& cap);

void print_progress(int current, int total);

int sample_video(const string& video_path, unsigned int frame_step,
                 std::function<void(const cv::Mat& frame, int index, double timestamp)> on_frame);

int sample_video_to_files(const std::string& video_path, unsigned int frame_step,
                          const std::string& out_dir);

int sample_video_to_mem(const std::string& video_path, unsigned int frame_step,
                        std::vector<cv::Mat>& out_frames);