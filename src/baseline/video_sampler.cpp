#include "video_sampler.h"

using std::string;

void print_video_metadata(cv::VideoCapture& cap) {
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

void print_progress(int current, int total) {
    const int width = 50;
    double ratio = total ? (double)current / total : 0.0;
    int filled = (int)(ratio * width);

    // Print bar, then current/total, then percentage
    std::cout << "\r[" << std::string(filled, '=') << std::string(width - filled, ' ') << "] "
              << current << "/" << total << " " << std::setw(3) << (int)(ratio * 100) << "%";
    std::cout.flush();

    if (current >= total)
        std::cout << std::endl;
}

int sample_video(const string& video_path, unsigned int frame_step,
                 std::function<void(const cv::Mat& frame, int index)> on_frame) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video " + video_path);
    }
    std::cout << "Handling video " << video_path << std::endl;
    print_video_metadata(cap);
    // Read the video by frames
    cv::Mat frame;
    unsigned int write_counter = 0;
    unsigned int n = 0;
    // For progress bar
    int saveFrameCount =
        static_cast<int>(ceil(ceil(cap.get(cv::CAP_PROP_FRAME_COUNT)) / frame_step));

    while (cap.read(frame)) {
        n %= frame_step;
        if (n == 0) {
            on_frame(frame, write_counter);
            write_counter++;
            print_progress(write_counter, saveFrameCount);
        }
        n++;
    }
    return 0;
}

int sample_video_to_files(const std::string& video_path, unsigned int frame_step,
                          const std::string& out_dir) {
    std::filesystem::path out_dir_pathchecker = out_dir;
    if (std::filesystem::exists(out_dir_pathchecker)) {
        if (std::filesystem::is_regular_file(out_dir_pathchecker)) {
            throw std::filesystem::filesystem_error(
                "Output path exists and is not a directory", out_dir,
                std::make_error_code(std::errc::not_a_directory));
        }
    } else {
        std::filesystem::create_directories(out_dir);
    }
    // forward to the generic sampler; capture out_dir by value for the lambda
    int written = sample_video(video_path, frame_step, [out_dir](const cv::Mat& f, int idx) {
        std::string path = out_dir;
        if (!path.empty() && path.back() != '/' && path.back() != '\\')
            path.push_back('/');
        path += "frame_" + std::to_string(idx) + ".jpeg";
        cv::imwrite(path, f);
    });
    return written;
}

int sample_video_to_mem(const std::string& video_path, unsigned int frame_step,
                        std::vector<cv::Mat>& out_frames) {
    int written = sample_video(video_path, frame_step, [&out_frames](const cv::Mat& f, int idx) {
        out_frames.push_back(f.clone());
    });
    return written;
}