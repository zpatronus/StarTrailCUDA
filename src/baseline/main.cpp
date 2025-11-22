#include <getopt.h>
#include <iostream>
#include <string>

#include "video_sampler.h"

using std::string;

int main(int argc, char* argv[]) {
    std::cout << "Running Star Trail CUDA baseline" << std::endl;
    string video_path = "";
    string out_dir = "";
    unsigned int frame_step = 0;

    static struct option long_options[] = {{"video_path", required_argument, 0, 'v'},
                                           {"out_dir", required_argument, 0, 'o'},
                                           {"frame_step", required_argument, 0, 'f'},
                                           {0, 0, 0, 0}}; // End of options array

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "v:o:f:", long_options, &option_index)) != -1) {
        switch (c) {
        case 'v':
            video_path = optarg;
            break;
        case 'o':
            out_dir = optarg;
            break;
        case 'f':
            try {
                frame_step = std::stoi(optarg);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument for frame_step: " << optarg << std::endl;
                exit(1);
            } catch (const std::out_of_range& e) {
                std::cerr << "frame_step out of range" << std::endl;
                exit(1);
            }
            if (frame_step == 0) {
                std::cerr << "frame_step must be a positive integer" << std::endl;
                exit(1);
            }
            break;
        default:
            break;
        }
    }

    if (video_path == "" || out_dir == "" || frame_step == 0) {
        std::cerr << "Missing arguments:";
        if (video_path == "") {
            std::cerr << " [video_path]";
        }
        if (out_dir == "") {
            std::cerr << " [out_dir]";
        }
        if (frame_step == 0) {
            std::cerr << " [frame_step]";
        }
        std::cerr << std::endl;
        exit(1);
    }

    std::cout << "Current arguments:" << std::endl;
    std::cout << "[video_path]: " << video_path << std::endl;
    std::cout << "[out_dir]: " << out_dir << std::endl;
    std::cout << "[frame_step]: " << frame_step << std::endl;

    int sampled = sample_video_to_files(video_path, frame_step, out_dir);
    return sampled;
}