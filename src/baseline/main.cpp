#include "frame_reader/frame_reader.h"
#include "frame_reader/video_frame_reader.h"
#include "renderer/video_renderer.h"
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input VIDEO_PATH      Input video file path (required)\n";
    std::cout << "  -o, --output VIDEO_PATH     Output video file path (required)\n";
    std::cout << "  -f, --fps FPS               Output video fps (required, must be > 0)\n";
    std::cout
        << "  -s, --step STEP             Input video sampling step (required, must be > 0)\n";
    std::cout << "  -a, --algorithm ALGORITHM   Render algorithm: MAX, AVERAGE, EXPONENTIAL, DUMMY "
                 "(required)\n";
    std::cout << "  -h, --help                  Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " -i input.mp4 -o output.mp4 -f 30 -s 1 -a MAX\n";
}

RenderAlgo parse_algorithm(const std::string& algo_str) {
    if (algo_str == "MAX")
        return MAX;
    if (algo_str == "AVERAGE")
        return AVGRAGE;
    if (algo_str == "EXPONENTIAL")
        return EXPONENTIAL;
    if (algo_str == "DUMMY")
        return DUMMY;
    throw std::invalid_argument("Invalid algorithm: " + algo_str +
                                ". Must be one of: MAX, AVERAGE, EXPONENTIAL, DUMMY");
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    std::string output_path = "";
    int fps = 0;
    int step = 0;
    std::string algorithm_str = "";

    static struct option long_options[] = {{"input", required_argument, 0, 'i'},
                                           {"output", required_argument, 0, 'o'},
                                           {"fps", required_argument, 0, 'f'},
                                           {"step", required_argument, 0, 's'},
                                           {"algorithm", required_argument, 0, 'a'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "i:o:f:s:a:h", long_options, &option_index)) != -1) {
        switch (c) {
        case 'i':
            input_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'f':
            try {
                fps = std::stoi(optarg);
                if (fps <= 0) {
                    std::cerr << "Error: fps must be a positive integer, got: " << optarg
                              << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid fps value: " << optarg << std::endl;
                return 1;
            }
            break;
        case 's':
            try {
                step = std::stoi(optarg);
                if (step <= 0) {
                    std::cerr << "Error: step must be a positive integer, got: " << optarg
                              << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid step value: " << optarg << std::endl;
                return 1;
            }
            break;
        case 'a':
            algorithm_str = optarg;
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        case '?':
            std::cerr << "Use -h or --help for usage information." << std::endl;
            return 1;
        default:
            break;
        }
    }

    // Check required arguments
    if (input_path.empty() || output_path.empty() || fps == 0 || step == 0 ||
        algorithm_str.empty()) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        std::cerr << "Required: --input, --output, --fps, --step, --algorithm" << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        return 1;
    }

    // Parse and validate algorithm
    RenderAlgo algorithm;
    try {
        algorithm = parse_algorithm(algorithm_str);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Input: " << input_path << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Step: " << step << std::endl;
    std::cout << "  Algorithm: " << algorithm_str << std::endl;
    std::cout << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::unique_ptr<FrameReader> frame_reader =
        std::make_unique<VideoFrameReader>(VideoFrameReader(input_path, step));
    VideoRenderer renderer(std::move(frame_reader), output_path, fps, algorithm);
    renderer.render();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nProcessing completed in " << duration.count() << " ms ("
              << duration.count() / 1000.0 << " seconds)" << std::endl;

    return 0;
}