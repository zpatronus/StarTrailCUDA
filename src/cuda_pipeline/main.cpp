#include "decoder/video_decoder.h"
#include "encoder/video_encoder.h"
#include "renderer/video_renderer.h"
#include "utils/cuda_utils.h"
#include "utils/frame_queue.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "CUDA-accelerated star trail video renderer\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input VIDEO_PATH      Input video file path (required)\n";
    std::cout << "  -o, --output VIDEO_PATH     Output video file path (required)\n";
    std::cout << "  -f, --fps FPS               Output video fps (default: input video fps)\n";
    std::cout << "  -s, --step STEP             Input video sampling step (default: 1)\n";
    std::cout << "  -a, --algorithm ALGORITHM   Render algorithm: MAX, AVERAGE, EXPONENTIAL, "
                 "LINEAR, LINEARAPPROX, DUMMY (required)\n";
    std::cout << "  -w, --window-size SIZE      Window size for AVERAGE, LINEAR, LINEARAPPROX "
                 "(default: fps)\n";
    std::cout << "  -h, --help                  Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -i input.mp4 -o output.mp4 -a MAX\n";
    std::cout << "  " << program_name << " -i input.mp4 -o output.mp4 -a LINEARAPPROX -w 60\n";
    std::cout << "  " << program_name
              << " -i input.mp4 -o output.mp4 -a AVERAGE -w 12 -f 30 -s 2\n";
}

RenderAlgorithm parse_algorithm(const std::string& algo_str) {
    if (algo_str == "MAX")
        return RenderAlgorithm::MAX;
    if (algo_str == "AVERAGE")
        return RenderAlgorithm::AVERAGE;
    if (algo_str == "EXPONENTIAL")
        return RenderAlgorithm::EXPONENTIAL;
    if (algo_str == "LINEAR")
        return RenderAlgorithm::LINEAR;
    if (algo_str == "LINEARAPPROX")
        return RenderAlgorithm::LINEARAPPROX;
    if (algo_str == "DUMMY")
        return RenderAlgorithm::DUMMY;
    throw std::invalid_argument(
        "Invalid algorithm: " + algo_str +
        ". Must be one of: MAX, AVERAGE, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY");
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    std::string output_path = "";
    int fps = -1;
    int step = 1;
    int window_size = 0;
    std::string algorithm_str = "";

    static struct option long_options[] = {{"input", required_argument, 0, 'i'},
                                           {"output", required_argument, 0, 'o'},
                                           {"fps", required_argument, 0, 'f'},
                                           {"step", required_argument, 0, 's'},
                                           {"algorithm", required_argument, 0, 'a'},
                                           {"window-size", required_argument, 0, 'w'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "i:o:f:s:a:w:h", long_options, &option_index)) != -1) {
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
        case 'w':
            try {
                window_size = std::stoi(optarg);
                if (window_size <= 0) {
                    std::cerr << "Error: window-size must be a positive integer, got: " << optarg
                              << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid window-size value: " << optarg << std::endl;
                return 1;
            }
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

    if (input_path.empty() || output_path.empty() || algorithm_str.empty()) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        std::cerr << "Required: --input, --output, --algorithm" << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        return 1;
    }

    RenderAlgorithm algorithm;
    try {
        algorithm = parse_algorithm(algorithm_str);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cuda_device_info();

    auto decode_queue = std::make_shared<FrameQueue>(10);
    auto render_queue = std::make_shared<FrameQueue>(10);

    std::cout << "Initializing decoder...\n";
    auto decoder = std::make_unique<VideoDecoder>(input_path, step, decode_queue);
    decoder->start();

    int width = decoder->get_width();
    int height = decoder->get_height();
    int video_fps = decoder->get_fps();

    if (fps == -1) {
        fps = video_fps;
    }

    if (window_size == 0) {
        window_size = fps;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Input: " << input_path << "\n";
    std::cout << "  Output: " << output_path << "\n";
    std::cout << "  FPS: " << fps << "\n";
    std::cout << "  Step: " << step << "\n";
    std::cout << "  Algorithm: " << algorithm_str << "\n";
    std::cout << "  Window Size: " << window_size << "\n\n";

    std::cout << "Starting renderer...\n";
    auto renderer = std::make_unique<VideoRenderer>(width, height, window_size, algorithm,
                                                    decode_queue, render_queue);

    std::cout << "Starting encoder...\n";
    auto encoder = std::make_unique<VideoEncoder>(output_path, width, height, fps, render_queue);
    encoder->start();

    cudaDeviceSynchronize();

    auto start_time = std::chrono::high_resolution_clock::now();

    renderer->start();

    decoder->wait();
    renderer->wait();
    encoder->wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nProcessing completed in " << duration.count() << " ms ("
              << duration.count() / 1000.0 << " seconds)\n";

    decoder->print_stats();
    renderer->print_stats();
    encoder->print_stats();

    return 0;
}
