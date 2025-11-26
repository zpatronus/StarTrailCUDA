#pragma once
#include "../frame_reader/frame_reader.h"
#include <string>

enum RenderAlgo { AVGRAGE, MAX, EXPONENTIAL, LINEAR, LINEARAPPROX, DUMMY };

class VideoRenderer {
  private:
    const RenderAlgo algo;
    std::unique_ptr<FrameReader> frame_reader;
    std::string output_path;
    int fps;
    int window_size;

    cv::VideoWriter writer;

    void averageRenderer();
    void maxRenderer();
    void exponentialRenderer();
    void linearRenderer();
    void linearApproxRenderer();
    void dummyRenderer();

  public:
    explicit VideoRenderer(std::unique_ptr<FrameReader> reader, std::string output_path, int fps,
                           RenderAlgo algorithm, int window_size = 0)
        : frame_reader(std::move(reader)), output_path(output_path), fps(fps), algo(algorithm),
          window_size(window_size){

          };
    ~VideoRenderer(){};
    void render();
};