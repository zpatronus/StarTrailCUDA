#pragma once
#include "../frame_reader/frame_reader.h"
#include <string>

enum RenderAlgo { AVGRAGE, MAX, EXPONENTIAL, DUMMY };

class VideoRenderer {
  private:
    const RenderAlgo algo;
    std::unique_ptr<FrameReader> frame_reader;
    std::string output_path;
    int fps;

    cv::VideoWriter writer;

    void averageRenderer();
    void maxRenderer();
    void exponentialRenderer();
    void dummyRenderer();

  public:
    explicit VideoRenderer(std::unique_ptr<FrameReader> reader, std::string output_path, int fps,
                           RenderAlgo algorithm)
        : frame_reader(std::move(reader)), output_path(output_path), fps(fps), algo(algorithm){

                                                                               };
    ~VideoRenderer(){};
    void render();
};