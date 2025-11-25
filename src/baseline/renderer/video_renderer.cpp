#include "video_renderer.h"
#include "../frame_reader/frame_reader.h"

void VideoRenderer::averageRenderer() {
    cv::Mat acc;
    std::queue<cv::Mat> render_window;
    const unsigned int RENDER_WINDOW_SIZE = 12;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            render_window.push(frame.value());
            if (acc.empty()) {
                frame.value().convertTo(acc, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                acc += frame_f;
                if (render_window.size() > RENDER_WINDOW_SIZE) {
                    auto frame_rm = render_window.front();
                    render_window.pop();
                    cv::Mat frame_rm_f;
                    frame_rm.convertTo(frame_rm_f, CV_32FC3);
                    acc -= frame_rm_f;
                }
            }
            cv::Mat acc_write = acc / render_window.size();
            acc_write.convertTo(acc_write, CV_8UC3);
            writer.write(acc_write);
        } else {
            break;
        }
    }
}

void VideoRenderer::maxRenderer() {
    cv::Mat maxFrame;
    const double DECAY_FACTOR = 0.95;
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            if (maxFrame.empty()) {
                frame.value().convertTo(maxFrame, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                maxFrame = cv::max(maxFrame * DECAY_FACTOR, frame_f);
            }
            cv::Mat output_frame;
            maxFrame.convertTo(output_frame, CV_8UC3);
            writer.write(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::exponentialRenderer() {
    cv::Mat accFrame;
    const double EXP_FACTOR = 0.05;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            if (accFrame.empty()) {
                frame.value().convertTo(accFrame, CV_32FC3);
            } else {
                cv::Mat frame_f;
                frame.value().convertTo(frame_f, CV_32FC3);
                accFrame = (1 - EXP_FACTOR) * accFrame + EXP_FACTOR * frame_f;
            }
            cv::Mat output_frame;
            accFrame.convertTo(output_frame, CV_8UC3);
            writer.write(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::dummyRenderer() {
    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            writer.write(frame.value());
        } else {
            break;
        }
    }
}

void VideoRenderer::render() {
    writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), static_cast<double>(fps),
                frame_reader->getFrameSize());
    switch (algo) {
    case AVGRAGE:
        averageRenderer();
        break;
    case MAX:
        maxRenderer();
        break;
    case EXPONENTIAL:
        exponentialRenderer();
        break;
    case DUMMY:
        dummyRenderer();
        break;
    default:
        throw std::runtime_error("Unimplemented render algorithm");
        break;
    }
}
