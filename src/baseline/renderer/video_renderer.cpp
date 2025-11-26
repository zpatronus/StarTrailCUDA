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

void VideoRenderer::linearRenderer() {
    std::deque<cv::Mat> render_window;
    const int RENDER_WINDOW_SIZE = 16;

    while (1) {
        auto frame = frame_reader->nextFrame();
        if (frame.has_value()) {
            const cv::Mat& frame8u = frame.value(); // CV_8UC3
            render_window.push_front(frame8u);
            if (static_cast<int>(render_window.size()) > RENDER_WINDOW_SIZE) {
                render_window.pop_back();
            }

            cv::Mat accFrame(frame8u.rows, frame8u.cols, CV_32FC3, cv::Scalar(0, 0, 0));

            int windowSize = static_cast<int>(render_window.size());

            for (int i = 0; i < windowSize; ++i) {
                double weight = double(RENDER_WINDOW_SIZE - i) / double(RENDER_WINDOW_SIZE);

                cv::Mat curr32f;
                render_window[i].convertTo(curr32f, CV_32FC3, weight / 255.0);

                cv::max(accFrame, curr32f, accFrame);
            }

            cv::Mat output_frame;
            accFrame.convertTo(output_frame, CV_8UC3, 255.0); // [0,1] -> [0,255]
            writer.write(output_frame);
        } else {
            break;
        }
    }
}

void VideoRenderer::linearApproxRenderer() {
    using std::max;
    std::deque<cv::Mat> render_window;
    const int RENDER_WINDOW_SIZE = 16;
    const double L = 10.0;

    std::vector<float> weights(RENDER_WINDOW_SIZE);
    double y = 1.0; // Value at step = 0
    weights[0] = static_cast<float>(y);

    const double c = std::exp(1.0 / (L * RENDER_WINDOW_SIZE));

    for (int step = 1; step < RENDER_WINDOW_SIZE; ++step) {
        // y_{k+1} = max((L+1) - (1 + y_k) * e^(1/(L*N)), 0)
        y = max(0.0, (L + 1.0) - (1.0 + L - y) * c);
        weights[step] = static_cast<float>(y);
    }

    while (true) {
        auto frameOpt = frame_reader->nextFrame();
        if (frameOpt.has_value()) {
            const cv::Mat& frame8u = frameOpt.value();

            render_window.push_front(frame8u);
            if (static_cast<int>(render_window.size()) > RENDER_WINDOW_SIZE) {
                render_window.pop_back();
            }

            cv::Mat accFrame(frame8u.rows, frame8u.cols, CV_32FC3, cv::Scalar(0, 0, 0));

            const int windowSize = static_cast<int>(render_window.size());

            for (int i = 0; i < windowSize; ++i) {
                // i = 0 is the latest frame
                double w = weights[i];

                cv::Mat curr32f;
                render_window[i].convertTo(curr32f, CV_32FC3, (w / 255.0));

                // A(x,y) = max(A(x,y), w_i * F_i(x,y))
                cv::max(accFrame, curr32f, accFrame);
            }

            cv::Mat output_frame;
            accFrame.convertTo(output_frame, CV_8UC3, 255.0);
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
    case LINEAR:
        linearRenderer();
        break;
    case LINEARAPPROX:
        linearApproxRenderer();
        break;
    case DUMMY:
        dummyRenderer();
        break;
    default:
        throw std::runtime_error("Unimplemented render algorithm");
        break;
    }
}
