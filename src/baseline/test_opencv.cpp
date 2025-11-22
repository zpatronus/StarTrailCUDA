#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(img, "Hello OpenCV on WSL", cv::Point(30, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    cv::imshow("test", img);
    cv::waitKey(0);

    cv::imwrite("test_output.png", img);
    std::cout << "Saved test_output.png" << std::endl;
    return 0;
}
