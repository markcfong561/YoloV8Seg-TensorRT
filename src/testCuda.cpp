#include <NvInfer.h>
#include <cuda_runtime.h>

#include "yolov8seg.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
    auto detector = YoloV8Detector("/home/markc/Downloads/yolov8n-seg.trt", 0.2, 0.8);

    cv::Mat image = cv::imread("/home/markc/Downloads/dog.png", cv::IMREAD_UNCHANGED);
    cv::Mat convertedImage, floatImage;
    cv::cvtColor(image, convertedImage, cv::COLOR_BGR2RGB);

    // cv::imshow("window", convertedImage);
    // cv::waitKey(0);

    // image = image.t();

    convertedImage.convertTo(floatImage, CV_32FC3);
    floatImage /= 255.0;

    detector.runDetection(floatImage);

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++)
    {
    auto detections = detector.runDetection(floatImage);

    for (auto detection : detections)
    {
        cv::rectangle(image, detection.bbox(), cv::Scalar(255, 0, 0));
        // printf("Shape: %d %d\n", detection.mask().cols, detection.mask().rows);
        // cv::Mat resizedMask;
        // cv::resize(detection.mask(), resizedMask, cv::Size(640, 640));
        // cv::imshow("window", resizedMask);
        // cv::waitKey(0);
    }
    // printf("Size: %d\n", detections.size());
    // cv::imshow("window", image);
    // cv::waitKey(0);

    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    printf("Took %fms per inference\n", (float)duration.count() / 1e6 / 100);
}