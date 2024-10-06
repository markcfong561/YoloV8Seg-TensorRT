#include <opencv2/core.hpp>

#include "NvInfer.h"

// To make ILogger happy
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity,
             const nvinfer1::AsciiChar* msg) noexcept {
        if (severity == nvinfer1::ILogger::Severity::kERROR)
            printf("%s\n", msg);
    }
};

class Detection {
   public:
    Detection(int classId, float conf,
              cv::Rect boundingBox, cv::Mat mask);

    int classId();

    float confidence();

    cv::Rect bbox();

    cv::Mat mask();

   private:
    int classId_;
    float conf_;
    cv::Rect boundingBox_;
    cv::Mat mask_;
};

class YoloV8Detector {
   public:
    YoloV8Detector(std::string filepath, float conf_threshold,
                   float iou_threshold);

    ~YoloV8Detector();

    std::vector<Detection> runDetection(cv::Mat &image);

   private:
    float *input, *output0, *output1, *output0Copy, *output1Copy, *maskWeights, *mask;
    int numClasses_;
    float confThreshold_, iouThreshold_;

    nvinfer1::IRuntime *runtime;
    nvinfer1::IExecutionContext *context;
    cudaStream_t contextStream;

    void iou(std::vector<std::vector<Detection>> &detections, cv::Rect newBbox, int classId, float conf, int iVal);

    // Must be gpu pointers
    cv::Mat calculateMask(float *weights);
};