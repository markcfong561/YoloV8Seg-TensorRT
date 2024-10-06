#include "yolov8seg.h"
// #include "postprocess.cu"

#include <fstream>
#include <cmath>

#include "NvInferVersion.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#define BLOCK_SIZE 32

__global__ void gpu_matrix_mult(float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

float sigmoid(float num)
{
    return (float)((1 / (1 + exp(-num))) > 0.5);
}

Detection::Detection(int classId, float conf,
                     cv::Rect boundingBox, cv::Mat mask)
    : classId_(classId), conf_(conf), boundingBox_(boundingBox), mask_(mask) {}

int Detection::classId() { return classId_; }

float Detection::confidence() { return conf_; }

cv::Rect Detection::bbox() { return boundingBox_; }

cv::Mat Detection::mask() { return mask_; }

// extern void processDets(float *output0, float *output1, float *detections, int *numDets, int numClasses, float confThresh, float iouThresh);

YoloV8Detector::YoloV8Detector(std::string filepath,
                               float conf_threshold,
                               float iou_threshold)
    : confThreshold_(conf_threshold), iouThreshold_(iou_threshold)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        std::string errorMsg = "File " + filepath + " does not exist";

        throw std::runtime_error(errorMsg);
    }

    // printf()

    Logger logger;

    std::ifstream engineFile(
        filepath.substr(0, filepath.find_last_of('.')) + ".trt", std::ios::binary | std::ios::ate);

    if (engineFile.good())
    {
        printf("TRT engine file detected\n");
        // engineFile = std::move(file);
    }

    else if (filepath.substr(filepath.size() - 5, filepath.size()) == ".onnx")
    {
        // Convert

        printf("Engine file not detected: creating now...\n");
        nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

        auto explicitBatch =
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        // auto network = builder->createNetworkV2(0);

        std::unique_ptr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(explicitBatch));

        if (!network)
        {
            throw std::runtime_error("Failed to build network");
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
        {
            throw std::runtime_error("Error, unable to read onnx file");
        }
        file.close();

        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger));
        if (!parser)
        {
            throw std::runtime_error("Failed to create parser");
        }

        auto parsed = parser->parseFromFile(filepath.c_str(), 0);
        if (!parsed)
        {
            throw std::runtime_error("Failed to parse onnx file");
        }

        // printf("Errors %d\n", parser->getNbErrors());

        std::unique_ptr<nvinfer1::IBuilderConfig> config(
            builder->createBuilderConfig());
        if (!config)
        {
            throw std::runtime_error("Failed to build config");
        }

        // printf("%d\n", network->getNbOutputs());

        // Has to be raw or else it won't compile
        nvinfer1::IOptimizationProfile *optProfile =
            builder->createOptimizationProfile();
        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));

        config->addOptimizationProfile(optProfile);

        // Do we want to add functionality to change the percision of the NN?
        // config.setFlag(nvinfer1::BuilderFlag::KFP16);

        // cudaStream_t profileStream;
        // cudaError_t ret = cudaStreamCreate(&profileStream);
        // if (ret != cudaSuccess) {
        //     throw std::runtime_error("Failed to create profile stream: " +
        //                              std::string(cudaGetErrorName(ret)) +
        //                              '\n' +
        //                              std::string(cudaGetErrorString(ret)));
        // }
        // config->setProfileStream(profileStream);
        std::unique_ptr<nvinfer1::IHostMemory> plan(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
        {
            throw std::runtime_error("Failed to build engine");
        }
        std::ofstream outfile(filepath.substr(0, filepath.size() - 5) + ".trt",
                              std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()),
                      plan->size());
        outfile.close();

        printf("Successfully created engine file\n");

        // cudaStreamDestroy(profileStream);

        engineFile =
            std::ifstream(filepath.substr(0, filepath.size() - 5) + ".trt",
                          std::ios::binary | std::ios::ate);
    }
    else
    {
        std::string errorMsg = "Invalid file type: Expects .onnx or .trt";

        throw std::runtime_error(errorMsg);
    }

    std::streamsize engineFileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineFileBuffer(engineFileSize);
    if (!engineFile.read(engineFileBuffer.data(), engineFileSize))
    {
        auto msg = "Error, unable to read engine file";
        throw std::runtime_error(msg);
    }
    // printf("Size: %d\n", engineFileBuffer.size());

    // printf("creating runtime\n");
    runtime = nvinfer1::createInferRuntime(logger);

    // printf("building runtime\n");
    nvinfer1::ICudaEngine *engine = (runtime->deserializeCudaEngine(
        engineFileBuffer.data(), engineFileBuffer.size()));
    // printf("built runtime\n");
    if (!engine)
    {
        throw std::runtime_error("Failed to create engine");
    }

    // printf("creating context\n");
    context = engine->createExecutionContext();
    // printf("created context\n");

    cudaError_t ret = cudaStreamCreate(&contextStream);

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to create context stream: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }

    numClasses_ = engine->getTensorShape("output0").d[1] - 4 - 32;
    cudaDeviceSynchronize();

    output0Copy = new float[8400 * (numClasses_ + 4 + 32)];
    output1Copy = new float[160 * 160 * 32];

    ret = cudaMallocManaged((void **)(&input), 640 * 640 * 3 * sizeof(float));

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to malloc input: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }
    ret = cudaMallocManaged((void **)(&output0), 8400 * (numClasses_ + 4 + 32) * sizeof(float));

    // printf("Num classes: %d\n", numClasses_);

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to malloc output0: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }
    ret = cudaMallocManaged((void **)(&output1), 160 * 160 * 32 * sizeof(float));

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to malloc output1: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }
    ret = cudaMallocManaged((void **)(&maskWeights), 32 * sizeof(float));

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to malloc maskWeights: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }
    ret = cudaMallocManaged((void **)(&mask), 160 * 160 * sizeof(float));

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to malloc mask: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }

    bool worked = context->setInputTensorAddress("images", input);
    if (!worked)
    {
        printf("Failed to set images\n");
    }
    worked = context->setTensorAddress("output0", output0);
    if (!worked)
    {
        printf("Failed to set output0\n");
    }
    worked = context->setTensorAddress("output1", output1);
    if (!worked)
    {
        printf("Failed to set output1\n");
    }
}

YoloV8Detector::~YoloV8Detector()
{
    // context->destroy();
    cudaStreamDestroy(contextStream);
    cudaFree(input);
    cudaFree(output0);
    cudaFree(output1);
    if (runtime != nullptr)
    {
        runtime = nullptr;
    }

    delete output0Copy;
    delete output1Copy;
}

std::vector<Detection> YoloV8Detector::runDetection(cv::Mat &image)
{
    using namespace std::chrono;
    cv::Mat blob = cv::dnn::blobFromImage(image);
    auto start = high_resolution_clock::now();
    cudaError_t ret = cudaMemcpy(input, blob.data, 640 * 640 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    auto firstMemcpy = high_resolution_clock::now();

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to memcpy to input: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }

    bool success = context->enqueueV3(contextStream);

    cudaStreamSynchronize(contextStream);
    auto enqueued = high_resolution_clock::now();

    if (!success)
    {
        throw std::runtime_error("Failed to run inference\n");
    }

    ret = cudaMemcpy(output0Copy, output0, 8400 * (numClasses_ + 4 + 32) * 4,
                     cudaMemcpyDeviceToHost);

    auto output0Memcpy = high_resolution_clock::now();

    if (ret != cudaSuccess)
    {
        throw std::runtime_error("Failed to memcpy to output0Copy: " +
                                 std::string(cudaGetErrorName(ret)) + '\n' +
                                 std::string(cudaGetErrorString(ret)));
    }

    std::vector<std::vector<Detection>> classDetections(numClasses_);
    std::vector<Detection> detections;
    for (int i = 0; i < 8400; i++)
    {
        // Detection detection;
        float maxConf = -1;
        int classId = 0;
        for (int j = 0; j < numClasses_; j++)
        {
            float conf = output0Copy[i + (j + 4) * 8400];
            if (conf > maxConf)
            {
                classId = j;
                maxConf = conf;
            }
        }
        if (maxConf > confThreshold_)
        {
            // printf("%f\n", )
            float normBbox[4];
            normBbox[0] = output0Copy[i];
            normBbox[1] = output0Copy[i + 8400];
            normBbox[2] = output0Copy[i + 2 * 8400];
            normBbox[3] = output0Copy[i + 3 * 8400];
            cv::Rect bbox = cv::Rect(normBbox[0] - normBbox[2] / 2,
                                     normBbox[1] - normBbox[3] / 2, normBbox[2],
                                     normBbox[3]);
            iou(classDetections, bbox, classId, maxConf, i);
        }
    }

    auto postProcess = high_resolution_clock::now();

    printf("Enqueue time: %f\n", (float)duration_cast<nanoseconds>(enqueued - start).count() / 1e6);
    printf("output0 memcpy time: %f\n", (float)duration_cast<nanoseconds>(output0Memcpy - enqueued).count() / 1e6);
    printf("postprocess time: %f\n", (float)duration_cast<nanoseconds>(postProcess - output0Memcpy).count() / 1e6);

    for (auto classDets : classDetections)
    {
        for (auto detection : classDets)
        {
            detections.push_back(detection);
        }
    }

    return detections;
}

void YoloV8Detector::iou(std::vector<std::vector<Detection>> &detections, cv::Rect newBbox, int classId, float conf, int iVal)
{
    bool changed = false;
    std::vector<Detection> &classDetections = detections[classId];
    printf("Class detections size: %d\n", classDetections.size());

    for (int i = 0; i < classDetections.size(); i++)
    {
        float area1 = classDetections[i].bbox().width * classDetections[i].bbox().height;
        float area2 = newBbox.width * newBbox.height;

        /* If points are (x1, y1) (a1, b1) and (x2, y2) (a2, b2)
           Then the intersection width would be the furthest left "right edge"
           minus the furthest right "left edge"
        */
        float xx = std::min<float>(classDetections[i].bbox().x - (classDetections[i].bbox().width / 2), newBbox.x - (newBbox.width / 2));
        float yy = std::max<float>(classDetections[i].bbox().y - (classDetections[i].bbox().height / 2), newBbox.y - (newBbox.height / 2));
        float aa = std::min<float>(classDetections[i].bbox().x + (classDetections[i].bbox().width / 2), newBbox.x + (classDetections[i].bbox().width / 2));
        float bb = std::min<float>(classDetections[i].bbox().y + (classDetections[i].bbox().height / 2), newBbox.y + (classDetections[i].bbox().height / 2));

        float intersectionWidth = std::max<float>(0, (aa - xx));
        float intersectionHeight = std::max<float>(0, (bb - yy));

        float intersectionArea = intersectionWidth * intersectionHeight;

        float unionArea = area1 + area2 - intersectionArea;
        float iou = intersectionArea / unionArea;

        if (iou > iouThreshold_)
        {
            if (classDetections[i].confidence() < conf)
            {
                for (int j = 0; j < 32; j++)
                {
                    cudaMemcpy(maskWeights + j, output0Copy + (iVal + (4 + numClasses_ + j) * 8400), sizeof(float), cudaMemcpyHostToDevice);
                }
                classDetections[i] = Detection(classId, conf, newBbox, calculateMask(maskWeights));
                changed = true;
            }
            else
            {
                changed = true;
            }
        }
    }
    if (!changed)
    {
        for (int i = 0; i < 32; i++)
        {
            cudaMemcpy(maskWeights + i, output0Copy + (iVal + (4 + numClasses_ + i) * 8400), sizeof(float), cudaMemcpyHostToDevice);
        }
        classDetections.push_back(Detection(classId, conf, newBbox, calculateMask(maskWeights)));
    }
}

cv::Mat YoloV8Detector::calculateMask(float *maskWeights)
{
    unsigned int gridRows = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int gridCols = (160 * 160 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(gridCols, gridRows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(maskWeights, output1, mask, 1, 32, 160 * 160);
    cudaDeviceSynchronize();
    cv::Mat cpuMask(160, 160, CV_32FC1);

    cudaMemcpy(cpuMask.data, mask, 160 * 160 * sizeof(float), cudaMemcpyDeviceToHost);

    // double maxVal;
    // cv::minMaxLoc(cpuMask, nullptr, &maxVal);

    // printf("Max value: %f\n", maxVal);
    std::transform(cpuMask.begin<float>(), cpuMask.end<float>(), cpuMask.begin<float>(), sigmoid);

    return cpuMask;
}