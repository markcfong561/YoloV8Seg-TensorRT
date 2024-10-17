cmake_minimum_required(VERSION 3.12)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

enable_language(CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED COMPONENTS cudart)
find_package(TensorRT REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} include)

add_library(yolov8seg-tensorrt src/yolov8seg.cu)
target_link_libraries(yolov8seg-tensorrt ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
