cmake_minimum_required(VERSION 3.12)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

enable_language(CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED COMPONENTS cudart)
find_package(TensorRT REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR} include)

add_library(yolov8seg-tensorrt src/yolov8seg.cu)
target_link_libraries(yolov8seg-tensorrt ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
target_include_directories(yolov8seg-tensorrt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TensorRT_INCLUDE_DIR}
)
