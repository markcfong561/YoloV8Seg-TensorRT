cmake_minimum_required(VERSION 3.12)
project(cudaTest LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# SET(CUDA_NVCC_FLAGS "-arch=sm_75" CACHE STRING "nvcc flags" FORCE)
# set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)

# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
find_package(TensorRT REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} include)

add_executable(cuda src/testCuda.cpp src/yolov8seg.cu)
target_link_libraries(cuda ${OpenCV_LIBS} ${TensorRT_LIBRARIES} CUDA::cudart ${TensorRT_LIBRARIES})
# cuda_add_library(cuda src/postprocess.cu)
