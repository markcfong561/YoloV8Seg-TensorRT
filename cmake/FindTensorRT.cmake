if (NOT TensorRT_FOUND)
    set(TensorRT_ROOT /usr)
    set(TR_INFER_LIB libnvinfer.a)
    set(TR_INFER_RT libnvinfer.so)
    set(TR_PARSERS libnvparsers.so)
    set(TR_ONNX_PARSERS libnvonnxparser.so)
    set(TR_INFER_PLUGIN_RT libnvinfer_plugin.so)

    find_path(TensorRT_INCLUDE_DIR NvInfer.h
        PATHS ${TensorRT_ROOT} ${TensorRT_ROOT}/include ${TensorRT_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}
        $ENV{TensorRT_ROOT} $ENV{TensorRT_ROOT}/include
        $ENV{TensorRT_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}
        NO_DEFAULT_PATH
        )
    find_path(TensorRT_LIBRARY_DIR NAMES ${TR_INFER_LIB} ${TR_INFER_RT}
        PATHS ${TensorRT_ROOT} ${TensorRT_ROOT}/lib
        ${TensorRT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}
        $ENV{TensorRT_ROOT} $ENV{TensorRT_ROOT}/lib
        $ENV{TensorRT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}
        NO_DEFAULT_PATH
        DOC “Path to TensorRT library.”
        )

    find_library(TensorRT_LIBRARY NAMES ${TR_INFER_LIB} ${TR_INFER_RT}
        PATHS ${TensorRT_LIBRARY_DIR}
        NO_DEFAULT_PATH
        DOC “Path to TensorRT library.”)

    find_library(TensorRT_LIBRARY_PARSERS NAMES ${TR_PARSERS}
        PATHS ${TensorRT_LIBRARY_DIR}
        NO_DEFAULT_PATH
        DOC “Path to TensorRT library.”)

    find_library(TensorRT_LIBRARY_PARSERS_ONNX NAMES ${TR_ONNX_PARSERS}
        PATHS ${TensorRT_LIBRARY_DIR}
        NO_DEFAULT_PATH
        DOC “Path to TensorRT library.”)

    find_library(TensorRT_LIBRARY_PLUGIN NAMES ${TR_INFER_PLUGIN_RT}
        PATHS ${TensorRT_LIBRARY_DIR}
        NO_DEFAULT_PATH
        DOC “Path to TensorRT library.”)

    list(APPEND TensorRT_LIBRARY ${TensorRT_LIBRARY_PARSERS} ${TensorRT_LIBRARY_PLUGIN} ${TensorRT_LIBRARY_PARSERS_ONNX})
    if (TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
        set(TensorRT_FOUND ON)
    else ()
        set(TensorRT_FOUND OFF)
    endif ()

    if (TensorRT_FOUND)
        file(READ ${TensorRT_INCLUDE_DIR}/NvInferVersion.h TensorRT_VERSION_FILE_CONTENTS)
        string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TensorRT_MAJOR_VERSION "${TensorRT_VERSION_FILE_CONTENTS}")
        if ("${TensorRT_MAJOR_VERSION}" STREQUAL "")
            file(READ ${TensorRT_INCLUDE_DIR}/NvInferVersion.h TensorRT_VERSION_FILE_CONTENTS)
            string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TensorRT_MAJOR_VERSION "${TensorRT_VERSION_FILE_CONTENTS}")
        endif ()
        if ("${TensorRT_MAJOR_VERSION}" STREQUAL "")
            message(SEND_ERROR "Failed to detect TensorRT version.")
        endif ()
        string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)" TensorRT_MINOR_VERSION "${TensorRT_VERSION_FILE_CONTENTS}")
        string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)" TensorRT_PATCH_VERSION "${TensorRT_VERSION_FILE_CONTENTS}")

        string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1" TensorRT_MAJOR_VERSION "${TensorRT_MAJOR_VERSION}")
        string(REGEX REPLACE "define NV_TENSORRT_MINOR +([0-9]+)" "\\1" TensorRT_MINOR_VERSION "${TensorRT_MINOR_VERSION}")
        string(REGEX REPLACE "define NV_TENSORRT_PATCH +([0-9]+)" "\\1" TensorRT_PATCH_VERSION "${TensorRT_PATCH_VERSION}")
        set(TensorRT_VERSION "${TensorRT_MAJOR_VERSION}.${TensorRT_MINOR_VERSION}.${TensorRT_PATCH_VERSION}")
        set(TensorRT_VERSION_STRING "${TensorRT_VERSION}")
        message(DEBUG "Current TensorRT header is ${TensorRT_INCLUDE_DIR}/NvInfer.h. " "Current TensorRT version is v${TensorRT_VERSION}. ")
    endif ()
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}" CACHE INTERNAL "The path to TensorRT header files.")
    set(TensorRT_VERSION "${TensorRT_VERSION}" CACHE INTERNAL "The TensorRT version")
    set(TensorRT_LIBRARIES "${TensorRT_LIBRARY}" CACHE INTERNAL "The libraries of TensorRT")
    set(TensorRT_LIBRARY_DIR "${TensorRT_LIBRARY_DIR}" CACHE INTERNAL "The library directory of TensorRT")
    set(TensorRT_FOUND "${TensorRT_FOUND}" CACHE BOOL "A flag stating if TensorRT has been found")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
        TensorRT
        FOUND_VAR
        TensorRT_FOUND
        REQUIRED_VARS
        TensorRT_LIBRARIES
        TensorRT_INCLUDE_DIRS
        TensorRT_LIBRARY_DIR
        VERSION_VAR
        TensorRT_VERSION
    )
    mark_as_advanced(TensorRT_MAJOR_VERSION TensorRT_INCLUDE_DIR TensorRT_LIBRARY_DIR TensorRT_LIBRARY TensorRT_LIBRARY_DIR TensorRT_FOUND)
endif ()