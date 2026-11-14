# FindTensorRT
# ------------
# Locates TensorRT headers and libraries without TensorRT's own CMake package
# (TensorRT 11's TensorRT-EnterpriseConfig.cmake requires the Windows builder
# resource package to be installed as well).
#
# Hints: TensorRT_ROOT (CMake variable or environment), e.g. ~/opt/tensorrt-11.4/usr
#
# Imported targets:
#   TensorRT::nvinfer        core runtime/builder
#   TensorRT::nvonnxparser   ONNX parser
# Variables:
#   TensorRT_FOUND, TensorRT_VERSION, TensorRT_INCLUDE_DIR

set(_trt_hints ${TensorRT_ROOT} $ENV{TensorRT_ROOT} $ENV{TENSORRT_ROOT})

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${_trt_hints}
    PATH_SUFFIXES include/x86_64-linux-gnu include)
find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64)
find_library(TensorRT_nvonnxparser_LIBRARY nvonnxparser
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib/x86_64-linux-gnu lib lib64)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_defs
        REGEX "^#define TRT_(MAJOR|MINOR|PATCH)_ENTERPRISE [0-9]+")
    if(NOT _trt_defs)
        file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_defs
            REGEX "^#define NV_TENSORRT_(MAJOR|MINOR|PATCH) [0-9]+")
    endif()
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH "(TRT_${_part}_ENTERPRISE|NV_TENSORRT_${_part}) ([0-9]+)" _m "${_trt_defs}")
        set(_trt_${_part} "${CMAKE_MATCH_2}")
    endforeach()
    set(TensorRT_VERSION "${_trt_MAJOR}.${_trt_MINOR}.${_trt_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()
    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
