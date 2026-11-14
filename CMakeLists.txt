cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW) # honour upper-case <PACKAGE>_ROOT env vars such as TENSORRT_ROOT
endif()
project(cuRF-Stream VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# AUTO: build CUDA parts when a CUDA compiler is found; ON: require it; OFF: CPU only.
set(CURF_CUDA AUTO CACHE STRING "Build CUDA components (AUTO/ON/OFF)")
set_property(CACHE CURF_CUDA PROPERTY STRINGS AUTO ON OFF)
option(CURF_TENSORRT "Build TensorRT baseline components" OFF)
option(CURF_OMNICORE "Build the OmniCore hardware source" OFF)
option(CURF_TESTS "Build tests" ON)
option(CURF_BENCH "Build benchmarks" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CURF_HAVE_CUDA OFF)
if(NOT CURF_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 86)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(CURF_HAVE_CUDA ON)
    elseif(CURF_CUDA STREQUAL "ON")
        message(FATAL_ERROR "CURF_CUDA=ON but no CUDA compiler was found")
    endif()
endif()
message(STATUS "cuRF-Stream: CUDA ${CURF_HAVE_CUDA}")

if(CURF_TENSORRT)
    if(NOT CURF_HAVE_CUDA)
        message(FATAL_ERROR "CURF_TENSORRT requires CUDA")
    endif()
    find_package(TensorRT REQUIRED)
endif()

add_library(curf_warnings INTERFACE)
target_compile_options(curf_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --expt-relaxed-constexpr -lineinfo>)

# CPU core: source interfaces, synthetic/file sources, reference implementations.
find_package(Threads REQUIRED)

add_library(curf_core
    src/engine/model.cpp
    src/reference/unpack.cpp
    src/runtime/host_ring.cpp
    src/source/synthetic_source.cpp
    src/source/file_source.cpp)
target_include_directories(curf_core PUBLIC include)
target_link_libraries(curf_core PUBLIC Threads::Threads PRIVATE curf_warnings)

if(CURF_HAVE_CUDA)
    add_library(curf_cuda
        src/kernels/unpack.cu
        src/kernels/psd.cu
        src/kernels/conv1d.cu
        src/kernels/conv1d_mma.cu
        src/kernels/block_fused.cu
        src/engine/engine.cpp
        src/runtime/pipeline.cpp)
    target_include_directories(curf_cuda PUBLIC include)
    target_link_libraries(curf_cuda PUBLIC curf_core CUDA::cudart CUDA::cufft PRIVATE curf_warnings)
    if(TARGET CUDA::nvtx3)
        target_link_libraries(curf_cuda PRIVATE CUDA::nvtx3)
    else()
        target_link_libraries(curf_cuda PRIVATE ${CMAKE_DL_LIBS}) # header-only NVTX3 uses dlopen
    endif()
    target_compile_definitions(curf_cuda PUBLIC CURF_HAVE_CUDA=1)
endif()

if(CURF_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
if(CURF_BENCH)
    add_subdirectory(bench)
endif()
