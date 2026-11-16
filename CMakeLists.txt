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
option(CURF_CUDNN "Build the cuDNN baseline when cuDNN is found" ON)
option(CURF_CUTLASS "Build the CUTLASS baseline when CURF_CUTLASS_DIR has the headers" ON)
set(CURF_CUTLASS_DIR "$ENV{HOME}/opt/cutlass-4.5.0/include" CACHE PATH "CUTLASS include directory")
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
    src/adsb/decode.cpp
    src/engine/model.cpp
    src/reference/channelizer.cpp
    src/reference/unpack.cpp
    src/runtime/host_ring.cpp
    src/runtime/producer.cpp
    src/source/synthetic_source.cpp
    src/source/file_source.cpp)
target_include_directories(curf_core PUBLIC include)
target_link_libraries(curf_core PUBLIC Threads::Threads PRIVATE curf_warnings)

# Live hardware source: an installed OmniCore Host (OmniCoreHost_DIR / CMAKE_PREFIX_PATH, e.g.
# /opt/omnicore/0.8.0) or its source tree (CURF_OMNICORE_SOURCE), built here as a subproject.
set(CURF_OMNICORE_SOURCE "" CACHE PATH "OmniCore-SDR-Host source tree (empty: find_package)")
if(CURF_OMNICORE)
    if(CURF_OMNICORE_SOURCE)
        set(OCH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory(${CURF_OMNICORE_SOURCE} omnicore EXCLUDE_FROM_ALL)
    else()
        find_package(OmniCoreHost 0.8 CONFIG REQUIRED)
    endif()
    target_sources(curf_core PRIVATE src/source/omnicore_source.cpp)
    target_link_libraries(curf_core PUBLIC OmniCore::omnicore)
    target_compile_definitions(curf_core PUBLIC CURF_HAVE_OMNICORE=1)
    message(STATUS "cuRF-Stream: OmniCore source ON")
endif()

if(CURF_HAVE_CUDA)
    add_library(curf_cuda
        src/kernels/adsb.cu
        src/kernels/unpack.cu
        src/kernels/psd.cu
        src/kernels/channelizer.cu
        src/kernels/windows.cu
        src/kernels/conv1d.cu
        src/kernels/conv1d_mma.cu
        src/kernels/conv1d_int8.cu
        src/kernels/block_fused.cu
        src/kernels/block_int8.cu
        src/engine/engine.cpp
        src/runtime/pipeline.cpp
        src/runtime/classifier.cu)
    target_include_directories(curf_cuda PUBLIC include)
    target_link_libraries(curf_cuda PUBLIC curf_core CUDA::cudart CUDA::cufft PRIVATE curf_warnings)
    if(TARGET CUDA::nvtx3)
        target_link_libraries(curf_cuda PRIVATE CUDA::nvtx3)
    else()
        target_link_libraries(curf_cuda PRIVATE ${CMAKE_DL_LIBS}) # header-only NVTX3 uses dlopen
    endif()
    target_compile_definitions(curf_cuda PUBLIC CURF_HAVE_CUDA=1)

    # Library baselines for the engine (registered as conv variants "cudnn" / "cutlass.*").
    find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS /usr/include/x86_64-linux-gnu ${CUDAToolkit_INCLUDE_DIRS})
    find_library(CUDNN_LIBRARY cudnn HINTS /usr/lib/x86_64-linux-gnu ${CUDAToolkit_LIBRARY_DIR})
    if(CURF_CUDNN AND CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        target_sources(curf_cuda PRIVATE src/kernels/baseline_cudnn.cpp)
        target_include_directories(curf_cuda SYSTEM PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(curf_cuda PRIVATE ${CUDNN_LIBRARY})
        target_compile_definitions(curf_cuda PRIVATE CURF_HAVE_CUDNN=1)
        message(STATUS "cuRF-Stream: cuDNN baseline ${CUDNN_LIBRARY}")
    endif()
    if(CURF_CUTLASS AND EXISTS "${CURF_CUTLASS_DIR}/cutlass/cutlass.h")
        target_sources(curf_cuda PRIVATE src/kernels/baseline_cutlass.cu)
        target_include_directories(curf_cuda SYSTEM PRIVATE ${CURF_CUTLASS_DIR})
        target_compile_definitions(curf_cuda PRIVATE CURF_HAVE_CUTLASS=1)
        message(STATUS "cuRF-Stream: CUTLASS baseline ${CURF_CUTLASS_DIR}")
    endif()
endif()

if(CURF_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
if(CURF_BENCH)
    add_subdirectory(bench)
endif()
