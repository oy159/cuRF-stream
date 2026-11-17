find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
        URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(Catch2)
endif()
include(Catch)

# Converted model + golden activations (model/convert.py); tests that need them skip when absent.
set(CURF_MODEL_DIR "${PROJECT_SOURCE_DIR}/model/artifacts/run2/export" CACHE PATH "Directory with amc.curf / amc.golden")

add_executable(curf_tests_cpu
    test_adsb.cpp
    test_extraction.cpp
    test_reference.cpp
    test_sources.cpp
    test_host_ring.cpp
    test_engine_model.cpp)
target_link_libraries(curf_tests_cpu PRIVATE curf_core curf_warnings Catch2::Catch2WithMain)
target_compile_definitions(curf_tests_cpu PRIVATE CURF_MODEL_DIR="${CURF_MODEL_DIR}")
catch_discover_tests(curf_tests_cpu TEST_PREFIX "cpu.")

if(CURF_HAVE_CUDA)
    add_executable(curf_tests_cuda
        test_adsb_cuda.cpp
        test_extraction_cuda.cpp
        test_extraction_bank_cuda.cpp
        test_unpack_cuda.cpp
        test_pipeline_cuda.cpp
        test_engine_cuda.cpp
        test_channelizer_cuda.cpp
        test_windows_cuda.cpp)
    target_link_libraries(curf_tests_cuda PRIVATE curf_cuda curf_warnings Catch2::Catch2WithMain)
    target_compile_definitions(curf_tests_cuda PRIVATE CURF_MODEL_DIR="${CURF_MODEL_DIR}")
    catch_discover_tests(curf_tests_cuda TEST_PREFIX "cuda." DISCOVERY_TIMEOUT 60)
endif()
