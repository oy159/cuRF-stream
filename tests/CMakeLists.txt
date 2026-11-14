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

add_executable(curf_tests_cpu
    test_reference.cpp
    test_sources.cpp)
target_link_libraries(curf_tests_cpu PRIVATE curf_core curf_warnings Catch2::Catch2WithMain)
catch_discover_tests(curf_tests_cpu TEST_PREFIX "cpu.")

if(CURF_HAVE_CUDA)
    add_executable(curf_tests_cuda
        test_unpack_cuda.cpp)
    target_link_libraries(curf_tests_cuda PRIVATE curf_cuda curf_warnings Catch2::Catch2WithMain)
    catch_discover_tests(curf_tests_cuda TEST_PREFIX "cuda.")
endif()
