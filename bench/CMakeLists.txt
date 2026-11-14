add_executable(curf_bench_source bench_source.cpp)
target_link_libraries(curf_bench_source PRIVATE curf_core curf_warnings)

if(CURF_HAVE_CUDA)
    add_executable(curf_bench_unpack bench_unpack.cpp)
    target_link_libraries(curf_bench_unpack PRIVATE curf_cuda curf_warnings)
endif()
