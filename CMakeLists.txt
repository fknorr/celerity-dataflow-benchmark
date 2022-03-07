cmake_minimum_required(VERSION 3.21)
project(celerity_dataflow_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Celerity 0.3.2 REQUIRED)

add_executable(benchmark_sycl src/benchmark_sycl.cc)
add_celerity_to_target(TARGET benchmark_sycl SOURCES src/benchmark_sycl.cc)

add_executable(benchmark_celerity src/benchmark_celerity.cc)
add_celerity_to_target(TARGET benchmark_celerity SOURCES src/benchmark_celerity.cc)
