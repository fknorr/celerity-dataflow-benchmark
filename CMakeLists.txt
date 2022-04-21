cmake_minimum_required(VERSION 3.13)
project(celerity_dataflow_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Celerity 0.3.2 REQUIRED)

add_executable(benchmark_sycl src/benchmark_sycl.cc)
add_celerity_to_target(TARGET benchmark_sycl SOURCES src/benchmark_sycl.cc)

add_executable(benchmark_celerity src/benchmark_celerity.cc)
add_celerity_to_target(TARGET benchmark_celerity SOURCES src/benchmark_celerity.cc)

find_package(Threads REQUIRED)
find_package(MPI 2.0 REQUIRED)
add_executable(benchmark_faux_celerity src/benchmark_faux_celerity.cc)
target_link_libraries(benchmark_faux_celerity PRIVATE Threads::Threads MPI::MPI_CXX)

add_executable(benchmark_mpi src/benchmark_mpi.cc)
add_celerity_to_target(TARGET benchmark_mpi SOURCES src/benchmark_mpi.cc)
