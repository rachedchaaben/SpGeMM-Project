cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(spgemm VERSION 1.0)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()


include(GoogleTest)
find_package(Ginkgo REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(
    spgemm
    spgemm.cpp)
target_link_libraries(spgemm Ginkgo::ginkgo gtest_main benchmark::benchmark)

add_executable(
    omp_spgemm
    omp_spgemm.cpp)
target_link_libraries(omp_spgemm Ginkgo::ginkgo gtest_main benchmark::benchmark  OpenMP::OpenMP_CXX)
