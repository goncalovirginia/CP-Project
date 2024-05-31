cmake_minimum_required(VERSION 3.16)
project(project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_HOST_COMPILER /usr/bin/gcc-10)
set(CMAKE_CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER})

# Autodetect CUDA architecture(s)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message("-- CUDA Architecture(s): " ${CMAKE_CUDA_ARCHITECTURES})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

###############
# libWB
###############

include_directories(${PROJECT_SOURCE_DIR}/libwb)
include(${PROJECT_SOURCE_DIR}/libwb/sources.cmake)
add_library(wb STATIC ${LIBWB_SOURCE_FILES})
target_link_libraries(wb ${CUDA_LIBRARIES})

###############
# project - sequential
###############

include_directories(include)

add_executable(project_sequential
        src/main.cpp
        src/histogram_eq.cpp
)

target_link_libraries(project_sequential wb)

####################
## project - parallel
####################

find_package(OpenMP REQUIRED)

add_executable(project_parallel
        src/main.cpp
        src/histogram_eq_parallel.cpp
)

target_link_libraries(project_parallel PUBLIC OpenMP::OpenMP_CXX wb)

####################
## project - parallel vectorized
####################

find_package(OpenMP REQUIRED)

add_executable(project_parallel_vectorized
        src/main.cpp
        src/histogram_eq_parallel.cpp
)

target_link_libraries(project_parallel_vectorized OpenMP::OpenMP_CXX wb)
target_compile_options(project_parallel_vectorized PRIVATE -O2 -ftree-vectorize -fopt-info-vec-optimized)

####################
## CUDA
####################

find_package(CUDA REQUIRED)

cuda_add_executable(project_cuda
        src/main.cpp
        src/histogram_eq_cuda.cu
)

target_link_libraries(project_cuda OpenMP::OpenMP_CXX wb ${CUDA_LIBRARIES})
target_compile_options(project_parallel_vectorized PRIVATE -O2 -ftree-vectorize -fopt-info-vec-optimized)

##############################################################
# Tests
##############################################################

enable_testing()

#############
# Download and unpack googletest at configure time
#############

set (GTEST_HOME ${CMAKE_CURRENT_BINARY_DIR}/googletest)
message ("-- Loading Google Test from ${GTEST_HOME}")
configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )

if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )

if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${GTEST_HOME}/src ${GTEST_HOME}/build EXCLUDE_FROM_ALL)
set (GTEST_BOTH_LIBRARIES gtest_main gtest)
add_subdirectory (test)
