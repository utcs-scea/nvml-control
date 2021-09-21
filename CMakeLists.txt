cmake_minimum_required(VERSION 3.15)
project(nvml_control CXX)

# find_package(CUDA COMPONENTS nvml REQUIRED)  # Requires CMAKE 3.17... instead link manually
add_library(nvml SHARED IMPORTED)
set_target_properties(nvml PROPERTIES IMPORTED_LOCATION /usr/local/cuda/lib64/stubs/libnvidia-ml.so)
target_include_directories(nvml INTERFACE /usr/local/cuda/include)

file(GLOB SRCS src/*.cpp)
add_library(nvml_control STATIC ${SRCS})
target_include_directories(nvml_control PUBLIC include)
target_link_libraries(nvml_control PUBLIC nvml)

include(cmake/ExternalGTest.cmake)
add_subdirectory(test)
