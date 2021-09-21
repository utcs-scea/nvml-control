cmake_minimum_required(VERSION 3.5)
project(nvml_control_tests)

file(GLOB TESTS test_*.cpp)
add_executable(${PROJECT_NAME} ${TESTS})
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} nvml_control)
