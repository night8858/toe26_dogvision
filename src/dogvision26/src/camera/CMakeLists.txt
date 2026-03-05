cmake_minimum_required(VERSION 3.10)

project(grab_test)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(OpenCV REQUIRED)
include_directories(/opt/MVS/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

link_directories(/opt/MVS/lib/aarch64)
add_executable(grab_test grab_test.cpp hikvision.cpp usbcam.cpp)
target_link_libraries(grab_test MvCameraControl ${OpenCV_LIBRARIES})