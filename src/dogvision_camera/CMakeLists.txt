cmake_minimum_required(VERSION 3.0.2)
project(dogvision_camera)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(OpenCV REQUIRED)

# MVS (Hikvision) SDK
find_library(MVS_LIB MvCameraControl
  PATHS /opt/MVS/lib /opt/MVS/lib/64 /opt/MVS/lib/32
  NO_DEFAULT_PATH
)
if(NOT MVS_LIB)
  message(WARNING "MvCameraControl not found. Hikvision camera will not link.")
  set(MVS_LIB "")
endif()

catkin_package(
  INCLUDE_DIRS include /opt/MVS/include
  LIBRARIES dogvision_camera_lib
  CATKIN_DEPENDS roscpp
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /opt/MVS/include
)

add_library(dogvision_camera_lib
  src/hikvision.cpp
  src/usbcam.cpp
)
target_compile_features(dogvision_camera_lib PUBLIC cxx_std_17)
add_dependencies(dogvision_camera_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dogvision_camera_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${MVS_LIB}
)
