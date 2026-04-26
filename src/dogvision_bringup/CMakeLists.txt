cmake_minimum_required(VERSION 3.0.2)
project(dogvision_bringup)

find_package(catkin REQUIRED)

catkin_package()

# 安装 launch 和 config 目录（供 roslaunch 和 rospack 找到）
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
