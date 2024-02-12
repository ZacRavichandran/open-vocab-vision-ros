cmake_minimum_required(VERSION 3.0.2)
project(lang_seg_ros)

find_package(catkin REQUIRED)

catkin_package(
)

include_directories(
)

catkin_install_python(PROGRAMS
  scripts/infer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
