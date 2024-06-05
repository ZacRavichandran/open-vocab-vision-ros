cmake_minimum_required(VERSION 3.0.2)
project(lang_seg_ros)

find_package(catkin REQUIRED
rospy
)

catkin_package(
)

include_directories(
)

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/lseg_infer.py
  scripts/yolo_infer_node.py
  scripts/tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
