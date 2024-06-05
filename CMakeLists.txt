cmake_minimum_required(VERSION 3.0.2)
project(lang_seg_ros)

find_package(catkin REQUIRED COMPONENTS
rospy
message_generation
geometry_msgs
std_msgs
)

include_directories(
)

add_message_files(
  FILES 
  Track.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES 
  std_msgs
  geometry_msgs
)

catkin_package(CATKIN_DEPENDS 
message_runtime
)

catkin_install_python(PROGRAMS
  scripts/lseg_infer.py
  scripts/yolo_infer_node.py
  scripts/tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
