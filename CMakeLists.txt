cmake_minimum_required(VERSION 3.0.2)
project(open_vocab_vision_ros)

find_package(catkin REQUIRED COMPONENTS
rospy
message_generation
geometry_msgs
std_msgs
)

add_service_files(
  FILES
  Query.srv
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
  scripts/grounding_dino_infer_node.py
  scripts/vlm_infer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
