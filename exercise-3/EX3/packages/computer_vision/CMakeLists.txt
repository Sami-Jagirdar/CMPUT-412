cmake_minimum_required(VERSION 2.8.3)
project(computer_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(
  FILES
  GetLaneInfo.srv
)

add_message_files(
  FILES
  LaneDistance.msg
)

generate_messages(
  DEPENDENCIES 
  std_msgs
)

catkin_package()