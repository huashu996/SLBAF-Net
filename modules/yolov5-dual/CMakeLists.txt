cmake_minimum_required(VERSION 2.8.3)
project(yolov5)

add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge image_transport sensor_msgs roscpp rospy std_msgs)
find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_image_transport
#  CATKIN_DEPENDS cv_bridge image_transport sensor_msgs
#  DEPENDS system_lib
)

include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS} )

catkin_install_python (PROGRAMS detect.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


