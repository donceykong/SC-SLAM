cmake_minimum_required(VERSION 3.8)
project(dcl_fast_lio)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  if(N GREATER 4)
    add_definitions(-DMP_EN -DMP_PROC_NUM=3)
  elseif(N GREATER 3)
    add_definitions(-DMP_EN -DMP_PROC_NUM=2)
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(dcl_slam REQUIRED)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED)
find_package(libnabo REQUIRED)
find_package(PythonLibs QUIET)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
  DEPENDENCIES geometry_msgs
)

set(dependencies
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  pcl_conversions
  pcl_ros
  visualization_msgs
  dcl_slam
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  include
  ${dcl_slam_INCLUDE_DIRS}
)

if(PythonLibs_FOUND)
  include_directories(${PYTHON_INCLUDE_DIRS})
endif()

link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})
find_library(DCL_SLAM_LIB dcl_slam_lib PATHS ${dcl_slam_DIR}/../../../lib NO_DEFAULT_PATH)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp)
ament_target_dependencies(fastlio_mapping ${dependencies})
target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  libnabo::nabo
  gtsam
  glog
  ${DCL_SLAM_LIB}
  "${cpp_typesupport_target}"
)
if(PythonLibs_FOUND)
  target_link_libraries(fastlio_mapping ${PYTHON_LIBRARIES})
endif()

install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
