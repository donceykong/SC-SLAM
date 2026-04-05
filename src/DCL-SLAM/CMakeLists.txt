cmake_minimum_required(VERSION 3.8)
project(dcl_slam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(distributed_mapper REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system thread)
find_package(libnabo REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LoopInfo.msg"
  "msg/GlobalDescriptor.msg"
  "msg/NeighborEstimate.msg"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

set(dependencies
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_conversions
  pcl_ros
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

# dcl slam library
add_library(dcl_slam_lib
  src/paramsServer.cpp
  src/distributedMappingBasis.cpp
  src/distributedMapping.cpp
  src/distributedMappingVisualization.cpp
  src/distributedLoopClosure.cpp
  src/scanContextDescriptor.cpp
  src/lidarIrisDescriptor.cpp
  src/m2dpDescriptor.cpp
)
ament_target_dependencies(dcl_slam_lib ${dependencies})
target_compile_options(dcl_slam_lib PRIVATE -Wno-return-type)
target_link_libraries(dcl_slam_lib
  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS}
  libnabo::nabo gtsam glog distributed_mapper::distributed_mapper_lib
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(dcl_slam_lib "${cpp_typesupport_target}")

add_executable(dcl_slam_loopVisualizationNode src/loopVisualizationNode.cpp)
ament_target_dependencies(dcl_slam_loopVisualizationNode ${dependencies})
target_link_libraries(dcl_slam_loopVisualizationNode
  ${PCL_LIBRARIES} gtsam "${cpp_typesupport_target}"
)

install(TARGETS dcl_slam_lib
  EXPORT dcl_slamTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS dcl_slam_loopVisualizationNode
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(dcl_slamTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp tf2 tf2_ros tf2_geometry_msgs tf2_eigen
  pcl_conversions pcl_ros
  std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
  distributed_mapper
)

ament_package()
