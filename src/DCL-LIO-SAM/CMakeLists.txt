cmake_minimum_required(VERSION 3.8)
project(dcl_lio_sam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(dcl_slam REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system thread)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(libnabo REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CloudInfo.msg"
  "srv/SaveMap.srv"
  DEPENDENCIES geometry_msgs std_msgs nav_msgs sensor_msgs
)

set(dependencies
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  cv_bridge
  pcl_conversions
  pcl_ros
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  dcl_slam
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${dcl_slam_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS} ${dcl_slam_LIBRARY_DIRS})
# Link path for dcl_slam_lib
find_library(DCL_SLAM_LIB dcl_slam_lib PATHS ${dcl_slam_DIR}/../../../lib NO_DEFAULT_PATH)
if(NOT DCL_SLAM_LIB)
  # Fallback: link by name and add the install lib dir
  link_directories(${dcl_slam_DIR}/../../../lib)
endif()

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
ament_target_dependencies(${PROJECT_NAME}_imageProjection ${dependencies})
target_link_libraries(${PROJECT_NAME}_imageProjection ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
ament_target_dependencies(${PROJECT_NAME}_featureExtraction ${dependencies})
target_link_libraries(${PROJECT_NAME}_featureExtraction ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
ament_target_dependencies(${PROJECT_NAME}_mapOptmization ${dependencies})
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization
  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS}
  libnabo::nabo gtsam glog ${DCL_SLAM_LIB} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
ament_target_dependencies(${PROJECT_NAME}_imuPreintegration ${dependencies})
target_link_libraries(${PROJECT_NAME}_imuPreintegration
  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam "${cpp_typesupport_target}")

install(TARGETS
  ${PROJECT_NAME}_imageProjection
  ${PROJECT_NAME}_featureExtraction
  ${PROJECT_NAME}_mapOptmization
  ${PROJECT_NAME}_imuPreintegration
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
