cmake_minimum_required(VERSION 3.8)
project(distributed_mapper)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/pairwise_consistency_maximization/third_parties)

# Build one combined library with everything
add_library(distributed_mapper_lib
  src/pairwise_consistency_maximization/graph_utils/graph_utils_functions.cpp
  src/pairwise_consistency_maximization/robot_measurements/robot_measurements.cpp
  src/pairwise_consistency_maximization/robot_measurements/interrobot_measurements.cpp
  src/pairwise_consistency_maximization/robot_measurements/robot_local_map.cpp
  src/pairwise_consistency_maximization/pairwise_consistency/pairwise_consistency.cpp
  src/pairwise_consistency_maximization/third_parties/fast_max-clique_finder/src/findClique.cpp
  src/pairwise_consistency_maximization/third_parties/fast_max-clique_finder/src/findCliqueHeu.cpp
  src/pairwise_consistency_maximization/third_parties/fast_max-clique_finder/src/utils.cpp
  src/pairwise_consistency_maximization/third_parties/fast_max-clique_finder/src/graphIO.cpp
  src/pairwise_consistency_maximization/global_map/global_map.cpp
  src/pairwise_consistency_maximization/distributed_pcm/distributed_pcm.cpp
  src/distributed_mapper/distributed_mapper.cpp
  src/distributed_mapper/evaluation_utils.cpp
  src/distributed_mapper/distributed_mapper_utils.cpp
  src/distributed_mapper/run_distributed_mapper.cpp
)
target_include_directories(distributed_mapper_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pairwise_consistency_maximization>
  $<BUILD_INTERFACE:${THIRD_PARTY_DIR}/fast_max-clique_finder/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(distributed_mapper_lib PUBLIC gtsam)
target_include_directories(distributed_mapper_lib PUBLIC ${GTSAM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_compile_options(distributed_mapper_lib PRIVATE -w)

# Install
install(TARGETS distributed_mapper_lib
  EXPORT distributed_mapperTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(distributed_mapperTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(GTSAM)

ament_package()
