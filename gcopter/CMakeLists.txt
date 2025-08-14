cmake_minimum_required(VERSION 3.5)
project(gcopter)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-Wall -O3 -fPIC)

# avoid Found-related error (ref: https://stackoverflow.com/questions/18327197/opencv-is-considered-to-be-not-found)
set(decomp_rviz_plugins_FOUND 1)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(decomp_ros_msgs REQUIRED)
find_package(decomp_rviz_plugins REQUIRED)

# Include directories
include_directories(
  include
  ${DECOMP_UTIL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
)

# Library
add_library(gcopter src/global_planning.cpp)

target_link_libraries(gcopter
  ${OMPL_LIBRARIES}
)

ament_target_dependencies(gcopter
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Executable
add_executable(global_planning_node src/global_planning.cpp)

target_link_libraries(global_planning_node
  gcopter
)

# Benchmarking
# add_executable(benchmark src/benchmark.cpp)
# target_link_libraries(benchmark gcopter ${OMPL_LIBRARIES})
# ament_target_dependencies(benchmark
#   rclcpp
#   std_msgs
#   geometry_msgs
#   sensor_msgs
#   visualization_msgs
#   decomp_ros_msgs
#   decomp_rviz_plugins
# )

add_executable(minco_bench_viz src/minco_bench_viz.cpp src/mighty_solver.cpp src/lbfgs_solver_utils.cpp)
target_link_libraries(minco_bench_viz gcopter ${OMPL_LIBRARIES})
ament_target_dependencies(minco_bench_viz
  rclcpp
  visualization_msgs
  nav_msgs
  decomp_ros_msgs
  decomp_rviz_plugins
  geometry_msgs
)

ament_target_dependencies(global_planning_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  decomp_rviz_plugins
)

# Install targets
install(TARGETS
  gcopter
  global_planning_node
  # benchmark
  minco_bench_viz
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

# Export package
ament_export_include_directories(include)
ament_export_libraries(gcopter)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  ompl
  Eigen3
)

ament_package()
