cmake_minimum_required(VERSION 3.0.2)
project(stsm_madp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  gazebo_msgs
  nav_msgs
  sensor_msgs
  trajectory_msgs
  tf
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs gazebo_msgs nav_msgs sensor_msgs trajectory_msgs tf visualization_msgs
)

catkin_install_python(PROGRAMS
  nodes/handover_node.py
  nodes/wheelchair_node.py
  nodes/metrics_node.py
  nodes/social_field_viz_node.py
  scripts/run_experiment.py
  scripts/visualization/generate_results_figures.py
  scripts/analysis/results_manager.py
  scripts/analysis/train_adp_critic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds urdf config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_arm_interest_gate.py)
  catkin_add_nosetests(test/test_interest_point_constraints.py)
endif()
