cmake_minimum_required(VERSION 2.8.3)
project(racecar_policy)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/policy_node.py scripts/pose_velocity_node.py scripts/self_check.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES src/racecar_policy_core.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
