#!/usr/bin/env python
"""Localization finite-difference baseline for observation-only bring-up."""
from __future__ import division

import math
import os
import sys

import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from racecar_policy_core import PoseVelocityEstimator, finite_array


class PoseVelocityNode(object):
    def __init__(self):
        self.estimator = PoseVelocityEstimator(rospy.get_param("~base_to_cog_x", 0.0),
                                              rospy.get_param("~velocity_filter_time_constant", 0.1))
        self.map_frame = rospy.get_param("~map_frame", "map").lstrip("/")
        self.state_frame = rospy.get_param("~state_frame", "base_link")
        self.max_xy_variance = rospy.get_param("~max_pose_xy_variance", 0.04)
        self.max_yaw_variance = rospy.get_param("~max_pose_yaw_variance", 0.04)
        self.publisher = rospy.Publisher(rospy.get_param("~state_topic"), Odometry, queue_size=1)
        rospy.Subscriber(rospy.get_param("~pose_topic"), PoseWithCovarianceStamped, self.receive, queue_size=1)
        rospy.logwarn("Pose derivative baseline is for dry runs; validate a high-rate fused estimator before driving")

    def receive(self, message):
        try:
            if message.header.frame_id.lstrip("/") != self.map_frame:
                raise ValueError("Pose is not in the configured map frame")
            variance = finite_array([message.pose.covariance[i] for i in (0, 7, 35)])
            if any(value < 0 for value in variance) or variance[0] > self.max_xy_variance or variance[1] > self.max_xy_variance or variance[2] > self.max_yaw_variance:
                raise ValueError("Pose covariance too large")
            q = message.pose.pose.orientation
            finite_array([q.x, q.y, q.z, q.w])
            if abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1.0) > 0.05:
                raise ValueError("Invalid pose quaternion")
            yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            estimate = self.estimator.update(message.header.stamp.to_sec(), message.pose.pose.position.x,
                                             message.pose.pose.position.y, yaw)
            if estimate is None:
                return
            x, y, yaw, vx, vy = estimate
            state = Odometry()
            state.header = message.header
            state.child_frame_id = self.state_frame
            state.pose.pose.position.x, state.pose.pose.position.y = x, y
            state.pose.pose.orientation.z, state.pose.pose.orientation.w = math.sin(yaw/2.0), math.cos(yaw/2.0)
            state.pose.covariance = message.pose.covariance
            state.twist.twist.linear.x, state.twist.twist.linear.y = vx, vy
            self.publisher.publish(state)
        except Exception as error:
            self.estimator.previous = None
            self.estimator.velocity = None
            rospy.logwarn_throttle(2.0, "Pose estimate withheld: %s" % error)


if __name__ == "__main__":
    rospy.init_node("racecar_pose_velocity")
    node = PoseVelocityNode()
    rospy.spin()
