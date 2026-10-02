#!/usr/bin/env python
"""ROS1 actor adapter. Dry run publishes only private diagnostic topics."""
from __future__ import division

import ctypes
import hashlib
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
import tf
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from vesc_msgs.msg import VescStateStamped

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from racecar_policy_core import CommandGate, NumpyPolicy, ObservationBuilder, decode_action, finite_array, load_config


class Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]


def monotonic_clock():
    # Python 2 has no time.monotonic. Linux CLOCK_MONOTONIC is independent of NTP.
    if hasattr(time, "monotonic"):
        return time.monotonic
    library = ctypes.CDLL("librt.so.1", use_errno=True)
    def read():
        value = Timespec()
        if library.clock_gettime(1, ctypes.byref(value)):
            raise RuntimeError("clock_gettime failed")
        return value.seconds + value.nanoseconds * 1e-9
    return read


class PolicyNode(object):
    def __init__(self):
        self.dry_run = bool(rospy.get_param("~dry_run", True))
        self.clock = monotonic_clock()
        self.gate_lock = threading.Lock()
        self.sensor_lock = threading.Lock()
        self.sensors = {}
        self.stop_event = threading.Event()
        self.gate = CommandGate(rospy.get_param("~command_timeout", 0.1), self.clock)
        self.sensor_timeout = float(rospy.get_param("~sensor_timeout", 0.1))
        self.max_skew = float(rospy.get_param("~max_sensor_skew", 0.04))
        self.map_frame = rospy.get_param("~map_frame", "map").lstrip("/")
        self.state_frame = rospy.get_param("~state_frame", "base_link").lstrip("/")
        self.base_frame = rospy.get_param("~base_frame", "base_link").lstrip("/")
        self.pole_pairs = float(rospy.get_param("~motor_pole_pairs", 0))
        self.gear_ratio = float(rospy.get_param("~motor_to_wheel_ratio", 0))
        self.erpm_sign = float(rospy.get_param("~erpm_sign", -1))
        if self.erpm_sign not in (-1, 1) or self.sensor_timeout <= 0 or self.max_skew < 0:
            raise ValueError("Invalid sensor calibration")
        self.limits = dict((key, float(rospy.get_param("~" + key))) for key in
                           ("speed_min", "speed_max", "steer_min", "steer_max"))
        finite_array(list(self.limits.values()))
        if self.limits["speed_min"] > 0 or self.limits["speed_max"] < 0 or self.limits["steer_min"] > 0 or self.limits["steer_max"] < 0:
            raise ValueError("Limits must contain physical zero")
        if not self.dry_run:
            required = ("calibration_confirmed", "state_estimator_confirmed", "wheel_speed_estimate_confirmed")
            if not all(rospy.get_param("~" + key, False) for key in required):
                raise ValueError("Live output requires validated calibration, state estimator and wheel-speed mapping")
            if rospy.get_param("/use_sim_time", False):
                raise ValueError("Live output requires wall-clock ROS timestamps")
        bundle = rospy.get_param("~bundle_dir")
        config = load_config(os.path.join(bundle, "contract.json"))
        for filename, expected in config["sha256"].items():
            with open(os.path.join(bundle, filename), "rb") as stream:
                actual = hashlib.sha256(stream.read()).hexdigest()
            if actual != expected:
                raise ValueError("Bundle checksum mismatch: " + filename)
        with np.load(os.path.join(bundle, "track.npz"), allow_pickle=False) as track:
            self.builder = ObservationBuilder(config, track)
        self.policy = NumpyPolicy(os.path.join(bundle, "actor.npz"), config["actor_observation_size"])
        self.config = config
        self.tf_listener = tf.TransformListener()
        self.obs_pub = rospy.Publisher("~observation", Float32MultiArray, queue_size=1)
        self.action_pub = rospy.Publisher("~proposed_drive", AckermannDriveStamped, queue_size=1)
        self.drive_pub = None if self.dry_run else rospy.Publisher(
            rospy.get_param("~drive_topic"), AckermannDriveStamped, queue_size=1)
        rospy.Subscriber(rospy.get_param("~state_topic"), Odometry, self.receive, callback_args="state", queue_size=1)
        rospy.Subscriber(rospy.get_param("~imu_topic"), Imu, self.receive, callback_args="imu", queue_size=1)
        rospy.Subscriber(rospy.get_param("~vesc_topic"), VescStateStamped, self.receive, callback_args="vesc", queue_size=1)
        rospy.on_shutdown(self.shutdown)
        self.watchdog = threading.Thread(target=self.publish_loop)
        self.watchdog.daemon = True
        self.watchdog.start()
        self.timer = rospy.Timer(rospy.Duration(1.0 / config["policy_hz"]), self.infer)
        rospy.loginfo("Policy ready: %d inputs, %.1f Hz, dry_run=%s", self.policy.obs_dim, config["policy_hz"], self.dry_run)

    def receive(self, message, name):
        with self.sensor_lock:
            self.sensors[name] = (message, self.clock())

    def snapshot(self):
        with self.sensor_lock:
            messages = dict(self.sensors)
        if set(messages) != set(("state", "imu", "vesc")):
            raise ValueError("Waiting for state estimate, IMU and VESC telemetry")
        now, now_ros = self.clock(), rospy.Time.now().to_sec()
        stamps = []
        for message, received in messages.values():
            stamp = message.header.stamp.to_sec()
            if stamp <= 0 or not (0 <= now - received <= self.sensor_timeout) or not (0 <= now_ros - stamp <= self.sensor_timeout):
                raise ValueError("Stale or future sensor timestamp")
            stamps.append(stamp)
        if max(stamps) - min(stamps) > self.max_skew:
            raise ValueError("Sensor timestamps are not aligned")
        return dict((name, pair[0]) for name, pair in messages.items())

    def infer(self, event):
        try:
            messages = self.snapshot()
            state, imu, vesc = messages["state"], messages["imu"], messages["vesc"]
            if state.header.frame_id.lstrip("/") != self.map_frame or state.child_frame_id.lstrip("/") != self.state_frame:
                raise ValueError("State frame mismatch; expected map pose and CoG body velocity")
            if imu.angular_velocity_covariance[0] == -1 or vesc.state.fault_code != 0:
                raise ValueError("IMU unavailable or VESC fault")
            if self.pole_pairs <= 0 or self.gear_ratio <= 0:
                raise ValueError("Configure measured motor_pole_pairs and motor_to_wheel_ratio")
            vector = Vector3Stamped()
            vector.header = imu.header
            vector.vector = imu.angular_velocity
            gyro = self.tf_listener.transformVector3(self.base_frame, vector)
            q = state.pose.pose.orientation
            finite_array([q.x, q.y, q.z, q.w])
            if abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1.0) > 0.05:
                raise ValueError("Invalid state orientation")
            yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            wheel_omega = self.erpm_sign * vesc.state.speed * 2.0 * math.pi / (60.0 * self.pole_pairs * self.gear_ratio)
            observation = self.builder.push([state.pose.pose.position.x, state.pose.pose.position.y, yaw,
                                             state.twist.twist.linear.x, state.twist.twist.linear.y,
                                             gyro.vector.z, wheel_omega])
            command = decode_action(self.policy.predict(observation), self.config, self.limits)
            self.snapshot()  # Inference may have taken longer than the sensor timeout.
            with self.gate_lock:
                self.gate.update(command)
            self.obs_pub.publish(Float32MultiArray(data=observation.tolist()))
        except Exception as error:
            with self.gate_lock:
                self.gate.invalidate()
            self.builder.reset()
            rospy.logwarn_throttle(2.0, "Policy output stopped: %s" % error)

    def publish_loop(self):
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                self.snapshot()
            except Exception:
                with self.gate_lock:
                    self.gate.invalidate()
            with self.gate_lock:
                speed, angle = self.gate.get()
            message = AckermannDriveStamped()
            message.header.stamp = rospy.Time.now()
            message.drive.speed, message.drive.steering_angle = speed, angle
            self.action_pub.publish(message)
            if self.drive_pub is not None:
                self.drive_pub.publish(message)
            self.stop_event.wait(0.02)

    def shutdown(self):
        self.stop_event.set()
        if hasattr(self, "watchdog"):
            self.watchdog.join(0.2)
        if self.drive_pub is not None:
            message = AckermannDriveStamped()
            message.header.stamp = rospy.Time.now()
            self.drive_pub.publish(message)  # Physical speed zero, never normalized action zero.


if __name__ == "__main__":
    rospy.init_node("racecar_policy")
    try:
        node = PolicyNode()
        rospy.spin()
    except Exception as error:
        rospy.logfatal("Policy startup failed: %s", error)
        sys.exit(1)
