#!/usr/bin/env python
"""Synthetic ROS test on an isolated master; never start vehicle drivers."""
from __future__ import division, print_function

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

from self_check import fixture


def main():
    port = 11312
    connection = socket.socket()
    try:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Test master port already in use; refusing to reuse it")
    finally:
        connection.close()
    directory = tempfile.mkdtemp(prefix="racecar-ros-smoke-")
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % port
    os.environ["ROS_HOME"] = directory
    processes = []
    streams = []
    try:
        for name in ("master", "policy"):
            streams.append(open(os.path.join(directory, name + ".log"), "w"))
        master = subprocess.Popen(["roscore", "-p", str(port)], stdout=streams[0], stderr=streams[0], preexec_fn=os.setsid)
        processes.append(master)
        import rosgraph
        deadline = time.time() + 15
        while time.time() < deadline:
            if master.poll() is not None:
                raise RuntimeError("Isolated ROS master exited")
            try:
                rosgraph.Master("/smoke_check").getPid()
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Isolated ROS master did not start")

        import rospy
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float32MultiArray
        from vesc_msgs.msg import VescStateStamped
        import yaml

        rospy.init_node("racecar_smoke_check", disable_signals=True)
        config, track = fixture()
        config.update(contract_version=1, action_order=["steering", "speed"], policy_hz=50.0)
        np.savez(os.path.join(directory, "track.npz"), **track)
        # Synthetic actor with known action: steering 0.2, speed -0.5.
        np.savez(os.path.join(directory, "actor.npz"), activation=np.array("tanh"), layer_count=np.array(1),
                 kernel_0=np.zeros((84, 2), dtype=np.float32),
                 bias_0=np.arctanh(np.array([0.2, -0.5], dtype=np.float32)))
        config["sha256"] = {}
        for filename in ("actor.npz", "track.npz"):
            with open(os.path.join(directory, filename), "rb") as stream:
                config["sha256"][filename] = hashlib.sha256(stream.read()).hexdigest()
        with open(os.path.join(directory, "contract.json"), "w") as stream:
            json.dump(config, stream)
        source = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(source, "..", "config", "vehicle.yaml")) as stream:
            vehicle = yaml.safe_load(stream)
        prefix = "/racecar_policy_smoke"
        vehicle.update(bundle_dir=directory, dry_run=True, motor_pole_pairs=7, motor_to_wheel_ratio=3.0,
                       state_topic=prefix+"/state", imu_topic=prefix+"/imu", vesc_topic=prefix+"/vesc",
                       drive_topic=prefix+"/forbidden_drive")
        for key, value in vehicle.items():
            rospy.set_param(prefix + "/" + key, value)
        observations, commands, forbidden = [], [], []
        rospy.Subscriber(prefix+"/observation", Float32MultiArray, lambda msg: observations.append(msg))
        rospy.Subscriber(prefix+"/proposed_drive", AckermannDriveStamped, lambda msg: commands.append(msg))
        rospy.Subscriber(prefix+"/forbidden_drive", AckermannDriveStamped, lambda msg: forbidden.append(msg))
        state_pub = rospy.Publisher(prefix+"/state", Odometry, queue_size=1)
        imu_pub = rospy.Publisher(prefix+"/imu", Imu, queue_size=1)
        vesc_pub = rospy.Publisher(prefix+"/vesc", VescStateStamped, queue_size=1)
        policy = subprocess.Popen([sys.executable, os.path.join(source, "policy_node.py"),
                                   "__name:=racecar_policy_smoke"], stdout=streams[1], stderr=streams[1], preexec_fn=os.setsid)
        processes.append(policy)
        deadline = time.time()+15
        while time.time() < deadline:
            if policy.poll() is not None:
                raise RuntimeError("Policy process exited; see " + directory)
            stamp = rospy.Time.now()
            state = Odometry()
            state.header.stamp, state.header.frame_id, state.child_frame_id = stamp, "map", "base_link"
            state.pose.pose.position.x, state.pose.pose.position.y = 1.0, 0.2
            state.pose.pose.orientation.w = 1.0
            state.twist.twist.linear.x, state.twist.twist.linear.y = 1.0, 0.1
            imu = Imu()
            imu.header.stamp, imu.header.frame_id = stamp, "base_link"
            imu.angular_velocity.z = 0.1
            vesc = VescStateStamped()
            vesc.header.stamp, vesc.state.speed = stamp, -20000
            state_pub.publish(state)
            imu_pub.publish(imu)
            vesc_pub.publish(vesc)
            if observations and any(abs(msg.drive.speed-0.5) < 1e-5 for msg in commands):
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("No valid synthetic inference; see " + directory)
        assert len(observations[-1].data) == 84
        assert any(abs(msg.drive.steering_angle-0.104) < 1e-5 for msg in commands)
        stopped_before = len(commands)
        time.sleep(0.25)  # Sensors stop; independent watchdog must publish physical zero.
        assert len(commands) > stopped_before
        assert commands[-1].drive.speed == 0.0 and commands[-1].drive.steering_angle == 0.0
        assert not forbidden
        # Inspect publishers as well as received data: dry-run must not advertise drive.
        publishers = rosgraph.Master("/smoke_check").getSystemState()[0]
        assert not any(topic == prefix+"/forbidden_drive" for topic, nodes in publishers)
        print("PASS: ROS dry run 84D -> actor -> physical command; stale sensors -> zero; no drive publisher")
        print("Synthetic ROS test logs:", directory)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                deadline = time.time()+5
                while process.poll() is None and time.time() < deadline:
                    time.sleep(0.05)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
