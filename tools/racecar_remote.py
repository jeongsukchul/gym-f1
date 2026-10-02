"""Read-only audit and isolated staging of the legacy ROS1 policy package."""

import argparse
import getpass
import hashlib
import os
from pathlib import Path

import paramiko


HOST = "192.168.45.49"
FINGERPRINT = "2a87d5532960be1be820bb21b1a02f8b"
DESTINATION = "/home/nvidia/racecar_policy_staging"
AUDIT = r"""
 . /opt/ros/melodic/setup.sh
 . /home/nvidia/f1tenth_real/devel/setup.sh
lsusb
ip -br addr
ls -l /dev/sensors /dev/imu /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB0 2>/dev/null
ps -eo pid,args | grep -E 'roscore|rosmaster|roslaunch|vesc|urg_node' | grep -v grep
python --version
python3 --version
python -c 'import numpy, rospy; print("ROS Python / numpy:", numpy.__version__)'
python3 -c 'import numpy; print("Python3 numpy:", numpy.__version__); import rospy; print("Python3 rospy available")'
python3 -c 'import importlib.util; print("onnxruntime:", bool(importlib.util.find_spec("onnxruntime")))'
find /home/nvidia/f1tenth_real -name AGENTS.md -print
git -C /home/nvidia/f1tenth_real/src/f1tenth_system status --short
"""
INTERFACES = r"""
root=/home/nvidia/f1tenth_real/src/f1tenth_system
cat "$root/racecar/racecar/config/racecar-v2/vesc.yaml"
cat "$root/racecar/racecar/config/racecar-v2/sensors.yaml"
cat "$root/racecar/racecar/launch/includes/common/sensors.launch.xml"
grep -nE 'advertise|subscribe|current_speed|current_angular|linear.y|linear.x' "$root/vesc/vesc_ackermann/src/vesc_to_odom.cpp"
grep -nE 'advertise|subscribe|erpm_msg|servo_msg' "$root/vesc/vesc_ackermann/src/ackermann_to_vesc.cpp"
grep -nE 'get_param|rospy.Timer|desired_rpm|publish' "$root/racecar/ackermann_cmd_mux/src/throttle_interpolator.py"
grep -nE 'frame_id|angle_min|angle_max|advertise|scan_topic|scan_time' "$root/urg_node/src/urg_node.cpp" "$root/urg_node/src/urg_c_wrapper.cpp" "$root/urg_node/cfg/URG.cfg" 2>/dev/null | head -45
ps -eo pid,args | grep -E 'racecar_policy|rosmaster.*11312|roscore.*11312' | grep -v grep || true
"""


class KnownBoard(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        if hashlib.md5(key.asbytes()).hexdigest() != FINGERPRINT:
            raise paramiko.SSHException("Board SSH host key changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "interfaces", "stage", "verify", "build", "smoke"))
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(KnownBoard())
    client.connect(HOST, username="nvidia", password=os.environ.get("RACECAR_SSH_PASSWORD") or getpass.getpass(),
                   timeout=10, look_for_keys=False, allow_agent=False)
    try:
        if args.action == "stage":
            source = Path(__file__).resolve().parents[1] / "deploy" / "ros1" / "racecar_policy"
            with client.open_sftp() as sftp:
                try:
                    sftp.stat(DESTINATION)
                except FileNotFoundError:
                    sftp.mkdir(DESTINATION)
                # A package is staged outside the existing workspace. Never replace unrelated files.
                for local in sorted(source.rglob("*")):
                    relative = local.relative_to(source).as_posix()
                    if "__pycache__" in local.parts or local.suffix == ".pyc":
                        continue
                    remote = DESTINATION + "/" + relative
                    if local.is_dir():
                        try:
                            sftp.stat(remote)
                        except FileNotFoundError:
                            sftp.mkdir(remote)
                    else:
                        sftp.put(str(local), remote)
                        if local.parent.name == "scripts":
                            sftp.chmod(remote, 0o755)
            print("Staged:", DESTINATION)
        if args.action == "interfaces":
            command = INTERFACES
        elif args.action == "build":
            # New overlay only. The legacy workspace is a read-only underlay.
            command = r"""set -e
. /opt/ros/melodic/setup.sh
. /home/nvidia/f1tenth_real/devel/setup.sh
mkdir -p /home/nvidia/racecar_policy_ws/src
if [ ! -e /home/nvidia/racecar_policy_ws/src/racecar_policy ]; then
  ln -s /home/nvidia/racecar_policy_staging /home/nvidia/racecar_policy_ws/src/racecar_policy
fi
test "$(readlink -f /home/nvidia/racecar_policy_ws/src/racecar_policy)" = /home/nvidia/racecar_policy_staging
cd /home/nvidia/racecar_policy_ws
catkin_make -j2
. devel/setup.sh
rospack find racecar_policy
python -c 'import runpy; runpy.run_path("/home/nvidia/racecar_policy_staging/scripts/policy_node.py", run_name="import_check"); runpy.run_path("/home/nvidia/racecar_policy_staging/scripts/pose_velocity_node.py", run_name="import_check"); print("ROS node imports passed (no master, no publishers)")'
roslaunch --files racecar_policy policy.launch bundle_dir:=/home/nvidia/policy_bundle
"""
        elif args.action == "smoke":
            command = ("set -e; . /opt/ros/melodic/setup.sh; . /home/nvidia/f1tenth_real/devel/setup.sh; "
                       "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python " + DESTINATION + "/scripts/ros_smoke_check.py")
        else:
            command = AUDIT if args.action == "audit" else (
                "set -e; python3 " + DESTINATION + "/scripts/self_check.py; "
                "python " + DESTINATION + "/scripts/self_check.py"
            )
        _, out, err = client.exec_command(command, timeout=120)
        print(out.read().decode("utf-8", errors="replace"))
        print(err.read().decode("utf-8", errors="replace"))
        if out.channel.recv_exit_status():
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
