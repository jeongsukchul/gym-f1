JAX actor deployment on the legacy MIT RACECAR
============================================

This package defaults to observation-only operation. It publishes diagnostics
under ``/racecar_policy`` and creates no publisher on the vehicle drive topic
unless ``dry_run`` is explicitly disabled. It does not start the VESC driver.

Verified board layout (2026-10-02)
--------------------------------

* ``/home/nvidia/catkin_ws`` contains the ROS simulator.
* ``/home/nvidia/f1tenth_real`` contains the existing real-car drivers.
* The device tree reports tegra186 / quill / p3310 (TX2 family).
* ROS Melodic uses Python 2.7, with NumPy 1.13.3; Python 3.6 also has NumPy.
* VESC/IMU serial devices are absent and Ethernet is down. No live sensor
  observations or motor calibration have been verified.

Export from the training PC
--------------------------

Add a fresh output directory to the existing JAX training command::

    python train/jax_sampler_ppo.py ... --deployment-output outputs/car_bundle

The exporter writes ``actor.npz``, ``track.npz``, and ``contract.json`` and checks
the NumPy actor against the deterministic JAX actor before writing them. It
supports tanh/swish MLP actors; actor LayerNorm is rejected. The critic and its
privileged domain parameters are not exported. Existing bundles are not
overwritten. Use the configuration from the actual checkpoint/run, not an
unrelated current default. The repository currently contains no trained bundle.

For an existing full JAX msgpack checkpoint, export without training::

    python -m train.export_racecar_bundle --checkpoint model.msgpack \
      --gym-config saved/gym_config.yaml --rl-config saved/rl_config.yaml \
      --output outputs/car_bundle

At the current defaults the actor consumes 14 floats per frame, six frames,
84 floats total, at 50 Hz. The order is vx, vy, heading error, lateral error,
yaw rate, slip angle, average wheel angular speed, five lookahead curvatures,
and two total widths. All bounds, action scaling, sample spacing, history
length, masks and geometry are included in the contract. SHA256 checks prevent
accidentally mixing the actor and track from different exports.

History is newest to oldest; reset repeats the first valid observation. The
default explicit delay is the lower training delay (currently one 20 ms policy
step). After measuring physical sensor/estimator latency, set
``deployment_obs_delay_steps`` appropriately, including zero when the real
pipeline already supplies the trained latency. Do not also sample artificial
sensor noise in deployment. JAX nearest-by-s lookup uses ``ss[-1]`` as its wrap
period, while pose projection uses track length; this adapter intentionally
matches that existing behavior, including near the seam.

The JAX normalized actor output is [steering, speed]. Under current nominal
parameters, delta = 0.52 * action[0] and speed = 12.5 * action[1] + 7.5. A zero
normalized speed action means 7.5 m/s. Shutdown/watchdog commands use physical
speed zero. The initial vehicle limits allow only forward motion at 0.5 m/s
and +/-0.2 rad; output clipping changes behavior and is for initial bring-up,
not equivalent racing performance. Reverse slip-angle inference is not
validated; use a dedicated slip estimator before supporting reverse motion.

Hardware data required
----------------------

Package-derived command path (the installed launch remappings)::

    normalized [steering, speed]
      -> physical [speed m/s, steering rad]
      -> /vesc/high_level/ackermann_cmd_mux/input/nav_0
      -> high_level mux output -> low_level mux input/navigation
      -> /vesc/low_level/ackermann_cmd_mux/output
      -> ackermann_to_vesc
      -> /vesc/commands/motor/unsmoothed_speed [ERPM]
         /vesc/commands/servo/unsmoothed_position [0..1]
      -> throttle_interpolator
      -> /vesc/commands/motor/speed
         /vesc/commands/servo/position
      -> vesc_driver -> USB serial VESC packets

Installed mappings are ERPM = 4614 * speed and
servo = -1.2135 * steering + 0.5304. For physical 0.5 m/s and 0.1 rad,
these produce ERPM 2307 and servo 0.40905 before smoothing/clipping.
The config clamps ERPM to +/-23250 and servo to [0.15, 0.85]. These are
configuration values, not newly measured calibration.

The installed odometry publishes ``/vesc/odom``. Its velocity proxy is
(-ERPM - offset)/gain, with magnitude below 0.05 m/s replaced by zero;
vy is zero, and yaw rate is proxy_speed * tan(commanded_steering)/0.25.
This is a useful no-slip bring-up approximation, but it cannot supply the
drift actor's lateral velocity or measured yaw rate. LiDAR ``/scan`` goes
through localization/odometry to position and velocity; it is not concatenated
directly into the actor input. Package source defines LaserScan angles from
sensor-reported/limited bounds; config alone cannot establish actual beam count.

For localization-derived velocity, shift position to the CoG, differentiate
world x/y using message timestamps, and rotate:
vx = cos(yaw)*world_vx + sin(yaw)*world_vy;
vy = -sin(yaw)*world_vx + cos(yaw)*world_vy.
Slip beta = atan2(vy, vx) for forward motion. Project the CoG onto the exported
centerline to obtain arc length, left-positive lateral error, and wrapped
(vehicle yaw - track yaw). Use arc length to look up curvature and total width.
Normalize with clip(2*(value-low)/(high-low)-1, -1, 1), then build the delayed
history. The fixed normalization bounds come from the training export, not
from the current physical safety speed limits.

``config/vehicle.yaml`` must be adjusted from measurements:

* ``/racecar/state_estimate``: nav_msgs/Odometry with CoG position in the same
  map frame as exported track, and true CoG vx/vy in the vehicle body frame.
  Align the physical map, track direction, origin and scale first. Do not feed
  the existing VESC odometry's hardcoded vy=0 into a drift actor.
* ``/imu/data``: sensor_msgs/Imu gyro with a valid timestamp/covariance and an
  accurate mounting transform to base_link. Gyro yaw is transformed by tf.
  The installed sensor launch currently has no IMU driver; choose it after
  identifying the actual sensor.
* ``/vesc/sensors/core``: VescStateStamped with no fault. Configure motor pole
  pairs, motor-to-wheel reduction and measured ERPM sign. Formula:
  omega_wheel = sign * ERPM * 2*pi / (60 * pole_pairs * reduction).
  A single motor's shaft speed is only a proxy for the front/rear wheel mean;
  explicitly validate this proxy. Do not estimate wheel speed from ground
  velocity during slip.

``pose_velocity.launch`` supplies a finite-difference localization baseline.
It translates base_link pose to the CoG using a measured offset, rotates world
velocity into the body frame, rejects large covariance, and resets on time
gaps. Sparse AMCL poses and their differentiated velocities are not validated
for drift control; the baseline is intended for dry runs. A high-rate fused
LiDAR/IMU state estimator should replace it before driving. Until that estimator
exists, the package remains waiting for valid observations rather than inventing
vx/vy/beta.

Run on the board
----------------

An isolated overlay preserves the existing workspace::

    source /opt/ros/melodic/setup.bash
    source /home/nvidia/f1tenth_real/devel/setup.bash
    source /home/nvidia/racecar_policy_ws/devel/setup.bash
    roslaunch racecar_policy policy.launch bundle_dir:=/home/nvidia/policy_bundle

Copy the real bundle to ``/home/nvidia/policy_bundle`` first. No mock/synthetic
actor is used for driving. Policy launch does not bring up drivers or sensors.
Monitor ``/racecar_policy/observation`` and
``/racecar_policy/proposed_drive``. With missing sensors or unconfigured motor
parameters, the proposed command remains physical zero.

Live output additionally requires ``calibration_confirmed``,
``state_estimator_confirmed``, and ``wheel_speed_estimate_confirmed`` after
their actual validation. ``dry_run:=false`` then publishes to the existing
high-level mux nav_0 input. Do not bypass the mux or its human override.

Command freshness uses an independent monotonic-clock thread, not the inference
timer. Missing/stale/future timestamps, excessive sensor skew, invalid values,
IMU unavailability, VESC faults, or failed/stalled inference cause zero output.
This only guarantees a zero speed command while the node runs. The installed
throttle interpolator can continue the last command if the entire process/mux
dies, and also smooths zero commands. Configure and verify VESC firmware timeout
and an independent actuator watchdog before motor tests. Its existing throttle
timer uses 1/max_delta_rpm instead of 1/throttle_smoother_rate (153.8 Hz rather
than 75 Hz at installed settings); this was observed but left unchanged.

Verification
------------

``python scripts/self_check.py`` exercises history, physical command scaling,
timeout, velocity estimation and a synthetic NumPy MLP without ROS or hardware.
It creates and removes its own temporary weights. It is not a test of trained
policy driving quality. On the PC run::

    python -m pytest --noconftest tests/test_racecar_deployment.py -q

These tests compare actual JAX source array expressions against the adapter
using NumPy, so they do not require installing JAX on the legacy board. Export
of a real model additionally validates actual JAX vs NumPy actor outputs.

``python scripts/ros_smoke_check.py`` uses a temporary localhost ROS master on
port 11312 and a synthetic actor under a test namespace. It verifies 84D
observations, action scaling, sensor timeout stopping, and absence of any drive
publisher in dry run, then stops its own processes. It never starts a driver.
