"""JAX deployment contract, compatible with Python 2.7 / NumPy 1.13.

No ROS, JAX or torch imports. Coordinates and wheel speed must be measured
at the vehicle CoG; ERPM is not a substitute for ground velocity in a drift.
"""
from __future__ import division

import json
import math
import time

import numpy as np


def finite_array(value, shape=None):
    value = np.asarray(value, dtype=np.float32)
    if shape is not None and value.shape != shape:
        raise ValueError("Unexpected shape: %s, expected %s" % (value.shape, shape))
    if not np.all(np.isfinite(value)):
        raise ValueError("Non-finite input")
    return value


def load_config(path):
    with open(path) as stream:
        config = json.load(stream)
    if config.get("contract_version") != 1:
        raise ValueError("Unsupported deployment contract")
    if config.get("action_order") != ["steering", "speed"]:
        raise ValueError("Unsupported action order")
    return config


class ObservationBuilder(object):
    def __init__(self, config, track):
        self.config = config
        self.track = dict((key, finite_array(track[key])) for key in
                          ("xs", "ys", "ss", "yaws", "curvatures", "widths"))
        n = len(self.track["xs"])
        if n < 3 or any(value.shape != (n,) for value in self.track.values()):
            raise ValueError("Track arrays must have equal length >= 3")
        self.length = float(track["length"])
        if not np.isfinite(self.length) or self.length <= self.track["ss"][-1]:
            raise ValueError("Track length must include the closing segment")
        if np.any(np.diff(self.track["ss"]) <= 0) or self.track["ss"][0] != 0:
            raise ValueError("Track arc lengths must increase from zero")
        points = np.column_stack([self.track["xs"], self.track["ys"]])
        closed = np.concatenate([points, points[:1]], axis=0)
        self.start = closed[:-1]
        self.segments = np.diff(closed, axis=0)
        self.segment_l2 = np.sum(self.segments ** 2, axis=1)
        if np.any(self.segment_l2 <= 0):
            raise ValueError("Duplicate track points")
        self.closed_s = np.concatenate([self.track["ss"], [self.length]])
        self.n_points = int(config["lookahead_n_points"])
        self.history_len = int(config["obs_history_len"])
        self.delay = int(config["deployment_obs_delay_steps"])
        self.dim = 7 + self.n_points + (2 if config["sparse_width_obs"] else self.n_points)
        if self.n_points < 1 or self.history_len < 1 or self.delay < 0:
            raise ValueError("Invalid observation geometry")
        if self.dim * self.history_len != int(config["actor_observation_size"]):
            raise ValueError("Actor input does not match history contract")
        self.lows = finite_array(config["norm_lows"], (self.dim,))
        self.highs = finite_array(config["norm_highs"], (self.dim,))
        if np.any(self.highs <= self.lows):
            raise ValueError("Invalid normalization bounds")
        self.history = None

    def reset(self):
        self.history = None

    def project(self, x, y, yaw):
        point = finite_array([x, y])
        t = np.clip(np.sum((point - self.start) * self.segments, axis=1) / self.segment_l2, 0, 1)
        projections = self.start + t[:, None] * self.segments
        distances = np.sum((point - projections) ** 2, axis=1)
        idx = int(np.argmin(distances))
        s = (self.closed_s[idx] + t[idx] * (self.closed_s[idx + 1] - self.closed_s[idx])) % self.length
        track_yaw = self.track["yaws"][idx]
        normal = np.asarray([-np.sin(track_yaw), np.cos(track_yaw)])
        ey = np.sqrt(distances[idx]) * np.sign(np.sum((point - projections[idx]) * normal))
        ephi = math.atan2(math.sin(yaw - track_yaw), math.cos(yaw - track_yaw))
        return s, ey, ephi

    def frame(self, state):
        x, y, yaw, vx, vy, yaw_rate, wheel_omega = finite_array(state, (7,))
        s, ey, ephi = self.project(x, y, yaw)
        # Keep JAX nearest_by_s semantics, including its ss[-1] wrap period.
        queries = (s + np.arange(1, self.n_points + 1) * self.config["lookahead_ds"]) % self.track["ss"][-1]
        indices = np.argmin(np.abs(self.track["ss"][None, :] - queries[:, None]), axis=1)
        widths = self.track["widths"][indices]
        if self.config["sparse_width_obs"]:
            widths = widths[[0, -1]]
        # Slip is undefined at standstill; use zero inside the configured speed floor.
        beta = math.atan2(vy, vx) if math.hypot(vx, vy) > self.config.get("beta_speed_floor", 0.05) else 0.0
        obs = np.concatenate([[vx, vy, ephi, ey, yaw_rate, beta, wheel_omega],
                              self.track["curvatures"][indices], widths]).astype(np.float32)
        if self.config["normalize_obs"]:
            obs = np.clip(2.0 * (obs - self.lows) / (self.highs - self.lows) - 1.0, -1.0, 1.0)
        if self.config["mask_track_obs"]:
            obs[7:] = 0.0
        return obs

    def push(self, state):
        obs = self.frame(state)
        if self.history is None:
            self.history = np.tile(obs[None, :], (self.history_len + self.delay, 1))
        else:
            self.history = np.concatenate([obs[None, :], self.history[:-1]], axis=0)
        return self.history[self.delay:self.delay + self.history_len].reshape(-1).astype(np.float32)


class NumpyPolicy(object):
    def __init__(self, path, expected_size):
        with np.load(path, allow_pickle=False) as weights:
            self.activation = weights["activation"].item()
            count = int(weights["layer_count"])
            self.layers = [(finite_array(weights["kernel_%d" % i]),
                            finite_array(weights["bias_%d" % i])) for i in range(count)]
        if self.activation not in ("tanh", "swish") or not self.layers:
            raise ValueError("Unsupported actor")
        size = int(expected_size)
        for kernel, bias in self.layers:
            if kernel.ndim != 2 or kernel.shape[0] != size or bias.shape != (kernel.shape[1],):
                raise ValueError("Actor layer dimensions do not match")
            size = kernel.shape[1]
        if size != 2:
            raise ValueError("Expected a two-action actor")
        self.obs_dim = int(expected_size)

    def predict(self, obs):
        x = finite_array(obs, (self.obs_dim,))
        for i, (kernel, bias) in enumerate(self.layers):
            x = np.dot(x, kernel) + bias
            if i == len(self.layers) - 1 or self.activation == "tanh":
                x = np.tanh(x)
            else:
                x = x / (1.0 + np.exp(np.clip(-x, -80, 80)))
        return finite_array(x, (2,))


def decode_action(action, config, limits):
    steer, speed = np.clip(finite_array(action, (2,)), -1.0, 1.0)
    p = config["action_params"]
    delta = float(steer * p["s_max"])
    velocity = float(speed * (p["v_max"] - p["v_min"]) * 0.5 + (p["v_max"] + p["v_min"]) * 0.5)
    if not (limits["speed_min"] <= 0 <= limits["speed_max"] and
            limits["steer_min"] <= 0 <= limits["steer_max"]):
        raise ValueError("Physical safety limits must include zero")
    return (float(np.clip(velocity, limits["speed_min"], limits["speed_max"])),
            float(np.clip(delta, limits["steer_min"], limits["steer_max"])))


class CommandGate(object):
    """Use monotonic receipt time; also independently check sensor stamp age."""
    def __init__(self, timeout, clock=None):
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Invalid command timeout")
        self.clock = clock or getattr(time, "monotonic", time.time)
        self.command = (0.0, 0.0)
        self.received = None

    def update(self, command):
        finite_array(command, (2,))
        self.command = tuple(command)
        self.received = self.clock()

    def invalidate(self):
        self.received = None
        self.command = (0.0, 0.0)

    def get(self):
        age = None if self.received is None else self.clock() - self.received
        return self.command if age is not None and 0 <= age <= self.timeout else (0.0, 0.0)


class PoseVelocityEstimator(object):
    """Baseline finite-difference localization velocity at the CoG, for dry runs.

    This is not a high-speed drift estimator. Sparse/noisy AMCL poses need a
    faster fused estimator before driving. Never substitute ERPM for vy.
    """
    def __init__(self, cog_offset_x, filter_time_constant=0.1):
        self.offset = float(cog_offset_x)
        self.tau = float(filter_time_constant)
        self.previous = None
        self.velocity = None

    def update(self, stamp, x, y, yaw):
        finite_array([stamp, x, y, yaw])
        position = np.array([x + self.offset * math.cos(yaw), y + self.offset * math.sin(yaw)])
        if self.previous is None:
            self.previous = (stamp, position)
            return None
        dt = stamp - self.previous[0]
        if dt <= 0:
            raise ValueError("Pose timestamps must increase")
        if dt > 0.5:
            self.previous = (stamp, position)
            self.velocity = None
            return None
        measured = (position - self.previous[1]) / dt
        gain = 1.0 if self.velocity is None else dt / (max(self.tau, 0.0) + dt)
        self.velocity = measured if self.velocity is None else self.velocity + gain * (measured - self.velocity)
        self.previous = (stamp, position)
        vx = self.velocity[0] * math.cos(yaw) + self.velocity[1] * math.sin(yaw)
        vy = -self.velocity[0] * math.sin(yaw) + self.velocity[1] * math.cos(yaw)
        return position[0], position[1], yaw, vx, vy
