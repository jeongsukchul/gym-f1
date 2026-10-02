#!/usr/bin/env python
"""Exercise the deployment contract without ROS master, sensors or actuators."""
from __future__ import division, print_function

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from racecar_policy_core import CommandGate, NumpyPolicy, ObservationBuilder, PoseVelocityEstimator, decode_action


def fixture():
    config = {
        "lookahead_n_points": 5, "lookahead_ds": 0.5, "sparse_width_obs": True,
        "obs_history_len": 6, "deployment_obs_delay_steps": 1, "actor_observation_size": 84,
        "normalize_obs": True, "mask_track_obs": False, "beta_speed_floor": 0.05,
        "action_params": {"s_max": 0.52, "v_min": -5.0, "v_max": 20.0},
        "norm_lows": [-5, -10, -np.pi, -1.1, -5, -np.pi/3, 0] + [-1.95]*5 + [1.2]*2,
        "norm_highs": [20, 10, np.pi, 1.1, 5, np.pi/3, 20/0.0434*6.4] + [1.95]*5 + [2.2]*2,
    }
    track = {"xs": [0, 4, 4, 0], "ys": [0, 0, 4, 4], "ss": [0, 4, 8, 12],
             "yaws": [0, np.pi/2, np.pi, -np.pi/2], "curvatures": [0, 0.1, -0.1, 0.2],
             "widths": [1.4, 1.6, 1.8, 2.0], "length": 16}
    return config, track


def main():
    config, track = fixture()
    builder = ObservationBuilder(config, track)
    state = [1, 0.2, 0, 1, 0.1, 0.2, 30]
    assert builder.project(1, 0.2, 0)[1] > 0
    assert builder.push(state).shape == (84,)
    oldest = builder.frame(state)
    state[3] = 2
    assert np.allclose(builder.push(state).reshape(6, 14)[0], oldest)
    limits = {"speed_min": -5, "speed_max": 20, "steer_min": -0.52, "steer_max": 0.52}
    assert np.allclose(decode_action([0, 0], config, limits), [7.5, 0])
    assert np.allclose(decode_action([0, -0.6], config, limits), [0, 0], atol=1e-6)
    clock = [0.0]
    gate = CommandGate(0.1, lambda: clock[0])
    gate.update((0.5, 0.1))
    assert gate.get() == (0.5, 0.1)
    clock[0] = 0.101
    assert gate.get() == (0.0, 0.0)
    estimator = PoseVelocityEstimator(0)
    assert estimator.update(1.0, 0, 0, 0) is None
    assert np.allclose(estimator.update(1.1, 0.1, 0.02, 0)[3:], [1, 0.2])
    directory = tempfile.mkdtemp(prefix="racecar-policy-check-")
    path = os.path.join(directory, "test-actor.npz")
    try:
        random = np.random.RandomState(0)
        weights = {"layer_count": np.array(4), "activation": np.array("tanh")}
        sizes = [84, 256, 256, 256, 2]
        for i in range(4):
            weights["kernel_%d" % i] = (random.randn(sizes[i], sizes[i+1])*0.01).astype(np.float32)
            weights["bias_%d" % i] = np.zeros(sizes[i+1], dtype=np.float32)
        np.savez(path, **weights)
        policy = NumpyPolicy(path, 84)
        observation = builder.push(state)
        assert policy.predict(observation).shape == (2,)
        start = time.time()
        for _ in range(100):
            policy.predict(observation)
        print("PASS: 84D history, action scaling, timeout, pose velocity, NumPy actor")
        print("Python %s / NumPy %s: synthetic 256x3 actor mean %.3f ms (not trained-policy validation)" %
              (sys.version.split()[0], np.__version__, (time.time()-start)*10.0))
    finally:
        os.remove(path)
        os.rmdir(directory)


if __name__ == "__main__":
    main()
