"""Independent contract and safety checks for the legacy ROS1 adapter."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deploy/ros1/racecar_policy"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_file("racecar_policy_core", PACKAGE / "src/racecar_policy_core.py")
fixture = load_file("racecar_self_check", PACKAGE / "scripts/self_check.py").fixture


def source_function(path, name, namespace):
    """Run the actual JAX array expressions with NumPy when JAX is unavailable."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    code = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    return namespace[name]


def test_projection_and_lookahead_match_jax_source():
    config, data = fixture()
    builder = core.ObservationBuilder(config, data)
    track = SimpleNamespace(**builder.track, length=builder.length)
    track.closed_xs = np.r_[track.xs, track.xs[:1]]
    track.closed_ys = np.r_[track.ys, track.ys[:1]]
    track.closed_ss = np.r_[track.ss, track.length]
    namespace = {"jnp": np, "wrap_angle": lambda x: np.arctan2(np.sin(x), np.cos(x))}
    project = source_function(ROOT/"gymkhana/jax_env/track.py", "project_to_centerline", namespace)
    source_function(ROOT/"gymkhana/jax_env/track.py", "nearest_by_s", namespace)
    ahead = source_function(ROOT/"gymkhana/jax_env/track.py", "sample_lookahead", namespace)
    random = np.random.RandomState(2)
    for _ in range(50):
        x, y, yaw = random.uniform([-1, -1, -np.pi], [5, 5, np.pi])
        expected = project(track, np.array([x]), np.array([y]), np.array([yaw]))
        actual = builder.project(x, y, yaw)
        np.testing.assert_allclose(actual, [item[0] for item in expected[:3]], atol=2e-6)
        k, w = ahead(track, np.array([actual[0]]))
        frame = builder.frame([x, y, yaw, 2, 0.3, 0.4, 100])
        raw = np.r_[2, 0.3, actual[2], actual[1], 0.4, np.arctan2(0.3, 2), 100, k[0], w[0]]
        norm = source_function(ROOT/"gymkhana/jax_env/env.py", "_normalize_obs", {"jnp": np})
        env = SimpleNamespace(params=SimpleNamespace(v_min=-5, v_max=20, R_w=0.0434),
                              lookahead_n_points=5, sparse_width_obs=True)
        np.testing.assert_allclose(frame, norm(env, raw), atol=2e-6)


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
def test_history_newest_first_and_delayed(delay):
    config, track = fixture()
    config["deployment_obs_delay_steps"] = delay
    builder = core.ObservationBuilder(config, track)
    frames = []
    for speed in range(12):
        state = [1, 0.1, 0, speed, 0, 0, 0]
        frames.insert(0, builder.frame(state))
        observation = builder.push(state)
        padded = frames + [frames[-1]] * 9
        np.testing.assert_allclose(observation, np.asarray(padded[delay:delay+6]).reshape(-1))
    builder.reset()
    np.testing.assert_allclose(builder.push(state).reshape(6, 14), np.tile(builder.frame(state), (6, 1)))


def test_action_scaling_and_physical_clipping():
    config, _ = fixture()
    limits = {"speed_min": -5, "speed_max": 20, "steer_min": -0.52, "steer_max": 0.52}
    np.testing.assert_allclose(core.decode_action([0, 0], config, limits), [7.5, 0])
    np.testing.assert_allclose(core.decode_action([0, -0.6], config, limits), [0, 0], atol=1e-6)
    limits.update(speed_min=0, speed_max=0.5, steer_min=-0.2, steer_max=0.2)
    np.testing.assert_allclose(core.decode_action([1, 1], config, limits), [0.5, 0.2])
    np.testing.assert_allclose(core.decode_action([-1, -1], config, limits), [0, -0.2])
    with pytest.raises(ValueError):
        core.decode_action([np.nan, 0], config, limits)


def test_watchdog_stall_invalidation_and_clock_reversal():
    now = [1.0]
    gate = core.CommandGate(0.1, lambda: now[0])
    assert gate.get() == (0, 0)
    gate.update((0.4, 0.1))
    now[0] += 0.05
    assert gate.get() == (0.4, 0.1)
    now[0] += 0.06
    assert gate.get() == (0, 0)
    gate.update((0.4, 0.1))
    now[0] -= 1
    assert gate.get() == (0, 0)
    gate.update((0.4, 0.1))
    gate.invalidate()
    assert gate.get() == (0, 0)


def test_pose_velocity_body_frame_and_cog_offset():
    estimator = core.PoseVelocityEstimator(0.15, filter_time_constant=0)
    assert estimator.update(1.0, 0, 0, np.pi/2) is None
    result = estimator.update(1.1, 0.02, 0.1, np.pi/2)
    np.testing.assert_allclose(result[:2], [0.02, 0.25], atol=1e-6)
    np.testing.assert_allclose(result[3:], [1, -0.2], atol=1e-6)
    with pytest.raises(ValueError):
        estimator.update(1.1, 0.02, 0.1, 0)
    assert estimator.update(2.0, 1, 1, 0) is None


def test_numpy_actor_flax_orientation_and_tanh_squash(tmp_path):
    random = np.random.RandomState(0)
    kernel = random.randn(84, 16).astype(np.float32)
    output = random.randn(16, 2).astype(np.float32)
    path = tmp_path/"actor.npz"
    np.savez(path, activation=np.array("tanh"), layer_count=np.array(2), kernel_0=kernel,
             bias_0=np.zeros(16), kernel_1=output, bias_1=np.array([0.2, -0.4]))
    actor = core.NumpyPolicy(path, 84)
    obs = random.randn(84).astype(np.float32)
    np.testing.assert_allclose(actor.predict(obs), np.tanh(np.tanh(obs@kernel)@output+[0.2, -0.4]), atol=2e-6)
    with pytest.raises(ValueError):
        actor.predict(np.zeros(14))


def test_nonfinite_state_and_invalid_track_are_rejected():
    config, track = fixture()
    builder = core.ObservationBuilder(config, track)
    with pytest.raises(ValueError):
        builder.push([0, 0, 0, np.inf, 0, 0, 0])
    track["xs"][1] = track["xs"][0]
    track["ys"][1] = track["ys"][0]
    with pytest.raises(ValueError):
        core.ObservationBuilder(config, track)
