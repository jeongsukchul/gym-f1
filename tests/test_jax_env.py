import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from gymkhana.envs.dynamic_models import vehicle_dynamics_std_py
from gymkhana.envs.params import load_params
from gymkhana.jax_env import JaxRaceEnv, JaxTrack, load_vehicle_params, vehicle_dynamics_std


def test_jax_std_dynamics_matches_numpy_reference():
    params_dict = load_params("f1tenth_std")
    params = load_vehicle_params("f1tenth_std")
    x = np.array([1.0, 2.0, 0.08, 4.0, 0.3, 0.4, 0.12, 80.0, 90.0], dtype=np.float64)
    u = np.array([0.25, 1.5], dtype=np.float64)

    expected = vehicle_dynamics_std_py(x, u, params_dict)
    actual = np.asarray(vehicle_dynamics_std(jnp.asarray(x), jnp.asarray(u), params))

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_jax_race_env_reset_and_step_shapes():
    env = JaxRaceEnv.from_track_name("Drift", batch_size=32)
    state, obs = env.reset(jax.random.PRNGKey(0))

    assert state.x.shape == (32, 9)
    assert obs.shape == (32, 14)

    action = jnp.zeros((32, 2), dtype=jnp.float32)
    out = env.step(state, action, key=jax.random.PRNGKey(1))

    assert out.state.x.shape == (32, 9)
    assert out.obs.shape == (32, 14)
    assert out.reward.shape == (32,)
    assert out.done.shape == (32,)
    assert jnp.all(jnp.isfinite(out.obs))
    assert jnp.all(jnp.isfinite(out.reward))


def test_jax_slip_reward_peaks_at_target_angle():
    env = JaxRaceEnv.from_track_name(
        "Drift",
        batch_size=1,
        slip_reward_enabled=True,
        slip_reward_target_deg=45.0,
        slip_reward_width_deg=20.0,
        slip_reward_shape=2.5,
    )
    rewards = env._slip_reward_raw_from_abs_deg(jnp.asarray([5.0, 45.0, 85.0], dtype=jnp.float32))

    np.testing.assert_allclose(np.asarray(rewards[1]), 1.0, rtol=1e-6)
    assert rewards[1] > rewards[0]
    assert rewards[1] > rewards[2]


def test_jax_sensor_noise_is_keyed_and_observation_only():
    clean_env = JaxRaceEnv.from_track_name("Drift", batch_size=8, normalize_obs=False)
    noisy_env = JaxRaceEnv.from_track_name(
        "Drift",
        batch_size=8,
        normalize_obs=False,
        sensor_noise_enabled=True,
        sensor_noise_s_std=0.0,
        sensor_noise_n_std=0.2,
        sensor_noise_psi_std=0.2,
    )
    state, _ = clean_env.reset(jax.random.PRNGKey(0))

    clean_obs, clean_frenet = clean_env.observe(state.x)
    noisy_obs_a, noisy_frenet_a = noisy_env.observe(state.x, jax.random.PRNGKey(1))
    noisy_obs_b, _ = noisy_env.observe(state.x, jax.random.PRNGKey(1))
    noisy_obs_c, _ = noisy_env.observe(state.x, jax.random.PRNGKey(2))

    np.testing.assert_allclose(np.asarray(noisy_obs_a), np.asarray(noisy_obs_b), rtol=1e-6, atol=1e-6)
    for clean_value, noisy_value in zip(clean_frenet[:3], noisy_frenet_a[:3]):
        np.testing.assert_allclose(np.asarray(clean_value), np.asarray(noisy_value), rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(clean_obs[:, 2:4]), np.asarray(noisy_obs_a[:, 2:4]))
    assert not np.allclose(np.asarray(noisy_obs_a[:, 2:4]), np.asarray(noisy_obs_c[:, 2:4]))


def test_jax_race_env_step_is_jittable():
    env = JaxRaceEnv(
        track=JaxTrack.from_track_name("Drift"),
        batch_size=16,
        sensor_noise_enabled=True,
        sensor_noise_s_std=0.01,
        sensor_noise_n_std=0.01,
        sensor_noise_psi_std=0.01,
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((16, 2), dtype=jnp.float32)

    step_jit = jax.jit(env.step)
    out = step_jit(state, action, jax.random.PRNGKey(1))

    assert out.obs.shape == (16, 14)
    assert out.reward.shape == (16,)
