from pathlib import Path

import numpy as np
import pytest
import yaml

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("onnxruntime")
pytest.importorskip("optax")
pytest.importorskip("torch")

import jax.numpy as jnp

from gymkhana.inference import OnnxPolicyRunner
from gymkhana.jax_env import JaxRaceEnv
from gymkhana.jax_sampler_ppo import (
    BoundedGMMVISampler,
    F1TenthAdvWrapper,
    SamplerPPOConfig,
    SamplerPPOTrainer,
    UniformDRSampler,
    evaluate_policy,
    export_trainer_policy_to_onnx,
    make_domain_spec,
    params_from_vector,
    record_policy_trajectory,
)
from gymkhana.jax_sampler_ppo.evaluator import generate_adv_unroll
from gymkhana.jax_sampler_ppo.networks import sample_action
from train.jax_sampler_ppo import _evaluation_schedule, _make_eval_fn, _save_eval_render, _track_boundary_lines


def _domain_ranges_from_config():
    with (Path(__file__).resolve().parents[1] / "train/config/rl_config.yaml").open("r") as f:
        config = yaml.safe_load(f)
    return config["jax_sampler_ppo"]["domain_randomization_ranges"]


def _domain_spec_from_config():
    return make_domain_spec(ranges=_domain_ranges_from_config())


def test_domain_spec_uses_configured_multiplicative_bounds():
    ranges = _domain_ranges_from_config()
    spec = _domain_spec_from_config()

    for idx, name in enumerate(spec.names):
        low_factor, high_factor = ranges[name]
        endpoints = np.asarray(
            [
                spec.nominal_vector[idx] * low_factor,
                spec.nominal_vector[idx] * high_factor,
            ]
        )
        np.testing.assert_allclose(spec.low[idx], np.min(endpoints), rtol=1e-6)
        np.testing.assert_allclose(spec.high[idx], np.max(endpoints), rtol=1e-6)


def test_sampler_plot_hist_handles_degenerate_values():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from gymkhana.jax_sampler_ppo.gmmvi.utils import _plot_safe_hist

    fig, ax = plt.subplots()
    try:
        _plot_safe_hist(ax, np.full((4096,), 0.1, dtype=np.float32), low=0.0, high=1.0)
    finally:
        plt.close(fig)


def test_domain_spec_applies_coupled_vehicle_params():
    spec = _domain_spec_from_config()
    values = spec.nominal_vector.at[spec.names.index("lf")].set(spec.nominal_vector[spec.names.index("lf")] * 1.05)
    values = values.at[spec.names.index("s_max")].set(0.4)
    values = values.at[spec.names.index("sv_max")].set(5.0)

    params = params_from_vector(spec, values)

    np.testing.assert_allclose(np.asarray(params.lr), spec.wheelbase - np.asarray(params.lf), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(params.s_min), -0.4, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(params.sv_min), -5.0, rtol=1e-6)


def test_gmmvi_sampler_samples_inside_domain_and_updates():
    spec = _domain_spec_from_config()
    sampler = BoundedGMMVISampler(spec.low, spec.high, num_components=2, num_envs=16, batch_size=16)
    state = sampler.init(jax.random.PRNGKey(0), spec.nominal_vector)

    samples, log_prob, component_ids = sampler.sample(state, jax.random.PRNGKey(1), 16)
    assert samples.shape == (16, spec.size)
    assert component_ids.shape == (16,)
    assert jnp.all(samples >= spec.low)
    assert jnp.all(samples <= spec.high)
    assert jnp.all(jnp.isfinite(log_prob))

    scores = jnp.linspace(-1.0, 1.0, 16)
    update = sampler.update(state, samples, scores, component_ids)

    assert jnp.all(update.state.num_updates == 1)
    assert update.state.sample_db_state.samples.shape[1] == spec.size
    assert jnp.all(jnp.isfinite(update.state.model_state.gmm_state.means))
    active = update.state.model_state.gmm_state.component_mask > 0
    assert jnp.allclose(jnp.sum(jnp.exp(update.state.model_state.gmm_state.log_weights[active])), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        update.metrics["sampler/rollout_score_mean"],
        np.mean(np.asarray(scores)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        update.metrics["sampler/rollout_score_p25"],
        np.percentile(np.asarray(scores), 25),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        update.metrics["sampler/rollout_score_p75"],
        np.percentile(np.asarray(scores), 75),
        rtol=1e-6,
        atol=1e-6,
    )


def test_uniform_dr_sampler_samples_inside_domain_and_updates():
    spec = _domain_spec_from_config()
    sampler = UniformDRSampler(spec.low, spec.high)
    state = sampler.init(jax.random.PRNGKey(0), spec.nominal_vector)

    samples, log_prob, component_ids = sampler.sample(state, jax.random.PRNGKey(1), 16)
    assert samples.shape == (16, spec.size)
    assert component_ids.shape == (16,)
    assert jnp.all(samples >= spec.low)
    assert jnp.all(samples <= spec.high)
    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.allclose(log_prob, log_prob[0])

    scores = jnp.linspace(-1.0, 1.0, 16)
    update = sampler.update(state, samples, scores, component_ids)

    assert update.state.num_updates == 1
    assert update.metrics["sampler/num_components"] == 0
    np.testing.assert_allclose(
        update.metrics["sampler/rollout_score_p25"],
        np.percentile(np.asarray(scores), 25),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        update.metrics["sampler/rollout_score_p75"],
        np.percentile(np.asarray(scores), 75),
        rtol=1e-6,
        atol=1e-6,
    )


def test_num_evals_schedule_matches_total_eval_semantics():
    run_initial, eval_updates = _evaluation_schedule(num_updates=100, num_evals=5)
    assert run_initial
    assert eval_updates == {25, 50, 75, 100}

    run_initial, eval_updates = _evaluation_schedule(num_updates=100, num_evals=1)
    assert not run_initial
    assert eval_updates == {100}

    run_initial, eval_updates = _evaluation_schedule(num_updates=0, num_evals=3)
    assert run_initial
    assert eval_updates == set()


def test_sampler_ppo_training_step_is_jittable():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config())
    config = SamplerPPOConfig(
        sampler="gmmvi",
        unroll_length=2,
        batch_size=4,
        num_epochs=1,
        gmm_components=2,
        policy_hidden_layer_sizes=(16,),
        value_hidden_layer_sizes=(16,),
    )
    trainer = SamplerPPOTrainer(wrapper, config)
    state = trainer.init_state(jax.random.PRNGKey(2))

    state, metrics = jax.jit(trainer.training_step)(state)

    assert state.env_state.obs["actor_obs"].shape == (4, 14)
    assert state.env_state.obs["value_obs"].shape == (4, 14)
    np.testing.assert_allclose(
        np.asarray(state.env_state.obs["value_obs"]),
        np.asarray(state.env_state.obs["actor_obs"]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert metrics["train/env_steps"] == 8
    assert jnp.isfinite(metrics["loss/total"])
    assert jnp.all(state.sampler_state.num_updates == 1)


def test_policy_repeat_steps_hold_action_across_physics_steps():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config(), action_repeat_steps=2)
    env_state = wrapper.reset(jax.random.PRNGKey(13), wrapper.nominal_dynamics_params)

    def zero_policy(obs, key):
        del key
        action = jnp.zeros((obs.shape[0], 2), dtype=jnp.float32)
        return action, {
            "raw_action": action,
            "log_prob": jnp.zeros((obs.shape[0],), dtype=jnp.float32),
        }

    final_state, data = generate_adv_unroll(
        wrapper,
        env_state,
        wrapper.nominal_dynamics_params,
        zero_policy,
        jax.random.PRNGKey(14),
        unroll_length=2,
    )

    assert data.action.shape == (2, 4, 2)
    np.testing.assert_array_equal(np.asarray(final_state.env_state.step_count), np.full((4,), 4))


def test_asymmetric_value_obs_uses_actor_history_plus_domain_params():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config(), obs_history_len=3, asymmetric_critic=True)
    state = wrapper.reset(jax.random.PRNGKey(0), wrapper.nominal_dynamics_params)

    actor_obs = state.obs["actor_obs"]
    value_obs = state.obs["value_obs"]

    assert actor_obs.shape == (4, wrapper.base_observation_size * 3)
    assert value_obs.shape == (4, wrapper.base_observation_size * 3 + wrapper.dynamics_param_size)
    np.testing.assert_allclose(
        np.asarray(value_obs[:, : actor_obs.shape[-1]]),
        np.asarray(actor_obs),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(value_obs[:, actor_obs.shape[-1] :]),
        np.asarray(state.dynamics_params),
        rtol=1e-6,
        atol=1e-6,
    )


def test_obs_delay_reset_samples_configured_uniform_range():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=128, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(
        base_env,
        _domain_spec_from_config(),
        obs_history_len=3,
        obs_delay_min_steps=2,
        obs_delay_max_steps=6,
    )

    state = wrapper.reset(jax.random.PRNGKey(0), wrapper.nominal_dynamics_params)

    delay_steps = np.asarray(state.obs_delay_steps)
    assert delay_steps.shape == (128,)
    assert delay_steps.min() >= 2
    assert delay_steps.max() <= 6
    assert state.obs_history.shape[1] == wrapper.obs_history_len + 6
    assert state.obs["actor_obs"].shape == (128, wrapper.base_observation_size * 3)


def test_obs_delay_selects_delayed_history_window_without_changing_actor_size():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=2, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(
        base_env,
        _domain_spec_from_config(),
        obs_history_len=3,
        obs_delay_min_steps=2,
        obs_delay_max_steps=6,
    )
    obs_history = jnp.reshape(
        jnp.arange(2 * wrapper.obs_history_buffer_len * wrapper.base_observation_size, dtype=jnp.float32),
        (2, wrapper.obs_history_buffer_len, wrapper.base_observation_size),
    )
    delay_steps = jnp.asarray([2, 4], dtype=jnp.int32)

    obs = wrapper._make_obs(obs_history, wrapper.nominal_dynamics_params, delay_steps)

    assert obs["actor_obs"].shape == (2, wrapper.base_observation_size * wrapper.obs_history_len)
    expected = jnp.stack(
        [
            jnp.ravel(obs_history[0, 2:5, :]),
            jnp.ravel(obs_history[1, 4:7, :]),
        ]
    )
    np.testing.assert_allclose(np.asarray(obs["actor_obs"]), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_symmetric_value_obs_matches_actor_obs_when_asymmetric_critic_disabled():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config(), obs_history_len=3, asymmetric_critic=False)
    state = wrapper.reset(jax.random.PRNGKey(0), wrapper.nominal_dynamics_params)

    assert wrapper.value_observation_size == wrapper.actor_observation_size
    np.testing.assert_allclose(
        np.asarray(state.obs["value_obs"]),
        np.asarray(state.obs["actor_obs"]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sampler_ppo_log_std_schedule_keeps_training_state_valid():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config())
    config = SamplerPPOConfig(
        total_timesteps=16,
        unroll_length=2,
        batch_size=4,
        num_epochs=1,
        gmm_components=2,
        policy_hidden_layer_sizes=(16,),
        value_hidden_layer_sizes=(16,),
        init_log_std=-1.0,
        end_log_std=-2.0,
    )
    trainer = SamplerPPOTrainer(wrapper, config)
    state = trainer.init_state(jax.random.PRNGKey(4))
    step = jax.jit(trainer.training_step)

    state, _ = step(state)
    state, metrics = step(state)

    assert jnp.isfinite(metrics["loss/total"])
    log_std = state.params.policy["params"]["log_std"]
    assert jnp.all(log_std <= -1.0)


def test_record_policy_trajectory_returns_arrays_for_rendering():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=1, max_episode_steps=4)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config(), auto_reset=False)
    trainer = SamplerPPOTrainer(
        wrapper,
        SamplerPPOConfig(
            unroll_length=2,
            batch_size=1,
            num_epochs=1,
            policy_hidden_layer_sizes=(16,),
            value_hidden_layer_sizes=(16,),
        ),
    )
    state = trainer.init_state(jax.random.PRNGKey(7))

    trajectory = record_policy_trajectory(
        wrapper,
        trainer.make_policy(state.params, deterministic=True),
        jax.random.PRNGKey(8),
        wrapper.nominal_dynamics_params,
        episode_length=4,
    )

    assert trajectory.states.shape == (4, 1, 9)
    assert trajectory.actions.shape == (4, 1, 2)
    assert trajectory.rewards.shape == (4, 1)
    assert trajectory.done.shape == (4, 1)


def test_eval_render_saves_trajectory_plot(tmp_path):
    pytest.importorskip("matplotlib")

    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=1, max_episode_steps=4)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config(), auto_reset=False)
    config = SamplerPPOConfig(
        unroll_length=2,
        batch_size=1,
        num_epochs=1,
        policy_hidden_layer_sizes=(16,),
        value_hidden_layer_sizes=(16,),
        eval_episode_steps=4,
        eval_render=True,
        eval_render_dir=str(tmp_path),
        eval_video=True,
        eval_video_dir=str(tmp_path),
        eval_video_fps=2,
        eval_video_max_frames=3,
    )
    trainer = SamplerPPOTrainer(wrapper, config)
    state = trainer.init_state(jax.random.PRNGKey(9))

    render_path, video_path = _save_eval_render(
        trainer,
        state,
        wrapper,
        jax.random.PRNGKey(10),
        config,
        eval_index=1,
        env_steps=0,
    )

    assert render_path is not None
    assert Path(render_path).exists()
    assert video_path is not None
    assert Path(video_path).exists()


def test_track_boundary_lines_match_closed_track_shape():
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=1, max_episode_steps=4)

    left_x, left_y, right_x, right_y = _track_boundary_lines(base_env.track)

    expected_points = base_env.track.xs.shape[0] + 1
    assert left_x.shape == (expected_points,)
    assert left_y.shape == (expected_points,)
    assert right_x.shape == (expected_points,)
    assert right_y.shape == (expected_points,)


def test_jax_evaluation_runs_one_episode_per_eval_env():
    eval_envs = 3
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=eval_envs, max_episode_steps=8)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config())
    trainer = SamplerPPOTrainer(
        wrapper,
        SamplerPPOConfig(
            unroll_length=2,
            batch_size=3,
            num_epochs=1,
            policy_hidden_layer_sizes=(16,),
            value_hidden_layer_sizes=(16,),
        ),
    )
    state = trainer.init_state(jax.random.PRNGKey(5))

    result = evaluate_policy(
        wrapper,
        trainer.make_policy(state.params, deterministic=True),
        jax.random.PRNGKey(6),
        wrapper.nominal_dynamics_params,
        episode_length=8,
    )

    assert result.rewards.shape == (eval_envs,)
    assert result.lengths.shape == (eval_envs,)
    assert jnp.isfinite(result.metrics["eval/episode_reward_mean"])
    assert jnp.isfinite(result.metrics["eval/episode_reward_p5"])
    assert jnp.isfinite(result.metrics["eval/episode_reward_p95"])


def test_jax_eval_fn_repeats_episodes_per_selected_dynamics():
    eval_envs = 2
    episodes_per_dynamics = 3
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=eval_envs, max_episode_steps=4)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config())
    trainer = SamplerPPOTrainer(
        wrapper,
        SamplerPPOConfig(
            unroll_length=2,
            batch_size=2,
            num_epochs=1,
            policy_hidden_layer_sizes=(16,),
            value_hidden_layer_sizes=(16,),
        ),
    )
    state = trainer.init_state(jax.random.PRNGKey(11))
    eval_fn = _make_eval_fn(
        wrapper,
        trainer,
        episode_steps=4,
        episodes_per_dynamics=episodes_per_dynamics,
        randomize_dynamics=False,
        jit=False,
    )

    metrics, rewards, lengths, dynamics_params = eval_fn(state.params, jax.random.PRNGKey(12))

    total_episodes = eval_envs * episodes_per_dynamics
    assert rewards.shape == (total_episodes,)
    assert lengths.shape == (total_episodes,)
    assert dynamics_params.shape == (total_episodes, wrapper.dynamics_param_size)
    assert int(metrics["eval/dynamics_count"]) == eval_envs
    assert int(metrics["eval/episodes_per_dynamics"]) == episodes_per_dynamics
    assert int(metrics["eval/total_episodes"]) == total_episodes
    np.testing.assert_allclose(
        np.asarray(dynamics_params),
        np.tile(np.asarray(wrapper.nominal_dynamics_params), (episodes_per_dynamics, 1)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_jax_policy_export_onnx_matches_existing_runner(tmp_path):
    base_env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=16)
    wrapper = F1TenthAdvWrapper(base_env, _domain_spec_from_config())
    config = SamplerPPOConfig(
        unroll_length=2,
        batch_size=4,
        num_epochs=1,
        gmm_components=2,
        policy_hidden_layer_sizes=(16,),
        value_hidden_layer_sizes=(16,),
    )
    trainer = SamplerPPOTrainer(wrapper, config)
    state = trainer.init_state(jax.random.PRNGKey(3))
    onnx_path = tmp_path / "jax_policy.onnx"

    export_trainer_policy_to_onnx(trainer, state, str(onnx_path))

    obs = np.asarray(state.env_state.obs["actor_obs"], dtype=np.float32)
    expected, _ = sample_action(
        trainer.networks.policy_network,
        state.params.policy,
        jnp.asarray(obs),
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    runner = OnnxPolicyRunner(str(onnx_path))

    np.testing.assert_allclose(runner.predict(obs), np.asarray(expected), atol=5e-4)
    assert runner.predict(obs[0]).shape == (2,)
