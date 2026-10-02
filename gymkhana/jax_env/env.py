"""Batched JAX race environment with no Gymnasium dependency."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dynamics import JaxVehicleParams, load_vehicle_params, rk4_step, speed_steering_angle_action, vehicle_dynamics_std
from .track import JaxTrack, project_to_centerline, sample_lookahead, wrap_angle


class EnvState(NamedTuple):
    x: jnp.ndarray
    last_s: jnp.ndarray
    step_count: jnp.ndarray
    steer_buffer: jnp.ndarray


class StepOutput(NamedTuple):
    state: EnvState
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: dict


class JaxRaceEnv:
    """Vectorized JAX race environment for PPO-style training.

    This first backend mirrors the current drift/race training path:
    STD dynamics, normalized ``[steering_angle, speed]`` actions,
    centerline Frenet boundary checking, progress reward with dense
    near-boundary shaping, and ``drift_real`` vector observations.
    Rendering, LiDAR, recovery mode, and Gym spaces are intentionally absent.
    """

    def __init__(
        self,
        track: JaxTrack,
        params: JaxVehicleParams | None = None,
        batch_size: int = 4096,
        timestep: float = 0.01,
        max_episode_steps: int = 4096,
        progress_gain: float = 5.0,
        out_of_bounds_penalty: float = -50.0,
        negative_vel_penalty: float = -1.0,
        lookahead_n_points: int = 5,
        lookahead_ds: float = 0.5,
        sparse_width_obs: bool = True,
        normalize_obs: bool = True,
        mask_track_obs: bool = False,
        steering_delay_steps: int = 2,
        slip_reward_enabled: bool = False,
        slip_reward_path_weight: float = 0.3,
        slip_reward_slip_weight: float = 0.7,
        slip_reward_target_deg: float = 45.0,
        slip_reward_width_deg: float = 20.0,
        slip_reward_shape: float = 2.5,
        edge_penalty_weight: float = 1.0,
        edge_penalty_start_ratio: float = 0.8,
        termination_boundary_margin_ratio: float = 1.0,
        sensor_noise_enabled: bool = False,
        sensor_noise_s_std: float = 0.0,
        sensor_noise_n_std: float = 0.0,
        sensor_noise_psi_std: float = 0.0,
    ):
        self.track = track
        self.params = params or load_vehicle_params("f1tenth_std")
        self.batch_size = int(batch_size)
        self.timestep = float(timestep)
        self.max_episode_steps = int(max_episode_steps)
        self.progress_gain = float(progress_gain)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.negative_vel_penalty = float(negative_vel_penalty)
        self.lookahead_n_points = int(lookahead_n_points)
        self.lookahead_ds = float(lookahead_ds)
        self.sparse_width_obs = bool(sparse_width_obs)
        self.normalize_obs = bool(normalize_obs)
        self.mask_track_obs = bool(mask_track_obs)
        self.steering_delay_steps = int(steering_delay_steps)
        self.slip_reward_enabled = bool(slip_reward_enabled)
        self.slip_reward_path_weight = float(slip_reward_path_weight)
        self.slip_reward_slip_weight = float(slip_reward_slip_weight)
        self.slip_reward_target_deg = float(slip_reward_target_deg)
        self.slip_reward_width_deg = float(slip_reward_width_deg)
        self.slip_reward_shape = float(slip_reward_shape)
        self.edge_penalty_weight = float(edge_penalty_weight)
        self.edge_penalty_start_ratio = float(edge_penalty_start_ratio)
        self.termination_boundary_margin_ratio = float(termination_boundary_margin_ratio)
        self.sensor_noise_enabled = bool(sensor_noise_enabled)
        self.sensor_noise_s_std = float(sensor_noise_s_std)
        self.sensor_noise_n_std = float(sensor_noise_n_std)
        self.sensor_noise_psi_std = float(sensor_noise_psi_std)
        if self.termination_boundary_margin_ratio < 1.0:
            raise ValueError(
                "termination_boundary_margin_ratio must be >= 1.0 so termination does not happen inside the track."
            )

    @classmethod
    def from_track_name(cls, track_name: str = "Drift", *, reversed: bool = False, **kwargs) -> "JaxRaceEnv":
        return cls(track=JaxTrack.from_track_name(track_name, reversed=reversed), **kwargs)

    def reset(self, key) -> tuple[EnvState, jnp.ndarray]:
        _, key_obs = jax.random.split(key)
        state = self._sample_reset_state(key, self.batch_size)
        obs, _ = self.observe(state.x, key_obs)
        return state, obs

    def step(
        self,
        state: EnvState,
        action: jnp.ndarray,
        key=None,
        auto_reset: bool = True,
        params: JaxVehicleParams | None = None,
    ) -> StepOutput:
        if params is None:
            params = self.params
        raw_steer = action[..., 0]
        delayed_steer = state.steer_buffer[..., -1]
        new_buffer = jnp.concatenate([raw_steer[..., None], state.steer_buffer[..., :-1]], axis=-1)
        delayed_action = action.at[..., 0].set(delayed_steer)

        u = speed_steering_angle_action(delayed_action, state.x, params, self.timestep)
        x_next = rk4_step(vehicle_dynamics_std, state.x, u, self.timestep, params)
        key_obs = key_reset = key_reset_obs = None
        if key is not None:
            key_obs, key_reset, key_reset_obs = jax.random.split(key, 3)
        obs, frenet = self.observe(x_next, key_obs)

        s, ey, _, idx = frenet
        widths = self.track.widths[idx % self.track.widths.shape[0]]
        half_width = 0.5 * widths
        boundary = jnp.abs(ey) > half_width
        edge_proximity, edge_cost = self._edge_cost_from_lateral_error(ey, half_width)
        terminal_boundary = edge_proximity > self.termination_boundary_margin_ratio
        collision_cost = terminal_boundary.astype(jnp.float32)
        progress = self._correct_wraparound_progress(s - state.last_s)
        path_reward = progress * self.progress_gain
        slip_angle_abs_deg = jnp.abs(x_next[..., 6]) * (180.0 / jnp.pi)
        slip_reward_raw = self._slip_reward_raw_from_abs_deg(slip_angle_abs_deg)
        slip_reward = self.slip_reward_slip_weight * slip_reward_raw
        reward = jnp.where(
            self.slip_reward_enabled,
            self.slip_reward_path_weight * path_reward + slip_reward,
            path_reward,
        )
        reward = reward - edge_cost
        # reward = jnp.where(boundary, self.out_of_bounds_penalty, reward)
        # reward = reward + jnp.where(x_next[..., 3] * jnp.cos(x_next[..., 6]) < 0.0, self.negative_vel_penalty, 0.0)

        step_count = state.step_count + 1
        truncated = step_count > self.max_episode_steps
        done = terminal_boundary | truncated
        next_state = EnvState(x=x_next, last_s=s, step_count=step_count, steer_buffer=new_buffer)

        if auto_reset and key is not None:
            reset_state = self._sample_reset_state(key_reset, self.batch_size)
            mask = done[..., None]
            next_state = EnvState(
                x=jnp.where(mask, reset_state.x, next_state.x),
                last_s=jnp.where(done, reset_state.last_s, next_state.last_s),
                step_count=jnp.where(done, reset_state.step_count, next_state.step_count),
                steer_buffer=jnp.where(mask, reset_state.steer_buffer, next_state.steer_buffer),
            )
            reset_obs, _ = self.observe(next_state.x, key_reset_obs)
            obs = jnp.where(mask, reset_obs, obs)

        return StepOutput(
            state=next_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                "s": s,
                "ey": ey,
                "boundary": boundary,
                "terminal_boundary": terminal_boundary,
                "collision_cost": collision_cost,
                "truncated": truncated,
                "path_reward": path_reward,
                "edge_proximity": edge_proximity,
                "edge_cost": edge_cost,
                "slip_angle_abs_deg": slip_angle_abs_deg,
                "slip_reward_raw": slip_reward_raw,
                "slip_reward": slip_reward,
            },
        )

    def observe(
        self,
        x: jnp.ndarray,
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        s, ey, ephi, idx = project_to_centerline(self.track, x[..., 0], x[..., 1], x[..., 4])
        s_obs, ey_obs, ephi_obs = self._apply_sensor_noise(s, ey, ephi, key)
        curvatures, widths = sample_lookahead(
            self.track,
            s_obs,
            n_points=self.lookahead_n_points,
            ds=self.lookahead_ds,
            sparse_widths=self.sparse_width_obs,
        )

        vx = x[..., 3] * jnp.cos(x[..., 6])
        vy = x[..., 3] * jnp.sin(x[..., 6])
        wheel_omega = 0.5 * (x[..., 7] + x[..., 8])
        obs = jnp.concatenate(
            [
                vx[..., None],
                vy[..., None],
                ephi_obs[..., None],
                ey_obs[..., None],
                x[..., 5:6],
                x[..., 6:7],
                wheel_omega[..., None],
                curvatures,
                widths,
            ],
            axis=-1,
        )
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        if self.mask_track_obs:
            obs = obs.at[..., 7:].set(0.0)
        return obs, (s, ey, ephi, idx)

    def _sample_reset_state(self, key, batch_size: int) -> EnvState:
        key_idx, _ = jax.random.split(key)
        idx = jax.random.randint(key_idx, shape=(batch_size,), minval=0, maxval=self.track.xs.shape[0])
        x0 = jnp.zeros((batch_size, 9), dtype=jnp.float32)
        x0 = x0.at[:, 0].set(self.track.xs[idx])
        x0 = x0.at[:, 1].set(self.track.ys[idx])
        x0 = x0.at[:, 4].set(self.track.yaws[idx])
        last_s = self.track.ss[idx]
        return EnvState(
            x=x0,
            last_s=last_s,
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            steer_buffer=jnp.zeros((batch_size, self.steering_delay_steps), dtype=jnp.float32),
        )

    def _correct_wraparound_progress(self, progress):
        half_track = 0.5 * self.track.length
        progress = jnp.where(progress < -half_track, progress + self.track.length, progress)
        progress = jnp.where(progress > half_track, progress - self.track.length, progress)
        max_progress = self.params.v_max * self.timestep * 10.0
        return jnp.clip(progress, -max_progress, max_progress)

    def _slip_reward_raw_from_abs_deg(self, slip_angle_abs_deg):
        normalized_error = jnp.abs((slip_angle_abs_deg - self.slip_reward_target_deg) / self.slip_reward_width_deg)
        return 1.0 / (1.0 + normalized_error ** (2.0 * self.slip_reward_shape))

    def _edge_cost_from_lateral_error(self, ey, half_width):
        edge_proximity = jnp.abs(ey) / (half_width + 1e-6)
        edge_margin = jnp.maximum(edge_proximity - self.edge_penalty_start_ratio, 0.0)
        edge_cost = self.edge_penalty_weight * jnp.square(edge_margin)
        return edge_proximity, edge_cost

    def _apply_sensor_noise(self, s, ey, ephi, key):
        if not self.sensor_noise_enabled or key is None:
            return s, ey, ephi
        noise = jax.random.normal(key, shape=s.shape + (3,), dtype=s.dtype)
        s_obs = jnp.mod(s + noise[..., 0] * self.sensor_noise_s_std, self.track.length)
        ey_obs = ey + noise[..., 1] * self.sensor_noise_n_std
        ephi_obs = wrap_angle(ephi + noise[..., 2] * self.sensor_noise_psi_std)
        return s_obs, ey_obs, ephi_obs

    def _normalize_obs(self, obs):
        p = self.params
        omega_max = p.v_max / p.R_w * 6.4
        lows = jnp.asarray(
            [
                p.v_min,
                -0.5 * p.v_max,
                -jnp.pi,
                -1.1,
                -5.0,
                -jnp.pi / 3.0,
                0.0,
                *([-1.95] * self.lookahead_n_points),
                *([1.2] * (2 if self.sparse_width_obs else self.lookahead_n_points)),
            ],
            dtype=obs.dtype,
        )
        highs = jnp.asarray(
            [
                p.v_max,
                0.5 * p.v_max,
                jnp.pi,
                1.1,
                5.0,
                jnp.pi / 3.0,
                omega_max,
                *([1.95] * self.lookahead_n_points),
                *([2.2] * (2 if self.sparse_width_obs else self.lookahead_n_points)),
            ],
            dtype=obs.dtype,
        )
        return jnp.clip(2.0 * (obs - lows) / (highs - lows) - 1.0, -1.0, 1.0)
