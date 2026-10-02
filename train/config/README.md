# Training configurations

| Backend | Environment | Learning |
| --- | --- | --- |
| JAX PPO / PPO-Lag / GMMVI | `jax/gym_config.yaml` | `jax/rl_config.yaml` |
| Gym / Stable-Baselines3 PPO | `ppo/gym_config.yaml` | `ppo/rl_config.yaml` |

`train/jax_sampler_ppo.py` defaults to the JAX pair; `--gym-config` and
`--rl-config` still accept explicit paths, including old experiment snapshots.
`train/ppo_race.py` and `train/ppo_recover.py` use the PPO pair through
`train/config/env_config.py`. JAX tuning runners also default to the JAX pair.

JAX domain randomization uses `jax_sampler_ppo.domain_randomization_ranges`
and profiles in its RL config. SB3 uses Gaussian `domain_randomization` in
its gym config. Shared environment parameters are independent copies, not
inherited from the other backend. Saved run artifacts retain their original
filenames; existing snapshots are not rewritten.

The SB3 gym config contains only keys consumed by `env_config.py`; JAX-only
history, delay, sensor-noise and reward-shaping keys are kept in the JAX pair.
An explicit `track_pool: null` supplies SB3's required single-track default.

The original mixed `train/config/gym_config.yaml` and `rl_config.yaml` are
retained as legacy copies, but are no longer trainer defaults. Edit the
backend-specific files for future runs. Values were copied from these legacy
defaults, not from an experiment snapshot; runners with explicit frozen
configs continue to use those configs.

JAX's rollout length is 256. `reset_state_on_rollout: false` preserves the
vehicle state, steering buffer, observation history and episode clock, while
resampling DR parameters at each rollout boundary. Collision/time-limit
auto-resets still apply. `completed_episode` carries partial episode costs
across rollouts and updates budget feedback only from completed episodes.
Sampler target scores use the cost of the current rollout; cost beta feedback
uses completed-episode cost, not incomplete episodes labeled as successes.
Dynamics now vary within an episode during training; static-DR evaluation is
unchanged. State-continuation semantics differ from legacy reset-per-rollout
experiments, so they should not be pooled as identical conditions.

Both RL configs use `rollout_length` for the number of policy/environment
steps collected per environment before each PPO update. Legacy snapshots
with `n_steps` are still readable. If both keys are supplied with different
values, loading fails explicitly. JAX's internal `unroll_length` and SB3's
library argument `n_steps` retain their upstream names.

The current JAX default selects a `first_episode` short-window ablation with
`allow_partial_first_episode: true`. It averages each env's cost up to its
first termination within the current rollout, including unfinished envs;
it does not carry their cost into lambda feedback. At rollout 256, this is
not equivalent to measuring a 12288-physics-step episode's collision rate.
Physical state continuation remains enabled, and full-episode evaluation
still uses the original collision budget. `completed_episode` remains
available for completed-episode feedback across rollout boundaries.

`jax_sampler_ppo.domain_randomization: false` fixes training dynamics to the
nominal vehicle parameters and disables sampler updates/plots. The DR vector
dimensions remain unchanged so actor/critic observation shapes stay identical.
Use `sampler: uniform` as the inactive sampler backend. Evaluation is controlled
independently by `eval_domain_randomization`; `--no-domain-randomization` on
the JAX trainer or experiment runner disables both training and evaluation DR.
Observation noise/delay and stochastic training actions are not disabled.
