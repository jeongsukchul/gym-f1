import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


DEFAULT_PERCENTILES = tuple(range(5, 100, 5))
DEFAULT_CVAR_LEVELS = (10, 20)
REWARD_WANDB_KEYS = (
    "n",
    "mean",
    "std",
    "min",
    "max",
    *(f"p{p}" for p in DEFAULT_PERCENTILES),
    *(f"cvar{level}" for level in DEFAULT_CVAR_LEVELS),
)
LENGTH_WANDB_KEYS = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class EvaluationResult:
    episode_rewards: list[float]
    episode_lengths: list[int]
    reward_stats: dict[str, float | int]
    length_stats: dict[str, float | int]


def compute_distribution_stats(
    values: list[float] | list[int] | np.ndarray,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    cvar_levels: tuple[int, ...] = DEFAULT_CVAR_LEVELS,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot compute evaluation statistics from zero episodes")

    stats: dict[str, float | int] = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }

    for percentile in percentiles:
        stats[f"p{percentile}"] = float(np.percentile(array, percentile))

    sorted_values = np.sort(array)
    for level in cvar_levels:
        tail_count = max(1, int(np.ceil(array.size * level / 100.0)))
        stats[f"cvar{level}"] = float(np.mean(sorted_values[:tail_count]))

    return stats


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _format_items(stats: dict[str, float | int], keys: list[str]) -> str:
    return ", ".join(f"{key}={_format_value(stats[key])}" for key in keys if key in stats)


def format_evaluation_result(result: EvaluationResult) -> str:
    reward_stats = result.reward_stats
    length_stats = result.length_stats
    percentile_keys = [f"p{p}" for p in DEFAULT_PERCENTILES]
    midpoint = len(percentile_keys) // 2

    lines = [
        f"Evaluated {int(reward_stats['n'])} episode(s)",
        "Reward summary: " + _format_items(reward_stats, ["mean", "std", "min", "max"]),
        "Reward percentiles: " + _format_items(reward_stats, percentile_keys[:midpoint]),
        "Reward percentiles: " + _format_items(reward_stats, percentile_keys[midpoint:]),
        "Reward lower-tail CVaR: " + _format_items(reward_stats, ["cvar10", "cvar20"]),
        "Episode length: " + _format_items(length_stats, ["mean", "std", "min", "max"]),
    ]
    return "\n".join(lines)


def evaluation_result_to_wandb_metrics(
    result: EvaluationResult,
    n_envs: int,
    prefix: str = "final_eval",
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}/n_envs": int(n_envs),
        f"{prefix}/n_episodes": int(result.reward_stats["n"]),
    }

    for key in REWARD_WANDB_KEYS:
        if key in result.reward_stats:
            metrics[f"{prefix}/reward/{key}"] = result.reward_stats[key]

    for key in LENGTH_WANDB_KEYS:
        if key in result.length_stats:
            metrics[f"{prefix}/episode_length/{key}"] = result.length_stats[key]

    return metrics


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: gym.Env | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    return_stats: bool = False,
) -> tuple[float, float] | tuple[list[float], list[int]] | EvaluationResult:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param return_stats: If True, return an ``EvaluationResult`` with min/max,
        percentiles, and lower-tail CVaR statistics.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    if return_episode_rewards and return_stats:
        raise ValueError("return_episode_rewards and return_stats are mutually exclusive")
    if n_eval_episodes <= 0:
        raise ValueError(f"n_eval_episodes must be positive, got {n_eval_episodes}")

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    reward_stats = compute_distribution_stats(episode_rewards)
    length_stats = compute_distribution_stats(episode_lengths)
    mean_reward = float(reward_stats["mean"])
    std_reward = float(reward_stats["std"])

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_stats:
        return EvaluationResult(
            episode_rewards=[float(reward) for reward in episode_rewards],
            episode_lengths=[int(length) for length in episode_lengths],
            reward_stats=reward_stats,
            length_stats=length_stats,
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
