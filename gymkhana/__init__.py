__version__ = "1.2.0"

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only in Gymnasium-free installs
    gym = None

if gym is not None:
    from .presets import drift_config

    gym.register(
        id="gymkhana-v0",
        entry_point="gymkhana.envs:GKEnv",
    )

    __all__ = ["drift_config"]
else:
    __all__ = []
