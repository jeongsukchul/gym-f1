from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.2.0"

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only in Gymnasium-free installs
    gym = None

if TYPE_CHECKING:
    from .presets import drift_config

if gym is not None:
    gym.register(
        id="gymkhana-v0",
        entry_point="gymkhana.envs:GKEnv",
    )

    __all__ = ["drift_config"]
else:
    __all__ = []


def __getattr__(name: str) -> Any:
    if name == "drift_config" and gym is not None:
        from .presets import drift_config

        return drift_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(["__version__", *__all__])
