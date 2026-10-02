"""Environment package exports.

Keep ``GKEnv`` available as ``gymkhana.envs.GKEnv`` without eagerly importing
the full Gymnasium-backed environment when callers only need lightweight
submodules such as ``gymkhana.envs.params`` or ``gymkhana.envs.track``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["GKEnv"]

if TYPE_CHECKING:
    from .gymkhana_env import GKEnv


def __getattr__(name: str) -> Any:
    if name == "GKEnv":
        from .gymkhana_env import GKEnv

        return GKEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
