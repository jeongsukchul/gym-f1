"""Shared rollout-length config key with legacy snapshot compatibility."""


def get_rollout_length(config):
    if "rollout_length" in config:
        value = config["rollout_length"]
        if "n_steps" in config and config["n_steps"] != value:
            raise ValueError("Conflicting rollout_length and legacy n_steps")
    elif "n_steps" in config:
        value = config["n_steps"]
    else:
        raise KeyError("Missing rollout_length (legacy n_steps is also accepted)")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("rollout_length must be a positive integer")
    return value
