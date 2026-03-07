"""Lazy exports for agent modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ActorCritic": ("vr_teleop.agents.actor_critic", "ActorCritic"),
    "PPO": ("vr_teleop.agents.ppo", "PPO"),
    "RolloutStorage": ("vr_teleop.agents.rollout_storage", "RolloutStorage"),
    "OnPolicyRunner": ("vr_teleop.agents.runner", "OnPolicyRunner"),
    "MlpAdaptModel": ("vr_teleop.agents.networks", "MlpAdaptModel"),
    "UnitreeTeacher": ("vr_teleop.agents.pretrained_adapter", "UnitreeTeacher"),
    "UpperBodyDefaultPolicy": ("vr_teleop.agents.pretrained_adapter", "UpperBodyDefaultPolicy"),
    "DistillationLoss": ("vr_teleop.agents.pretrained_adapter", "DistillationLoss"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
