"""Lazy exports for environment modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "G1BaseEnv": ("vr_teleop.envs.g1_base_env", "G1BaseEnv"),
    "MujocoVecEnv": ("vr_teleop.envs.mujoco_vec_env", "MujocoVecEnv"),
    "IsaacVecEnv": ("vr_teleop.envs.isaac_vec_env", "IsaacVecEnv"),
    "G1MultigaitEnv": ("vr_teleop.envs.g1_multigait_env", "G1MultigaitEnv"),
    "ObservationBuilder": ("vr_teleop.envs.observation", "ObservationBuilder"),
    "ObsConfig": ("vr_teleop.envs.observation", "ObsConfig"),
    "RewardComputer": ("vr_teleop.envs.reward", "RewardComputer"),
    "RewardConfig": ("vr_teleop.envs.reward", "RewardConfig"),
    "TerminationChecker": ("vr_teleop.envs.termination", "TerminationChecker"),
    "TerminationConfig": ("vr_teleop.envs.termination", "TerminationConfig"),
    "DomainRandomizer": ("vr_teleop.envs.domain_rand", "DomainRandomizer"),
    "DomainRandConfig": ("vr_teleop.envs.domain_rand", "DomainRandConfig"),
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
