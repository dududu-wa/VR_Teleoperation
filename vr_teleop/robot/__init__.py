"""Lazy exports for robot modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "G1Config": ("vr_teleop.robot.g1_config", "G1Config"),
    "ForwardKinematicsResult": ("vr_teleop.robot.g1_kinematics", "ForwardKinematicsResult"),
    "G1Kinematics": ("vr_teleop.robot.g1_kinematics", "G1Kinematics"),
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
