"""Lazy exports for utility modules.

This avoids importing optional dependencies (for example `omegaconf`)
during package import time.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "quat_rotate_inverse": ("vr_teleop.utils.math_utils", "quat_rotate_inverse"),
    "compute_projected_gravity": ("vr_teleop.utils.math_utils", "compute_projected_gravity"),
    "get_euler_xyz": ("vr_teleop.utils.math_utils", "get_euler_xyz"),
    "mujoco_quat_to_isaac": ("vr_teleop.utils.math_utils", "mujoco_quat_to_isaac"),
    "get_asset_path": ("vr_teleop.utils.config_utils", "get_asset_path"),
    "TrainingLogger": ("vr_teleop.utils.logger", "TrainingLogger"),
    "build_action_symmetry_matrix": ("vr_teleop.utils.symmetry", "build_action_symmetry_matrix"),
    "build_obs_symmetry_matrix": ("vr_teleop.utils.symmetry", "build_obs_symmetry_matrix"),
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
