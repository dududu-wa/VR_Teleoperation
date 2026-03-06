"""Lazy exports for intervention modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "InterventionGenerator": ("vr_teleop.intervention.intervention_generator", "InterventionGenerator"),
    "InterventionConfig": ("vr_teleop.intervention.intervention_generator", "InterventionConfig"),
    "FeasibilityFilter": ("vr_teleop.intervention.feasibility_filter", "FeasibilityFilter"),
    "FeasibilityConfig": ("vr_teleop.intervention.feasibility_filter", "FeasibilityConfig"),
    "DampedLeastSquaresIK": ("vr_teleop.intervention.ik_solver", "DampedLeastSquaresIK"),
    "BatchIKSolver": ("vr_teleop.intervention.ik_solver", "BatchIKSolver"),
    "MotionLibrary": ("vr_teleop.intervention.motion_library", "MotionLibrary"),
    "MotionClip": ("vr_teleop.intervention.motion_library", "MotionClip"),
    "MotionRetargeter": ("vr_teleop.intervention.motion_retarget", "MotionRetargeter"),
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
