"""Lazy exports for curriculum modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PhaseCurriculum": ("vr_teleop.curriculum.phase_curriculum", "PhaseCurriculum"),
    "PhaseConfig": ("vr_teleop.curriculum.phase_curriculum", "PhaseConfig"),
    "ADRScheduler": ("vr_teleop.curriculum.adr_scheduler", "ADRScheduler"),
    "ADRConfig": ("vr_teleop.curriculum.adr_scheduler", "ADRConfig"),
    "LPTeacher": ("vr_teleop.curriculum.lp_teacher", "LPTeacher"),
    "LPTeacherConfig": ("vr_teleop.curriculum.lp_teacher", "LPTeacherConfig"),
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
