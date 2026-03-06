"""Lazy exports for deployment modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PolicyWrapper": ("vr_teleop.deploy.policy_wrapper", "PolicyWrapper"),
    "Sim2SimRunner": ("vr_teleop.deploy.sim2sim_runner", "Sim2SimRunner"),
    "Sim2SimConfig": ("vr_teleop.deploy.sim2sim_runner", "Sim2SimConfig"),
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
