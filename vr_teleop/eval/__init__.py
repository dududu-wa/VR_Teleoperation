"""Lazy exports for evaluation modules."""

from importlib import import_module
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Evaluator": ("vr_teleop.eval.evaluator", "Evaluator"),
    "EvalConfig": ("vr_teleop.eval.evaluator", "EvalConfig"),
    "EvalResult": ("vr_teleop.eval.evaluator", "EvalResult"),
    "plot_eval_curves": ("vr_teleop.eval.visualization", "plot_eval_curves"),
    "plot_phase_metrics": ("vr_teleop.eval.visualization", "plot_phase_metrics"),
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
