"""
Configuration utilities for VR Teleoperation project.
Registers OmegaConf custom resolvers and provides config loading helpers.
"""

import math
import os
from omegaconf import OmegaConf, DictConfig

try:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception:
    pass  # Resolvers already registered


def class_to_dict(obj) -> dict:
    """Convert a config object (class or DictConfig) to a plain dict."""
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    if not isinstance(obj, type):
        obj = obj.__class__
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if callable(val):
            continue
        if isinstance(val, type):
            result[key] = class_to_dict(val)
        else:
            result[key] = val
    return result


def get_project_root() -> str:
    """Return absolute path to VR_Teleoperation project root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_config_path() -> str:
    """Return absolute path to configs directory."""
    return os.path.join(get_project_root(), "configs")


def get_asset_path() -> str:
    """Return absolute path to robot asset directory (unitree_mujoco)."""
    project_root = get_project_root()
    codespace = os.path.dirname(project_root)
    return os.path.join(codespace, "unitree_mujoco")
