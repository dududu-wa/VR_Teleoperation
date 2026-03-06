"""
Configuration utilities for VR Teleoperation project.
Registers OmegaConf custom resolvers and provides config loading helpers.
"""

import math
import os
from typing import Dict, Any
import copy
import yaml
try:
    from omegaconf import OmegaConf, DictConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some runtimes
    OmegaConf = None  # type: ignore

    class DictConfig(dict):  # type: ignore
        """Minimal DictConfig compatibility type when omegaconf is unavailable."""
        pass

if OmegaConf is not None:
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
    if OmegaConf is not None and isinstance(obj, DictConfig):
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
    """Return absolute path to robot assets inside this repo or configured path."""
    candidates = []
    for env_var in ("VR_TELEOP_ASSET_ROOT", "UNITREE_MUJOCO_ROOT"):
        env_root = os.getenv(env_var)
        if env_root:
            candidates.append(os.path.abspath(env_root))
    candidates.append(os.path.join(get_project_root(), "assets"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Robot asset path not found. Set VR_TELEOP_ASSET_ROOT (or UNITREE_MUJOCO_ROOT) "
        f"or place assets in {os.path.join(get_project_root(), 'assets')}."
    )


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment yaml and merge with configs/base.yaml when declared."""
    config_path = os.path.abspath(config_path)
    cfg = load_yaml(config_path)

    defaults = cfg.get("defaults", [])
    include_base = False
    if isinstance(defaults, list):
        for item in defaults:
            if item in ("/base", "base", {"": "base"}):
                include_base = True
            if isinstance(item, str) and item.strip() == "/base":
                include_base = True
            if isinstance(item, dict):
                for _, v in item.items():
                    if v == "base":
                        include_base = True

    if include_base:
        base_cfg = load_yaml(os.path.join(get_config_path(), "base.yaml"))
        merged = deep_merge_dict(base_cfg, cfg)
    else:
        merged = cfg

    return merged
