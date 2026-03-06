"""
Configuration utilities for VR Teleoperation project.
Registers OmegaConf custom resolvers and provides config loading helpers.
"""

import math
import os
from typing import Optional, Dict, Any
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


def get_workspace_root() -> str:
    """Return parent directory that contains sibling repos."""
    return os.path.dirname(get_project_root())


def get_config_path() -> str:
    """Return absolute path to configs directory."""
    return os.path.join(get_project_root(), "configs")


def get_asset_path() -> str:
    """Return absolute path to robot asset directory (unitree_mujoco)."""
    env_root = os.getenv("UNITREE_MUJOCO_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    return os.path.join(get_workspace_root(), "unitree_mujoco")


def _resolve_repo_root(env_var: str, default_repo_name: str) -> Optional[str]:
    """Resolve an external reference repo root path.

    Priority:
    1) explicit environment variable
    2) sibling folder under workspace root
    """
    env_val = os.getenv(env_var)
    if env_val:
        return os.path.abspath(env_val)

    candidate = os.path.join(get_workspace_root(), default_repo_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return None


def get_falcon_root() -> Optional[str]:
    """Return FALCON repo root if available."""
    return _resolve_repo_root("FALCON_ROOT", "FALCON")


def get_hugwbc_root() -> Optional[str]:
    """Return HugWBC repo root if available."""
    return _resolve_repo_root("HUGWBC_ROOT", "HugWBC")


def get_unitree_rl_gym_root() -> Optional[str]:
    """Return unitree_rl_gym repo root if available."""
    return _resolve_repo_root("UNITREE_RL_GYM_ROOT", "unitree_rl_gym")


def get_falcon_g1_yaml_path() -> Optional[str]:
    """Return FALCON G1 29-DOF YAML path if present."""
    falcon_root = get_falcon_root()
    if falcon_root is None:
        return None
    path = os.path.join(
        falcon_root,
        "humanoidverse",
        "config",
        "robot",
        "g1",
        "g1_29dof_waist_fakehand.yaml",
    )
    return path if os.path.exists(path) else None


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
