import os
import copy
import re
import json
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def load_cfg_override_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        override = json.load(f)
    if not isinstance(override, dict):
        raise ValueError(f"Config override must be a JSON object: {path}")
    unknown_sections = set(override.keys()) - {"env", "train", "notes"}
    if unknown_sections:
        raise ValueError(
            f"Unknown top-level override section(s): {sorted(unknown_sections)}. "
            "Supported sections are 'env', 'train', and metadata field 'notes'."
        )
    return override

def _recursive_update_cfg_obj(obj, override, prefix):
    if not isinstance(override, dict):
        raise ValueError(f"Override at '{prefix}' must be a JSON object")
    for key, val in override.items():
        if not hasattr(obj, key):
            raise AttributeError(f"Unknown config field '{prefix}.{key}'")
        attr = getattr(obj, key)
        # Nested config groups are BaseConfig-instantiated objects; recurse into them.
        if isinstance(val, dict) and hasattr(attr, "__dict__"):
            _recursive_update_cfg_obj(attr, val, f"{prefix}.{key}")
        else:
            setattr(obj, key, val)

def apply_cfg_override_json(env_cfg, train_cfg, args):
    override_path = getattr(args, "cfg_override_json", None)
    override = load_cfg_override_json(override_path)
    if override is None:
        return env_cfg, train_cfg
    if env_cfg is not None and "env" in override:
        _recursive_update_cfg_obj(env_cfg, override["env"], "env")
    if train_cfg is not None and "train" in override:
        _recursive_update_cfg_obj(train_cfg, override["train"], "train")
    return env_cfg, train_cfg

def validate_amp_cfg_dims(env_cfg, train_cfg=None):
    env_amp = getattr(env_cfg, "amp", None) if env_cfg is not None else None
    train_amp = getattr(train_cfg, "amp", None) if train_cfg is not None else None
    if env_amp is None:
        return
    key_body_names = getattr(env_amp, "key_body_names", None)
    env_amp_obs_dim = getattr(env_amp, "amp_obs_dim", None)
    if key_body_names is None or env_amp_obs_dim is None:
        return
    # AMP obs layout is robot state base (61) plus xyz for each selected key body.
    expected_dim = 61 + 3 * len(key_body_names)
    if env_amp_obs_dim != expected_dim:
        raise ValueError(
            f"env.amp.amp_obs_dim={env_amp_obs_dim} does not match "
            f"61 + 3 * len(env.amp.key_body_names)={expected_dim}"
        )
    if train_amp is not None and hasattr(train_amp, "amp_obs_dim"):
        train_amp_obs_dim = train_amp.amp_obs_dim
        if train_amp_obs_dim != expected_dim:
            raise ValueError(
                f"train.amp.amp_obs_dim={train_amp_obs_dim} does not match "
                f"env AMP dim {expected_dim}"
            )

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1 or load_run == "-1":
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -2:
        model = "model_best_task.pt"
    elif checkpoint == -3:
        model = "model_best_mixed.pt"
    elif checkpoint==-1:
        models = [
            file for file in os.listdir(load_run)
            if re.fullmatch(r"model_\d+\.pt", file)
        ]
        if not models:
            raise ValueError("No numeric model checkpoints in this directory: " + load_run)
        models.sort(key=lambda m: int(re.search(r"(\d+)", m).group(1)))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    if not os.path.exists(load_path):
        raise ValueError("Checkpoint does not exist: " + load_path)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {
            "name": "--checkpoint",
            "type": int,
            "help": (
                "Saved model checkpoint number. "
                "Use -1 for the last numeric checkpoint, -2 for model_best_task.pt, "
                "and -3 for model_best_mixed.pt. Overrides config file if provided."
            ),
        },
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--sim_joystick", "action": "store_true", "default":False, "help": "Sample commands from sim joystick"},
        {"name": "--cfg_override_json", "type": str, "help": "JSON file with top-level env/train config overrides."},
        {"name": "--num_episodes", "type": int, "default": 32, "help": "Evaluation episodes per preset."},
        {"name": "--output_dir", "type": str, "default": "outputs/eval", "help": "Evaluation metrics output directory."},
        {"name": "--episode_seconds", "type": float, "default": 10.0, "help": "Evaluation rollout length in seconds."},
        {"name": "--preset", "action": "append", "help": "Evaluation preset name. Repeat or omit for all presets."},
        {"name": "--compute_dtw", "action": "store_true", "default": False, "help": "Reserve DTW imitation metrics in evaluate.py."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


