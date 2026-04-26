# VR Teleoperation / HugWBC R2 Training

This repository contains an Isaac Gym based training and playback stack for the R2 humanoid controller. It keeps the original HugWBC structure, adds R2-specific tasks, and includes an AMP training path for style rewards from reference motion clips.

## Project Links

- HugWBC website: https://hugwbc.github.io/
- Paper: https://arxiv.org/abs/2502.03206/
- Demo video: https://www.youtube.com/watch?v=JP9A0EIu7nc
- Code structure guide: [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
- AMP motion directory contract: [legged_gym/motions/README.md](legged_gym/motions/README.md)

## Installation

Create a Python environment:

```bash
conda create -n hugwbc python=3.8 -y
conda activate hugwbc
```

Install PyTorch for your CUDA version. For CUDA 11.8:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install Isaac Gym Preview 4 from its extracted external directory, not from this repository:

```bash
cd <isaacgym_preview4_extract_dir>/isaacgym/python
pip install -e .
cd <repo_root>
```

Install this repository's RL package from the repo root:

```bash
pip install -e rsl_rl
```

All commands below should be run from the repository root.

## Training

Train the standard R2 interrupt task:

```bash
python legged_gym/scripts/train.py --task=r2int --headless
```

Train the AMP variant:

```bash
python legged_gym/scripts/train.py --task=r2amp --headless
```

Increase the number of parallel environments when GPU memory allows:

```bash
python legged_gym/scripts/train.py --task=r2amp --headless --num_envs 4096
```

Useful task names:

- `r2int`: standard PPO task using `R2InterruptRobot`, `R2InterruptCfg`, and `R2InterruptCfgPPO`.
- `r2amp`: AMP-enabled PPO task using `R2AmpCfg` and `R2AmpCfgPPO`.

Training logs and checkpoints are written under `logs/<experiment_name>/<MonDD_HH-MM-SS>_<run_name>`.

## Playback

Visualize a trained policy. `play.py` always loads a checkpoint, so train first or pass an existing run:

```bash
python legged_gym/scripts/play.py --task=r2int --load_run <run_dir_name> --checkpoint -1
```

Visualize an AMP checkpoint:

```bash
python legged_gym/scripts/play.py --task=r2amp --load_run <run_dir_name> --checkpoint -3
```

Checkpoint shortcuts:

- `--checkpoint -1`: latest numeric checkpoint, such as `model_2000.pt`.
- `--checkpoint -2`: `model_best_task.pt`, if that file exists.
- `--checkpoint -3`: `model_best_mixed.pt`, if that file exists.

The `r2amp` config saves best task and best mixed checkpoints by default. Other tasks may only have numeric `model_<iteration>.pt` checkpoints unless their config enables best-checkpoint saving.

Playback starts with the gait preset selected by `R2_PLAY_INITIAL_GAIT`. Supported values are `stand`, `jump`, `walk`, and `fast_walk`.

```bash
R2_PLAY_INITIAL_GAIT=walk python legged_gym/scripts/play.py --task=r2amp --checkpoint -3
```

On Windows PowerShell, use:

```powershell
$env:R2_PLAY_INITIAL_GAIT='walk'; python legged_gym/scripts/play.py --task=r2amp --checkpoint -3
```

## AMP Motion Data

AMP reference motions are stored under `legged_gym/motions`. The AMP loader reads `.npz` clips directly from that directory.

Each clip should contain these fields:

- `dof_names`
- `body_names`
- `dof_positions`
- `dof_velocities`
- `body_positions`
- `body_rotations`
- `body_linear_velocities`
- `body_angular_velocities`
- `dt` or `fps`, recommended. If both are omitted, the loader assumes 30 FPS.

When multiple clips are present, all clips must share the same `dof_names`, `body_names`, and `dt`.

Convert source motion files into the AMP directory:

```bash
python legged_gym/scripts/retarget_motion.py --input <source_npz_or_dir> --output legged_gym/motions
```

Skeptic note: check the generated `body_names` before using this output for `r2amp`. The current `retarget_motion.py` body map emits `left_hand_roll_link` and `right_hand_roll_link`, while `r2amp` expects `left_arm_yaw_link` and `right_arm_yaw_link` as AMP key bodies.

Convert LaFAN1-style R2V2 motion files:

```bash
python scripts/convert_lafan1_to_amp.py --input <lafan1_npz_r2v2_dir> --output legged_gym/motions
```

For the full data contract and naming guidance, see [legged_gym/motions/README.md](legged_gym/motions/README.md).

## Repository Map

- `legged_gym/envs/r2`: R2 environment, task configs, interrupt task, and AMP config.
- `legged_gym/scripts`: training, playback, retargeting, and asset utilities.
- `legged_gym/utils`: task registry, terrain utilities, motion loading, and shared helpers.
- `rsl_rl/rsl_rl`: PPO, AMP PPO, actor-critic modules, storage, and runner code.
- `r2_v2_with_shell_no_hand`: R2 robot asset and meshes.
- `scripts`: project-specific conversion utilities.
- `logs`: local training logs and checkpoints.

See [CODE_STRUCTURE.md](CODE_STRUCTURE.md) for a more detailed walkthrough.

## Citation

If you find HugWBC helpful, please cite:

```bibtex
@inproceedings{xue2025hugwbc,
  title={HugWBC: A Unified and General Humanoid Whole-Body Controller for Versatile Locomotion},
  author={Xue, Yufei and Dong, Wentao and Liu, Minghuan and Zhang, Weinan and Pang, Jiangmiao},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

## Acknowledgements

This code builds on:

- [RSL_RL](https://github.com/leggedrobotics/rsl_rl)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym)
- [Walk-These-Ways](https://github.com/Improbable-AI/walk-these-ways)
- [unitree_skd2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
