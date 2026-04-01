![Humanoid AMP](docs/human_amp.png)

---

# Humanoid AMP
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.2.0-silver.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Prerequisites

This project requires Isaac Sim and Isaac Lab installed via pip. If you haven't installed them yet, pleanse follow the instructions in [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)



## Setup

No symbolic links or special setup required! This project works as a standalone Isaac Lab extension.

Install the repo in editable mode (mirrors the workflow used in unitree_rl_lab):

```bash
pip install -e .
```

We copy the train and play script from isaaclab, note you do not need to do it yourself.

```bash
bash ./sync_skrl_scripts.sh
```

## Train

```bash
python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Direct-v0 --headless --num_envs 4096
```

or for dance training:

```bash
python -m humanoid_amp.train --task Isaac-G1-AMP-Dance-Direct-v0 --headless --num_envs 4096
```

Additional training options:
```bash
# Resume from checkpoint
python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Direct-v0 --checkpoint logs/skrl/path/to/checkpoint
```

## Eval

```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 32 --checkpoint logs/skrl/<run>/checkpoints/Latest.ckpt
```

## TensorBoard

```bash
python -m tensorboard.main --logdir logs/skrl/
```

Then open your browser to http://localhost:6006

Walk training: `master` branch. Dance training: **`dance`** branch.

The usage of some helper script functions is explained at the beginning of the file.



## Motions Scripts
- `motion_loader.py` - Load motion data from npz files and provide sampling functionality
- `motion_viewer.py` - 3D visualization player for motion data
- `data_convert.py` - Convert CSV motion data to npz format with interpolation and forward kinematics
- `motion_replayer.py` - Replay motion data in Isaac Sim with optional recording
- `record_data.py` - Recording and managing motion data utility classes
- `verify_motion.py` - Verify and display npz file contents
- `visualize_motion.py` - Generate interactive HTML charts to visualize motion data
- `update_pelvis_data.py` - Fix pelvis posture and body center issues in replay motion by replacing pelvis data from source file


## Dataset & URDF

**Note**: The original dataset and URDF files from [Unitree Robotics](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset) have been removed by the official source.

If you're still looking for the dataset, a third-party mirror is currently available here:  
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

Or you can also use the data from [AMASS](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)
## TODO

- ✅ Current: Dancing motion is working
- ✅ Test the `test` branch thoroughly and merge it into `dance` as soon as possible
- ✅ Write a more detailed README to cover the new features and usage
- [ ] Add clearer comments and explanations in all Python scripts


## Resources
[![Demo - Walk](https://img.shields.io/badge/Demo-Walk-ff69b4?style=for-the-badge&logo=bilibili)](https://www.bilibili.com/video/BV19cRvYhEL8/?vd_source=5159ce41348cd4fd3d83ef9169dc8dbc)
[![Demo - Dance (Bilibili)](https://img.shields.io/badge/Demo-Dance-ff69b4?style=for-the-badge&logo=bilibili)](https://www.bilibili.com/video/BV1vW36zEEG8/?share_source=copy_web&vd_source=0de36dd681c4f7ffab776ec97939e21f)
[![Demo - Dance (YouTube)](https://img.shields.io/badge/Demo-Dance-FF0000?style=for-the-badge&logo=youtube)](https://youtu.be/_ItIFkp-Xi4)

[![Documentation](https://img.shields.io/badge/Documentation-DeepWiki-blue?style=for-the-badge&logo=gitbook)](https://deepwiki.com/linden713/humanoid_amp)

**Contributions**, **discussions**, and **stars** are all welcome! ❥(^_-)
