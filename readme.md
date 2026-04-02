<div align="center">
  <h1><strong>HugWBC: A Unified and General Humanoid Whole-Body Controller for Versatile Locomotion</strong></h1>
  <p>
    <a href=''>Yufei Xue*</a> &nbsp;&nbsp;
    <a href='https://github.com/WentDong'>Wentao Dong*</a> &nbsp;&nbsp;
    <a href='https://minghuanliu.com'>Minghuan Liu^</a> &nbsp;&nbsp;
    <a href='https://wnzhang.net/'>Weinan Zhang</a> &nbsp;&nbsp;
    <a href='https://oceanpang.github.io/'>Jiangmiao Pang</a> &nbsp;&nbsp;
  </p>
  <p>
  * Equal contribution&nbsp;&nbsp;&nbsp;&nbsp;^ Project Lead
  </p>
  <p>
      <img src="./imgs/sjtu.png" height="100" alt="SJTU Logo"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
      <img src="./imgs/share-logo.png" height="100" alt="Share Logo">
  </p>
  <h3>Robotics: Science and Systems (RSS) 2025</h3>
  <h3>
    <a href="https://hugwbc.github.io/">Website</a> | 
    <a href="https://arxiv.org/abs/2502.03206/">arXiv</a> | 
    <a href="https://www.youtube.com/watch?v=JP9A0EIu7nc">Video</a> 
  </h3>
  <img src="./imgs/framework.png" width="90%" alt="HugWBC Framework">
</div>

## 🔥 News
- \[2025-06] We have open-sourced the training code for HugWBC.
- \[2025-02] The [paper](https://arxiv.org/abs/2502.03206) and [demos](https://hugwbc.github.io) for HugWBC have been released.

## 📚 Installation

First, create and activate a new conda environment:
```bash
conda create -n hugwbc python=3.8 -y
conda activate hugwbc
```

Next, install PyTorch. Please use the command that corresponds to your system's CUDA version. For example, for CUDA 11.8:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then, download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download). After extracting the file, install it by running:
```bash
cd isaacgym/python
pip install -e .
```
**Note:** Please follow the installation instructions from the official NVIDIA website for Isaac Gym, as there may be additional dependencies.

Finally, clone this repository and install the required packages:
```bash
git clone https://github.com/apexrl/HugWBC.git 
cd HugWBC
pip install -e rsl_rl
```

## 🚀 Training & Evaluation

All commands should be run from the root of the `HugWBC` repository.

### Training
To train a new policy, run:
```bash
python legged_gym/scripts/train.py --task=h1int --headless 
```

### Visualization
To visualize a trained policy, run:
```bash
python legged_gym/scripts/play.py --task=h1int
```

### Sim2Sim & Sim2Real Evaluation
We uses the official code base of [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) for Sim2Sim evaluation. And the interface with both mujoco simulation and the real robot is implemented through [unitree_skd2_python](https://github.com/unitreerobotics/unitree_sdk2_python).

## 🔗 Citation

If you find our work helpful, please cite:
```bibtex
@inproceedings{xue2025hugwbc,
  title={HugWBC: A Unified and General Humanoid Whole-Body Controller for Versatile Locomotion}, 
  author={Xue, Yufei and Dong, Wentao and Liu, Minghuan and Zhang, Weinan and Pang, Jiangmiao},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

## 👏 Acknowledgements

Our code is built upon the following open-source projects. We thank the authors for their great work.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym)
- [Walk-These-Ways](https://github.com/Improbable-AI/walk-these-ways)
- [unitree_skd2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

