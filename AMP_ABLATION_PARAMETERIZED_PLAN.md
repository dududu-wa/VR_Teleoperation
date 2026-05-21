# AMP 消融参数化运行计划

## 目标

本计划用于在不新增大量 `task_registry.register(...)` 任务的前提下，运行 `r2amp` 的 AMP 消融实验。所有消融都通过 `--cfg_override_json` 参数覆盖配置完成，保留现有 `r2int` / `r2amp` 两个 task。

## 已实现改动

- `legged_gym/utils/helpers.py`
  - 新增 `--cfg_override_json`。
  - 新增 JSON override 读取和递归覆盖逻辑。
  - 新增 AMP observation 维度校验：`amp_obs_dim == 61 + 3 * len(key_body_names)`。
  - 支持 `--load_run -1` 字符串形式选择最新 run。

- `legged_gym/utils/task_registry.py`
  - `get_cfgs()` 返回 deep copy，避免多次 override 污染注册表默认配置。
  - `make_env()` 和 `make_alg_runner()` 都应用同一个 JSON override。
  - JSON 先覆盖，显式 CLI 参数最后覆盖。

- `configs/ablation/`
  - 新增参数化消融配置：
    - `style0.json`
    - `nonorm.json`
    - `sw0005.json`
    - `sw001.json`
    - `sw002.json`
    - `sw005.json`
    - `obs1.json`
    - `obs4.json`
    - `feetonly.json`
    - `handsonly.json`
    - `handsfeetwaist.json`
    - `disc5.json`
    - `disc10.json`
    - `disc20.json`
    - `dtfalse.json`

- `legged_gym/scripts/evaluate.py`
  - 新增统一评估入口。
  - 复用现有 `task_registry` 和 runner checkpoint 加载逻辑。
  - 支持固定 presets：`stand`、`walk_slow`、`walk_fast`、`turn_left`、`strafe_right`。
  - 输出 `metrics.json` 和 `metrics.csv`。
  - 按 `AMPPPO.process_env_step()` 同公式显式计算 AMP style reward。
  - 使用 `--cfg_override_json` 时要求显式指定 `--load_run`，避免误加载其他消融组 checkpoint。

- `CODE_STRUCTURE.md`
  - 已同步新增参数化消融入口、JSON schema、评估脚本和命令模板。

## JSON Override Schema

```json
{
  "notes": "Optional human-readable note.",
  "env": {
    "amp": {
      "num_amp_obs_steps": 4,
      "key_body_names": [
        "left_arm_yaw_link",
        "right_arm_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link"
      ],
      "amp_obs_dim": 73
    }
  },
  "train": {
    "runner": {
      "run_name": "obs4"
    },
    "amp": {
      "num_amp_obs_steps": 4
    }
  }
}
```

顶层只允许：

- `env`：覆盖环境配置。
- `train`：覆盖训练配置。
- `notes`：人工说明，不参与配置覆盖。

## 训练命令

Task-only PPO 基线：

```powershell
python legged_gym/scripts/train.py --task=r2int --headless --seed=0 --max_iterations=3000
```

AMP 默认方法：

```powershell
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000
```

AMP style reward 移除：

```powershell
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/style0.json
```

AMP 关闭 style reward 归一化：

```powershell
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/nonorm.json
```

AMP style weight sweep 示例：

```powershell
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/sw0005.json
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/sw001.json
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/sw002.json
python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --max_iterations=3000 --cfg_override_json configs/ablation/sw005.json
```

## 评估命令

使用 override 评估时必须显式指定 `--load_run`：

```powershell
python legged_gym/scripts/evaluate.py --task=r2amp --load_run Apr01_00-00-00_style0 --checkpoint=-1 --cfg_override_json configs/ablation/style0.json --num_episodes=32 --output_dir outputs/r2amp_style0_eval
```

只评估单个 preset：

```powershell
python legged_gym/scripts/evaluate.py --task=r2amp --load_run Apr01_00-00-00_style0 --checkpoint=-1 --cfg_override_json configs/ablation/style0.json --preset walk_slow --num_episodes=32 --output_dir outputs/r2amp_style0_walk_slow_eval
```

## 评估输出字段

`evaluate.py` 输出：

- `metrics.json`
- `metrics.csv`

主要字段包括：

- `run_id`
- `task_name`
- `method_name`
- `ablation_name`
- `seed`
- `checkpoint`
- `preset_name`
- `num_episodes`
- `episode_seconds`
- `lin_vel_rmse`
- `yaw_vel_rmse`
- `task_return_mean`
- `fall_rate`
- `episode_length_mean_steps`
- `base_height_violation_rate`
- `roll_pitch_violation_rate`
- `amp_style_reward_mean`
- `amp_style_reward_raw_mean`
- `disc_ref_logit_mean`
- `disc_policy_logit_mean`
- `disc_gap_mean`
- `torque_l2_mean`
- `action_rate_l2_mean`
- `dof_acc_l2_mean`
- `wall_clock_seconds`
- `notes`

## 注意事项

- `handsfeetwaist.json` 要求 AMP motion `.npz` 的 `body_names` 包含 `waist_pitch_link`。如果当前 motion 数据只包含 `base_link`、双臂和双脚，需要先重新导出 motion。
- `--compute_dtw` 当前只保留 DTW 字段和说明。真正的 DTW pose error 需要先定义 preset 与 reference motion 的匹配规则。
- 当前实现不改变 AMP reward 主公式，只增加参数化实验控制和评估导出。

## 已完成验证

静态语法检查：

```powershell
python -m py_compile legged_gym/utils/helpers.py legged_gym/utils/task_registry.py legged_gym/utils/__init__.py legged_gym/scripts/train.py legged_gym/scripts/evaluate.py
```

已完成：

- 15 个 ablation JSON 结构校验。
- key body 配置的 `amp_obs_dim` 校验。
- mock override smoke test。

未完成：

- 当前环境缺少 `isaacgym` / `torch`，未运行真实 Isaac Gym training/evaluation smoke test。
