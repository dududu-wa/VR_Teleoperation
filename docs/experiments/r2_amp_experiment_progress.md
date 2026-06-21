# R2 AMP Experiment Progress

Last updated: 2026-06-21

This document is the running record for R2 AMP ablations. Keep it factual: record what was run, what changed, where the artifacts are, what the evaluation showed, and what conclusion is supported by the data.

## Current Question

The current multi-expert AMP policy can show useful early checkpoints but then regress late in training. The working hypothesis is that late collapse is driven more by command/curriculum/disturbance scheduling than by the discriminator architecture alone.

The first July19 batch tests four levers:

- `scratch_command_hold`: keep multi-expert AMP, but disable command curriculum and hold the command range fixed.
- `scratch_no_push`: keep curriculum and AMP, but remove randomized push impulses.
- `scratch_amp_slow_lowcap`: keep multi-expert routing, but make AMP style reward weaker and slower.
- `scratch_slow_penalty_ramp`: keep multi-expert AMP unchanged, but slow the shared penalty/push curriculum scale ramp by raising `penalize_curriculum_sigma` from `0.8` to `0.95`.

## Imported Historical Records

These records were imported from earlier session notes and filtered to keep only durable facts, verified paths, and experiment implications.

### r2int_v7 Interrupt Arm Asymmetry

Original question:

```text
D:\codebase\vr_project\VR_Teleoperation\logs\r2_interrupt\r2int_v7
```

The symptom was visually asymmetric hand/arm motion in `r2int_v7`. The useful conclusion is that this was primarily an interrupt configuration bug, not just a weak symmetry regularizer.

Verified runtime DOF order around the upper body:

| action index | DOF |
|---:|---|
| 14 | `head_yaw_joint` |
| 15 | `head_pitch_joint` |
| 16 | `left_shoulder_pitch_joint` |
| 17 | `left_shoulder_roll_joint` |
| 18 | `left_shoulder_yaw_joint` |
| 19 | `left_arm_pitch_joint` |
| 20 | `left_arm_yaw_joint` |
| 21 | `right_shoulder_pitch_joint` |
| 22 | `right_shoulder_roll_joint` |
| 23 | `right_shoulder_yaw_joint` |
| 24 | `right_arm_pitch_joint` |
| 25 | `right_arm_yaw_joint` |

The old interrupt config used:

```python
disturb_action_indices = [14, 15, 16, 17, 19, 20, 21, 22]
```

That meant the interrupt target hit `head_yaw/head_pitch`, skipped `left_shoulder_yaw`, skipped `right_shoulder_yaw`, and skipped `right_arm_pitch/right_arm_yaw`. So the effective perturbation was "head + partial left arm + partial right shoulder", not symmetric left/right arms.

Current E-checkout code has the corrected contract:

```python
DISTURB_DIM = 10
disturb_action_indices = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
```

Current file:

```text
legged_gym/envs/r2/r2interrupt_config.py
```

Important implication:

- Basic `evaluate.py` locomotion eval can load `r2int_v7` and show no falls, but it disables disturb/interruption during evaluation, so it cannot validate the arm asymmetry issue.
- The dedicated interrupt-asymmetry eval is the relevant diagnostic.
- Existing pre-fix output:
  - `outputs/eval_r2int_v7_interrupt_asymmetry/interrupt_asymmetry_metrics.csv`
- Existing post-fix smoke output:
  - `outputs/eval_r2int_v7_interrupt_asymmetry_after_fix/interrupt_asymmetry_metrics.csv`

Selected pre-fix forced-interrupt metrics:

| preset | interrupt fraction | endpoint mirror error | upper DOF mirror error | fall rate/env step |
|---|---:|---:|---:|---:|
| `stand` | 0.998 | 0.381 m | 1.350 | 0.001 |
| `walk_slow` | 1.000 | 0.368 m | 1.301 | 0.000 |
| `walk_fast` | 0.997 | 0.392 m | 1.511 | 0.003 |

Selected post-fix smoke metric:

| preset | mode | interrupt fraction | endpoint mirror error | upper DOF mirror error |
|---|---|---:|---:|---:|
| `stand` | `forced_interrupt` | 1.000 | 0.168 m | 0.508 |

Do not over-interpret the post-fix smoke: it was a short 2-second stand check, useful for verifying the corrected action indices, not a replacement for retraining.

### Discriminator Accuracy Definition

The AMP discriminator in this repo is trained as a least-squares GAN style discriminator, not as a sigmoid/BCE classifier. The durable definition for a derived "classification accuracy" is therefore sign-based balanced accuracy:

```text
agent_acc = mean(agent_logit < 0)
ref_acc   = mean(ref_logit > 0)
disc_acc  = 0.5 * (agent_acc + ref_acc)
```

Current training/eval code logs logits and gaps, not this accuracy directly:

- `disc_agent_logit`
- `disc_ref_logit`
- `disc_ref_logit_mean`
- `disc_policy_logit_mean`
- `disc_gap_mean`

Use discriminator accuracy only as an auxiliary diagnostic. A high discriminator separation does not guarantee task performance if the style reward is overweighted.

### July15 sw1 AMP Weight Failure

Historical run:

```text
logs/r2_amp/July15/sw1
```

Config:

```text
configs/ablation/sw1.json
```

Key setting:

```json
"style_reward_weight": 1.0
```

Filtered conclusion:

- `sw1` performed poorly because the AMP style reward dominated the task reward.
- Earlier analysis of the tail of training found `Loss/style_to_task_abs_ratio` around `6.71`, meaning style contribution was about 6.7 times the task contribution by absolute magnitude.
- The discriminator direction itself was not the main failure: historical tail logits were approximately `disc_agent_logit = -0.567` and `disc_ref_logit = +0.568`, so it still separated policy from reference in the expected direction.
- `Episode/disturb_curriculum` was `0`, so this run should not be primarily attributed to the r2int interrupt asymmetry bug.

Practical implication:

- Do not use `style_reward_weight=1.0` as a serious AMP baseline for this project.
- Prefer dt-scaled and capped style reward, lower style weights, warmup, and task-ratio caps.
- This failure motivated later experiments that keep AMP as an auxiliary prior rather than the dominant objective.

## Evaluation Protocol

Evaluation uses `legged_gym/scripts/evaluate.py` through WSL because the Windows runtime cannot directly run Isaac Gym here.

Common settings:

- WSL path: `/mnt/e/codebase/VR_Teleoperation`
- Task: `r2amp`
- Device: `--sim_device=cpu --rl_device=cpu`
- Parallel eval envs: `--num_envs=64`
- Episodes: `--num_episodes=64`
- Episode length: `--episode_seconds=10`
- Presets: default 7 presets from `evaluate.py`
  - `stand`
  - `walk_slow`
  - `walk_fast`
  - `run`
  - `jump`
  - `turn_left`
  - `strafe_right`
- Checkpoints evaluated for each trained run:
  - `--checkpoint=-2`: `model_best_task.pt`
  - `--checkpoint=8000`: final checkpoint

As of 2026-06-21, `evaluate.py` keeps DTW imitation metrics opt-in. The default fixed-preset protocol below does not pass `--compute_dtw`, so `joint_pose_error_dtw_m` and `key_body_error_dtw_m` remain empty. This preserves the task/fall/RMSE/smoothness/discriminator metrics while avoiding per-episode best-clip DTW cost during 64-episode batch evaluation.

The WSL command template and runtime caveat are documented in `CODE_STRUCTURE.md` under `legged_gym/scripts/evaluate.py`. GPU eval is blocked on this machine because the current `r2gym` PyTorch build does not support RTX 5080 Laptop `sm_120`.

## July19 Batch

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\July19
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\July19_amp_slow_lowcap_best
E:\codebase\VR_Teleoperation\outputs\eval\July19_amp_slow_lowcap_8000
E:\codebase\VR_Teleoperation\outputs\eval\July19_command_hold_best
E:\codebase\VR_Teleoperation\outputs\eval\July19_command_hold_8000
E:\codebase\VR_Teleoperation\outputs\eval\July19_no_push_best
E:\codebase\VR_Teleoperation\outputs\eval\July19_no_push_8000
E:\codebase\VR_Teleoperation\outputs\eval\July19_slow_penalty_ramp_best
E:\codebase\VR_Teleoperation\outputs\eval\July19_slow_penalty_ramp_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per preset.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | status |
|---|---|---|---|---|
| `scratch_amp_slow_lowcap` | `configs/ablation/scratch_amp_slow_lowcap.json` | `logs/r2_amp/July19/Jun19_16-08-42_scratch_amp_slow_lowcap` | `1214`, `1227`, `1806` | evaluated |
| `scratch_command_hold` | `configs/ablation/scratch_command_hold.json` | `logs/r2_amp/July19/Jun19_16-09-11_scratch_command_hold` | `7120`, `7219`, `7966` | evaluated |
| `scratch_no_push` | `configs/ablation/scratch_no_push.json` | `logs/r2_amp/July19/Jun19_16-12-37_scratch_no_push` | `1676`, `1685`, `1881` | evaluated |
| `scratch_slow_penalty_ramp` | `configs/ablation/scratch_slow_penalty_ramp.json` | `logs/r2_amp/July19/Jun20_04-58-31_scratch_slow_penalty_ramp` | `1163`, `1219`, `1222` | evaluated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | lin rmse | yaw rmse | height viol | roll/pitch viol | style reward | policy logit | disc gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scratch_amp_slow_lowcap` | `best` | -22.25 | 0.199 | 435.2 | 0.407 | 0.518 | 0.0035 | 0.0004 | 0.0037 | -0.806 | 1.675 |
| `scratch_amp_slow_lowcap` | `8000` | -142.87 | 0.125 | 447.9 | 0.385 | 0.654 | 0.0018 | 0.0006 | 0.0038 | -0.797 | 1.687 |
| `scratch_command_hold` | `best` | -21.57 | 0.183 | 436.5 | 0.456 | 0.638 | 0.0026 | 0.0002 | 0.0045 | -0.729 | 1.619 |
| `scratch_command_hold` | `8000` | -24.24 | 0.134 | 456.3 | 0.409 | 0.558 | 0.0038 | 0.0003 | 0.0047 | -0.722 | 1.607 |
| `scratch_no_push` | `best` | -11.48 | 0.181 | 428.5 | 0.419 | 0.557 | 0.0010 | 0.0003 | 0.0039 | -0.797 | 1.670 |
| `scratch_no_push` | `8000` | -64.47 | 0.161 | 428.9 | 0.385 | 0.473 | 0.0005 | 0.0004 | 0.0066 | -0.611 | 1.513 |
| `scratch_slow_penalty_ramp` | `best` | -14.85 | 0.185 | 438.3 | 0.413 | 0.568 | 0.0048 | 0.0004 | 0.0043 | -0.777 | 1.638 |
| `scratch_slow_penalty_ramp` | `8000` | -21.70 | 0.248 | 386.0 | 0.433 | 0.499 | 0.0002 | 0.0002 | 0.0072 | -0.529 | 1.420 |

### Supported Conclusion

The strongest standalone continuation target remains `scratch_command_hold`.

Reason:

- `scratch_no_push` has the best early checkpoint, but its final checkpoint regresses from `-11.48` to `-64.47`. Removing push improves early learning but does not fix late training drift.
- `scratch_amp_slow_lowcap` regresses from `-22.25` to `-142.87`. Weakening/slowing AMP style reward alone does not solve the problem and is not worth continuing as-is.
- `scratch_slow_penalty_ramp` improves the early checkpoint relative to `scratch_command_hold` (`-14.85` vs `-21.57`) and keeps final task return close to command-hold final (`-21.70` vs `-24.24`), but the final fall rate is much worse (`0.248` vs `0.134`). Its final `run` preset has `fall_rate = 1.000`, so it is not a cleaner standalone fix.
- `scratch_command_hold` stays close between best and final: `-21.57` to `-24.24`, with lower final fall rate than its best checkpoint. This supports the hypothesis that command/curriculum scheduling is a main driver of late instability.

In plain terms: push and penalty ramp both affect early quality, but the more important issue is still the late training schedule. Holding command curriculum stabilizes training better than only weakening AMP, removing push, or slowing the penalty ramp.

### Pure PPO `r2int_v7` Comparison

User concern:

```text
E:\codebase\VR_Teleoperation\logs\r2_interrupt
```

The relevant pure PPO run is:

```text
E:\codebase\VR_Teleoperation\logs\r2_interrupt\r2int_v7
```

It contains only `model_30000.pt`. A same-protocol WSL CPU eval was added for comparison:

```text
E:\codebase\VR_Teleoperation\outputs\eval_r2int_v7_30000_ep64_cpu
```

This eval uses the same fixed 7 presets and `--num_episodes=64` protocol as the July19 AMP evals. Existing older PPO evals such as `outputs/eval_r2int_v7_30000_ep4_cpu` used only 3-4 episodes per preset, so they are useful as smoke checks but not the primary comparison.

Aggregate result:

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | lin rmse | yaw rmse | height viol | roll/pitch viol | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r2int_v7` pure PPO | `30000` | -66.85 | 0.185 | 427.8 | 0.421 | 0.663 | 0.0015 | 0.0012 | 37838 | 33.042 | 380717 |
| `scratch_command_hold` AMP | `8000` | -24.24 | 0.134 | 456.3 | 0.409 | 0.558 | 0.0038 | 0.0003 | 32832 | 4.901 | 315275 |
| `scratch_no_push` AMP | `best` | -11.48 | 0.181 | 428.5 | 0.419 | 0.557 | 0.0010 | 0.0003 | 16269 | 2.393 | 302449 |
| `scratch_slow_penalty_ramp` AMP | `best` | -14.85 | 0.185 | 438.3 | 0.413 | 0.568 | 0.0048 | 0.0004 | 24321 | 2.924 | 320162 |

Interpretation:

- On fixed-preset quantitative eval, pure PPO is not better than the best July19 AMP checkpoints. It has worse avg task return than `scratch_command_hold_8000`, `scratch_no_push_best`, and `scratch_slow_penalty_ramp_best`.
- Pure PPO also has much higher action-rate L2 (`33.042`) than `scratch_command_hold_8000` (`4.901`), so if it visually looks better, that impression is not captured by this smoothness metric.
- Pure PPO may still be a useful visual or robustness reference because it is a long 30000-iteration run and does not carry AMP motion-prior artifacts. But under the current `evaluate.py` preset protocol, it should not be treated as a stronger quantitative baseline.
- The fair next baseline is not old `r2int_v7` alone; it is a fresh PPO/control run under the same current code, same motion-independent evaluation protocol, same command schedule decision, and comparable training budget.

## Current Decision

Do not continue:

- `scratch_amp_slow_lowcap`

Keep as diagnostic evidence, not as the final mechanism:

- `scratch_no_push`
- `scratch_slow_penalty_ramp`

Continue from:

- `scratch_command_hold`

Recommended next batch:

1. `command_hold + controlled_disturb_release`: checks whether `scratch_command_hold` was stable because command ranges were fixed or because interrupt/disturb never released.
2. `command_hold + no_push`: checks whether removing push on top of fixed commands improves early quality without the final collapse.
3. `command_hold + conservative penalty ramp`: tests whether the early benefit of `scratch_slow_penalty_ramp` can be kept while avoiding the final `run` preset collapse.
4. `command_hold + style low cap`: tests whether command-hold late stability improves when AMP style remains a weaker auxiliary prior.
5. `command_hold + staged command release`: start fixed, then gradually release walk/run before jump. This still needs a separate command-schedule implementation and is not represented by the JSONs below.

### Next-Batch Configs Added 2026-06-20

Code change:

```text
legged_gym/envs/r2/r2interrupt_config.py
legged_gym/envs/r2/r2interrupt.py
```

`cfg.disturb.start_by_curriculum` is now an active config switch. The default remains `true`, preserving the previous curriculum-style release. Setting it to `false` bypasses the `terrain_curriculum_mode` gate for noise-disturb environments while keeping the noise-disturb partition itself. This is meant to isolate the confound found in `scratch_command_hold`: fixed command curriculum also kept `disturb_curriculum=0.0000` in the tail training log.

Planned configs:

| experiment | config | hypothesis | status |
|---|---|---|---|
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | Fixed command ranges remain, but disturb release is decoupled from terrain/command curriculum via `env.disturb.start_by_curriculum=false`; if this collapses, the main stabilizer in `scratch_command_hold` was likely suppressed disturb rather than command range alone. | evaluated in July20 |
| `command_hold_no_push` | `configs/ablation/command_hold_no_push.json` | Fixed command ranges plus zero randomized base push tests whether push removal improves early quality without reintroducing late drift. | evaluated in July20 |
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | Fixed command ranges plus `penalize_curriculum_sigma=0.9` tests a middle ramp between default `0.8` and the too-slow `0.95` batch. | evaluated in July20 |
| `command_hold_style_lowcap` | `configs/ablation/command_hold_style_lowcap.json` | Fixed command ranges plus longer AMP warmup and lower task-ratio cap tests whether style pressure should stay weaker after task behavior forms. | evaluated in July20 |

Verification already completed for config plumbing only:

```text
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_amp_training_contracts.py
```

Training and WSL CPU evaluation are recorded in the July20 batch below.

## July20 Batch

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\July20
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_conservative_penalty_ramp_best
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_conservative_penalty_ramp_8000
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_controlled_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_controlled_disturb_release_8000
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_no_push_best
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_no_push_8000
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_style_lowcap_best
E:\codebase\VR_Teleoperation\outputs\eval\July20_command_hold_style_lowcap_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed in this batch because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | status |
|---|---|---|---|---:|---:|---|
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | `logs/r2_amp/July20/Jun20_15-18-58_command_hold_conservative_penalty_ramp` | `5818`, `7663`, `7930` | `23.49` | `0.0000` | evaluated |
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | `logs/r2_amp/July20/Jun20_15-19-48_command_hold_controlled_disturb_release` | `1166`, `1706`, `1944` | `-4.26` | `0.9956` | evaluated |
| `command_hold_no_push` | `configs/ablation/command_hold_no_push.json` | `logs/r2_amp/July20/Jun20_15-21-52_command_hold_no_push` | `6059`, `6973`, `7440` | `31.15` | `0.0000` | evaluated |
| `command_hold_style_lowcap` | `configs/ablation/command_hold_style_lowcap.json` | `logs/r2_amp/July20/Jun20_15-22-56_command_hold_style_lowcap` | `7439`, `7600`, `7937` | `28.81` | `0.0000` | evaluated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | lin rmse | yaw rmse | height viol | roll/pitch viol | style reward | policy logit | disc gap | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_conservative_penalty_ramp` | `best` | -79.37 | 0.087 | 472.3 | 0.317 | 0.426 | 0.0005 | 0.0001 | 0.0054 | -0.704 | 1.597 | 22363 | 2.676 | 267651 |
| `command_hold_conservative_penalty_ramp` | `8000` | -80.78 | 0.080 | 477.4 | 0.298 | 0.438 | 0.0015 | 0.0002 | 0.0054 | -0.701 | 1.590 | 22357 | 2.691 | 269447 |
| `command_hold_controlled_disturb_release` | `best` | -9.44 | 0.435 | 318.6 | 0.536 | 0.651 | 0.0021 | 0.0009 | 0.0042 | -0.782 | 1.638 | 16030 | 2.743 | 318689 |
| `command_hold_controlled_disturb_release` | `8000` | -67.70 | 0.049 | 485.1 | 0.318 | 0.466 | 0.0015 | 0.0001 | 0.0059 | -0.670 | 1.547 | 39303 | 5.165 | 258774 |
| `command_hold_no_push` | `best` | -22.34 | 0.205 | 414.7 | 0.441 | 0.604 | 0.0021 | 0.0006 | 0.0027 | -0.845 | 1.738 | 26161 | 3.854 | 398908 |
| `command_hold_no_push` | `8000` | -31.86 | 0.143 | 439.4 | 0.346 | 0.520 | 0.0005 | 0.0004 | 0.0026 | -0.853 | 1.734 | 25442 | 3.837 | 379619 |
| `command_hold_style_lowcap` | `best` | -45.33 | 0.154 | 441.2 | 0.358 | 0.498 | 0.0004 | 0.0001 | 0.0048 | -0.724 | 1.593 | 31362 | 4.366 | 300206 |
| `command_hold_style_lowcap` | `8000` | -44.22 | 0.167 | 433.6 | 0.368 | 0.489 | 0.0009 | 0.0001 | 0.0048 | -0.728 | 1.594 | 31139 | 4.344 | 298085 |

### Preset-Level Failure Notes

| experiment | checkpoint | worst-return preset | worst fall preset |
|---|---:|---|---|
| `command_hold_conservative_penalty_ramp` | `best` | `turn_left`, task `-93.46`, fall `0.000` | `run`, fall `0.188` |
| `command_hold_conservative_penalty_ramp` | `8000` | `turn_left`, task `-91.15`, fall `0.016` | `jump`, fall `0.391` |
| `command_hold_controlled_disturb_release` | `best` | `turn_left`, task `-13.58`, fall `0.141` | `run`, fall `1.000` |
| `command_hold_controlled_disturb_release` | `8000` | `run`, task `-94.09`, fall `0.016` | `jump`, fall `0.172` |
| `command_hold_no_push` | `best` | `jump`, task `-28.75`, fall `0.109` | `run`, fall `0.609` |
| `command_hold_no_push` | `8000` | `jump`, task `-41.82`, fall `0.078` | `run`, fall `0.500` |
| `command_hold_style_lowcap` | `best` | `walk_fast`, task `-61.84`, fall `0.031` | `run`, fall `1.000` |
| `command_hold_style_lowcap` | `8000` | `walk_fast`, task `-61.55`, fall `0.062` | `run`, fall `1.000` |

### Supported Conclusion

The July20 result does not support continuing the batch as a direct replacement for `scratch_command_hold`.

Facts:

- `command_hold_controlled_disturb_release` confirms the suspected confound: once disturb release is decoupled from command/terrain curriculum, the final training tail reaches `disturb_curriculum=0.9956` and final train task reward drops to `-4.26`. Its best checkpoint has the best average task return in the batch (`-9.44`) but fails the `run` preset completely (`fall_rate=1.000`), so the good average is not robust.
- `command_hold_no_push` is the best compromise inside July20 by average task return among the stable final checkpoints (`-31.86` at final) and improves from best to final in fall rate (`0.205` to `0.143`). However, it is still worse than July19 `scratch_command_hold_8000` (`-24.24`, fall `0.134`) and keeps a high `run` fall rate (`0.500` final).
- `command_hold_style_lowcap` is stable between best and final (`-45.33` to `-44.22`), but `run` fails completely at both checkpoints (`fall_rate=1.000`), so weaker style pressure alone is not a usable fix.
- `command_hold_conservative_penalty_ramp` has low fall rate (`0.080` final) but very poor task return (`-80.78`) and particularly bad `turn_left`, so conservative penalty ramp trades away task tracking quality.

Interpretation:

- The original `scratch_command_hold` stability was partly due to suppressing disturb release, not only due to fixed command ranges.
- Removing push on top of command hold is the only July20 lever worth keeping as a secondary diagnostic, but it does not beat the July19 command-hold baseline.
- The next serious experiment should not simply continue these four settings. It should use `scratch_command_hold` as the anchor and introduce a staged disturb or command release schedule that avoids a sudden jump to full disturb while separately monitoring the `run` preset.

## Next Step Implementation - 2026-06-21

Hypothesis: July19/July20 point to a coupled failure between command curriculum and disturbance release. The next useful evidence is not another horizontal AMP weight / no-push / penalty-ramp sweep, but a run-only disturbance threshold check plus two staged-release training experiments: one general command-hold anchor and one run-focused follow-up.

Implemented artifacts:

| artifact | status | purpose |
| --- | --- | --- |
| `legged_gym/utils/helpers.py` | implemented | Adds `--eval_disturb_ratio` for fixed disturbance-ratio evaluation. |
| `legged_gym/scripts/evaluate.py` | implemented | Keeps default evaluation disturb-free, but enables fixed-ratio noise disturbance when `--eval_disturb_ratio` is supplied; adds `survival_time_mean_s` to output metrics. |
| `scripts/run_run_disturb_sweep.ps1` | implemented, not run | Runs `run` preset at `0%, 20%, 40%, 60%, 80%, 100%` disturbance and aggregates fall rate, survival time, lin/yaw RMSE, and task return. |
| `legged_gym/envs/r2/r2interrupt_config.py` | implemented | Adds default-off `disturb.staged_release` and stage gate parameters. |
| `legged_gym/envs/r2/r2interrupt.py` | implemented | Clamps `disturb_rad_curriculum` by stage and only advances stage after recent task return and fall-rate gates pass. |
| `configs/ablation/command_hold_staged_disturb_release.json` | implemented, not trained | Fixed command-range anchor with staged disturbance release `0.0 -> 0.25 -> 0.5 -> 0.75 -> 1.0`. |
| `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | implemented, not trained | Same staged disturbance release, but command sampling is biased into the run expert region. |

Training experiment pair:

| experiment | config | controlled factor | status |
| --- | --- | --- | --- |
| `command_hold_staged_disturb_release` | `configs/ablation/command_hold_staged_disturb_release.json` | General fixed command range from `scratch_command_hold`; tests whether staged release alone prevents the July20 full-disturb collapse. | not trained |
| `command_hold_run_focused_staged_disturb_release` | `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | Run-focused command range with forward speed `1.1-1.6`, gait frequency `2.6-3.2`, narrow lateral/yaw range, and foot/body-height caps below the jump-routing thresholds. | not trained |

Run-only disturb sweep command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_run_disturb_sweep.ps1 -Checkpoint 8000
```

Expected output root:

```text
outputs/eval/run_disturb_sweep_command_hold_8000
```

Each ratio writes its own `metrics.csv` / `metrics.json` under `ratio_0p0`, `ratio_0p2`, ..., `ratio_1p0`. The helper also writes:

```text
outputs/eval/run_disturb_sweep_command_hold_8000/run_disturb_sweep_summary.csv
outputs/eval/run_disturb_sweep_command_hold_8000/run_disturb_sweep.png
```

The plot is best-effort; if matplotlib is missing in WSL, the CSV remains the authoritative artifact.

Training is intended to run on the remote Linux machine, not on this Windows host. Use the remote repo path and Conda environment shown in the training example:

```bash
cd ~/lzxworkspace/codespace/VR_Teleoperation
```

General staged disturb release training command from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_staged_disturb_release.json
```

Run-focused staged disturb release training command from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_run_focused_staged_disturb_release.json
```

Optional focused fine-tuning from the July19 command-hold final checkpoint can use either config, if that checkpoint exists on the remote machine:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --resume --load_run July19/Jun19_16-09-11_scratch_command_hold --checkpoint=8000 --cfg_override_json configs/ablation/command_hold_staged_disturb_release.json

CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --resume --load_run July19/Jun19_16-09-11_scratch_command_hold --checkpoint=8000 --cfg_override_json configs/ablation/command_hold_run_focused_staged_disturb_release.json
```

Current status:

- `run-only disturb sweep`: pending; no `metrics.csv` has been generated yet.
- `command_hold_staged_disturb_release`: not trained; no `logs/r2_amp/...command_hold_staged_disturb_release` run directory exists yet.
- `command_hold_run_focused_staged_disturb_release`: not trained; no `logs/r2_amp/...command_hold_run_focused_staged_disturb_release` run directory exists yet.
- Follow-up rule: evaluate the sweep first to locate the run failure threshold, then decide whether staged training should start from scratch or resume from `scratch_command_hold` `8000` / best-task checkpoint.

## Maintenance Rules

When a new experiment is trained or evaluated, update this document in the same turn:

1. Add the training root, config path, run directory, and checkpoint names.
2. State the hypothesis in one sentence before giving results.
3. Record the evaluation protocol if it differs from the default protocol above.
4. Add aggregate metrics from `metrics.csv`; do not rely on visual impression only.
5. Separate facts from interpretation:
   - facts: paths, checkpoint ids, metric values, preset counts.
   - interpretation: whether an experiment is worth continuing and why.
6. If an experiment was planned but not trained, mark it as `not trained` instead of silently dropping it.
7. If evaluation is incomplete, mark it as `pending` and list the missing output directory.
