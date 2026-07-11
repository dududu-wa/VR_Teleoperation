# R2 AMP Experiment Progress

Last updated: 2026-07-11

This document is the running record for R2 AMP ablations. Keep it factual: record what was run, what changed, where the artifacts are, what the evaluation showed, and what conclusion is supported by the data.

## Current Question

The current multi-expert AMP policy can show useful early checkpoints but then regress late in training. The working hypothesis is that late collapse is driven more by command/curriculum/disturbance scheduling than by the discriminator architecture alone.

The first June19 batch tests four levers:

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

### June15 sw1 AMP Weight Failure

Historical run:

```text
logs/r2_amp/Jun15/sw1
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

## June19 Batch

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\June19
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June19_amp_slow_lowcap_best
E:\codebase\VR_Teleoperation\outputs\eval\June19_amp_slow_lowcap_8000
E:\codebase\VR_Teleoperation\outputs\eval\June19_command_hold_best
E:\codebase\VR_Teleoperation\outputs\eval\June19_command_hold_8000
E:\codebase\VR_Teleoperation\outputs\eval\June19_no_push_best
E:\codebase\VR_Teleoperation\outputs\eval\June19_no_push_8000
E:\codebase\VR_Teleoperation\outputs\eval\June19_slow_penalty_ramp_best
E:\codebase\VR_Teleoperation\outputs\eval\June19_slow_penalty_ramp_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per preset.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | status |
|---|---|---|---|---|
| `scratch_amp_slow_lowcap` | `configs/ablation/scratch_amp_slow_lowcap.json` | `logs/r2_amp/Jun19/Jun19_16-08-42_scratch_amp_slow_lowcap` | `1214`, `1227`, `1806` | evaluated |
| `scratch_command_hold` | `configs/ablation/scratch_command_hold.json` | `logs/r2_amp/Jun19/Jun19_16-09-11_scratch_command_hold` | `7120`, `7219`, `7966` | evaluated |
| `scratch_no_push` | `configs/ablation/scratch_no_push.json` | `logs/r2_amp/Jun19/Jun19_16-12-37_scratch_no_push` | `1676`, `1685`, `1881` | evaluated |
| `scratch_slow_penalty_ramp` | `configs/ablation/scratch_slow_penalty_ramp.json` | `logs/r2_amp/Jun19/Jun20_04-58-31_scratch_slow_penalty_ramp` | `1163`, `1219`, `1222` | evaluated |

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

This eval uses the same fixed 7 presets and `--num_episodes=64` protocol as the June19 AMP evals. Existing older PPO evals such as `outputs/eval_r2int_v7_30000_ep4_cpu` used only 3-4 episodes per preset, so they are useful as smoke checks but not the primary comparison.

Aggregate result:

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | lin rmse | yaw rmse | height viol | roll/pitch viol | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r2int_v7` pure PPO | `30000` | -66.85 | 0.185 | 427.8 | 0.421 | 0.663 | 0.0015 | 0.0012 | 37838 | 33.042 | 380717 |
| `scratch_command_hold` AMP | `8000` | -24.24 | 0.134 | 456.3 | 0.409 | 0.558 | 0.0038 | 0.0003 | 32832 | 4.901 | 315275 |
| `scratch_no_push` AMP | `best` | -11.48 | 0.181 | 428.5 | 0.419 | 0.557 | 0.0010 | 0.0003 | 16269 | 2.393 | 302449 |
| `scratch_slow_penalty_ramp` AMP | `best` | -14.85 | 0.185 | 438.3 | 0.413 | 0.568 | 0.0048 | 0.0004 | 24321 | 2.924 | 320162 |

Interpretation:

- On fixed-preset quantitative eval, pure PPO is not better than the best June19 AMP checkpoints. It has worse avg task return than `scratch_command_hold_8000`, `scratch_no_push_best`, and `scratch_slow_penalty_ramp_best`.
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
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | Fixed command ranges remain, but disturb release is decoupled from terrain/command curriculum via `env.disturb.start_by_curriculum=false`; if this collapses, the main stabilizer in `scratch_command_hold` was likely suppressed disturb rather than command range alone. | evaluated in June20 and June21 |
| `command_hold_no_push` | `configs/ablation/command_hold_no_push.json` | Fixed command ranges plus zero randomized base push tests whether push removal improves early quality without reintroducing late drift. | evaluated in June20 |
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | Fixed command ranges plus `penalize_curriculum_sigma=0.9` tests a middle ramp between default `0.8` and the too-slow `0.95` batch. | evaluated in June20 and June21 |
| `command_hold_style_lowcap` | `configs/ablation/command_hold_style_lowcap.json` | Fixed command ranges plus longer AMP warmup and lower task-ratio cap tests whether style pressure should stay weaker after task behavior forms. | evaluated in June20 |

Verification already completed for config plumbing only:

```text
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_amp_training_contracts.py
```

Training and WSL CPU evaluation are recorded in the June20 batch below.

## June20 Batch

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\June20
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_conservative_penalty_ramp_best
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_conservative_penalty_ramp_8000
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_controlled_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_controlled_disturb_release_8000
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_no_push_best
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_no_push_8000
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_style_lowcap_best
E:\codebase\VR_Teleoperation\outputs\eval\June20_command_hold_style_lowcap_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed in this batch because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | status |
|---|---|---|---|---:|---:|---|
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | `logs/r2_amp/Jun20/Jun20_15-18-58_command_hold_conservative_penalty_ramp` | `5818`, `7663`, `7930` | `23.49` | `0.0000` | evaluated |
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | `logs/r2_amp/Jun20/Jun20_15-19-48_command_hold_controlled_disturb_release` | `1166`, `1706`, `1944` | `-4.26` | `0.9956` | evaluated |
| `command_hold_no_push` | `configs/ablation/command_hold_no_push.json` | `logs/r2_amp/Jun20/Jun20_15-21-52_command_hold_no_push` | `6059`, `6973`, `7440` | `31.15` | `0.0000` | evaluated |
| `command_hold_style_lowcap` | `configs/ablation/command_hold_style_lowcap.json` | `logs/r2_amp/Jun20/Jun20_15-22-56_command_hold_style_lowcap` | `7439`, `7600`, `7937` | `28.81` | `0.0000` | evaluated |

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

The June20 result does not support continuing the batch as a direct replacement for `scratch_command_hold`.

Facts:

- `command_hold_controlled_disturb_release` confirms the suspected confound: once disturb release is decoupled from command/terrain curriculum, the final training tail reaches `disturb_curriculum=0.9956` and final train task reward drops to `-4.26`. Its best checkpoint has the best average task return in the batch (`-9.44`) but fails the `run` preset completely (`fall_rate=1.000`), so the good average is not robust.
- `command_hold_no_push` is the best compromise inside June20 by average task return among the stable final checkpoints (`-31.86` at final) and improves from best to final in fall rate (`0.205` to `0.143`). However, it is still worse than June19 `scratch_command_hold_8000` (`-24.24`, fall `0.134`) and keeps a high `run` fall rate (`0.500` final).
- `command_hold_style_lowcap` is stable between best and final (`-45.33` to `-44.22`), but `run` fails completely at both checkpoints (`fall_rate=1.000`), so weaker style pressure alone is not a usable fix.
- `command_hold_conservative_penalty_ramp` has low fall rate (`0.080` final) but very poor task return (`-80.78`) and particularly bad `turn_left`, so conservative penalty ramp trades away task tracking quality.

Interpretation:

- The original `scratch_command_hold` stability was partly due to suppressing disturb release, not only due to fixed command ranges.
- Removing push on top of command hold is the only June20 lever worth keeping as a secondary diagnostic, but it does not beat the June19 command-hold baseline.
- The next serious experiment should not simply continue these four settings. It should use `scratch_command_hold` as the anchor and introduce a staged disturb or command release schedule that avoids a sudden jump to full disturb while separately monitoring the `run` preset.

## June21 Batch

Hypothesis: rerun the two high-signal June20 levers to check whether the poor June20 conservative-ramp result was reproducible, and whether full decoupled disturbance release remains unstable under a fresh run.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\June21
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June21_command_hold_conservative_penalty_ramp_best
E:\codebase\VR_Teleoperation\outputs\eval\June21_command_hold_conservative_penalty_ramp_8000
E:\codebase\VR_Teleoperation\outputs\eval\June21_command_hold_controlled_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June21_command_hold_controlled_disturb_release_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | best train task reward | status |
|---|---|---|---|---:|---:|---:|---|
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | `logs/r2_amp/Jun21/Jun21_12-28-33_command_hold_conservative_penalty_ramp` | `7075`, `7654`, `7657` | `32.80` | `0.0000` | `34.69` | evaluated |
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | `logs/r2_amp/Jun21/Jun21_12-28-55_command_hold_controlled_disturb_release` | `1450`, `1498`, `1608` | `9.36` | `0.9943` | `21.39` | evaluated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | survival s | lin rmse | yaw rmse | height viol | roll/pitch viol | style reward | policy logit | disc gap | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_conservative_penalty_ramp` | `best` | -34.71 | 0.051 | 483.8 | 9.68 | 0.340 | 0.399 | 0.0015 | 0.0001 | 0.0055 | -0.686 | 1.574 | 28744 | 2.786 | 288564 |
| `command_hold_conservative_penalty_ramp` | `8000` | -33.57 | 0.045 | 484.0 | 9.68 | 0.359 | 0.406 | 0.0003 | 0.0000 | 0.0054 | -0.692 | 1.577 | 28325 | 2.858 | 282446 |
| `command_hold_controlled_disturb_release` | `best` | -9.98 | 0.676 | 236.5 | 4.73 | 0.641 | 0.749 | 0.0012 | 0.0014 | 0.0044 | -0.774 | 1.640 | 19425 | 4.042 | 450292 |
| `command_hold_controlled_disturb_release` | `8000` | -33.83 | 0.498 | 309.7 | 6.19 | 0.634 | 0.614 | 0.0064 | 0.0014 | 0.0023 | -0.863 | 1.768 | 32892 | 7.140 | 393637 |

### Preset-Level Failure Notes

| experiment | checkpoint | worst-return preset | worst fall preset |
|---|---:|---|---|
| `command_hold_conservative_penalty_ramp` | `best` | `run`, task `-54.38`, fall `0.109` | `run`, fall `0.109` |
| `command_hold_conservative_penalty_ramp` | `8000` | `run`, task `-46.21`, fall `0.078` | `stand`, fall `0.188` |
| `command_hold_controlled_disturb_release` | `best` | `jump`, task `-14.95`, fall `0.984` | `stand`, fall `1.000`; `walk_fast` and `run` also fall `1.000` |
| `command_hold_controlled_disturb_release` | `8000` | `walk_fast`, task `-46.18`, fall `0.625` | `run`, fall `1.000` |

### Supported Conclusion

The June21 result partially revises the June20 interpretation.

Facts:

- `command_hold_conservative_penalty_ramp` no longer looks like a dead end. The June21 final checkpoint is much better than the June20 final checkpoint on task return (`-33.57` vs `-80.78`) while keeping a very low average fall rate (`0.045`). It is still worse than June19 `scratch_command_hold_8000` on average task return (`-33.57` vs `-24.24`), but it is better on fall rate (`0.045` vs `0.134`) and action-rate L2 (`2.858` vs `4.901`).
- `command_hold_controlled_disturb_release` remains unstable under full decoupled disturbance release. Its best checkpoint has the best average task return in June21 (`-9.98`) only because many episodes terminate early; the average fall rate is `0.676`, with `stand`, `walk_fast`, and `run` all at `fall_rate = 1.000`. The final checkpoint reduces average fall rate to `0.498`, but `run` still fails completely.
- The training tails are consistent with the mechanism split: conservative penalty ramp keeps `disturb_curriculum=0.0000`, while controlled release reaches `disturb_curriculum=0.9943` and remains much less robust.

Interpretation:

- Keep `command_hold_conservative_penalty_ramp` as a stability-biased control or fallback, especially if low fall rate and smoothness matter more than peak tracking return.
- Do not continue the full `command_hold_controlled_disturb_release` setting as-is. It is evidence that sudden full disturbance release is too harsh, not a usable continuation policy.
- The next experiment should still be the run-only disturb sweep and staged release plan below, because June21 strengthens the case for gradual disturbance release rather than direct full release.

## Next Step Implementation - 2026-06-21

Hypothesis: June19/June20 point to a coupled failure between command curriculum and disturbance release. The next useful evidence is not another horizontal AMP weight / no-push / penalty-ramp sweep, but a run-only disturbance threshold check plus two staged-release training experiments: one general command-hold anchor and one run-focused follow-up.

Update on 2026-06-22: June21 showed that aggregate task return can hide run-specific failure, so the staged-release gate now supports `stage_monitor_expert`. Both staged JSONs set `stage_monitor_expert="run"`; stage advancement therefore waits for run-routed noise-disturb episodes to satisfy the task-return/fall-rate gate.

Implemented artifacts:

| artifact | status | purpose |
| --- | --- | --- |
| `legged_gym/utils/helpers.py` | implemented | Adds `--eval_disturb_ratio` for fixed disturbance-ratio evaluation. |
| `legged_gym/scripts/evaluate.py` | implemented, updated 2026-06-29 | Keeps default evaluation disturb-free, but enables fixed-ratio noise disturbance when `--eval_disturb_ratio` is supplied; adds `survival_time_mean_s` to output metrics. With `--record_reward_terms`, it now writes per-preset `reward_terms.csv/json` diagnostics without changing the default `metrics.csv/json` schema. |
| `scripts/run_run_disturb_sweep.ps1` | implemented, run 2026-06-30 | Runs `run` preset at `0%, 20%, 40%, 60%, 80%, 100%` disturbance and aggregates fall rate, survival time, lin/yaw RMSE, and task return. |
| `legged_gym/envs/r2/r2interrupt_config.py` | implemented, updated 2026-06-25 | Adds default-off `disturb.staged_release`, stage gate parameters, optional `stage_monitor_expert` / `stage_monitor_profiles` filtering, scalar-or-list staged gate thresholds, and optional adaptive stage regression after repeated failed windows. |
| `legged_gym/envs/r2/r2_config.py` | implemented, updated 2026-06-25 | Adds default-off `commands.profile_mixture=None` so targeted JSONs can opt into profile-based command sampling without changing legacy rectangular command sampling. |
| `legged_gym/envs/r2/r2interrupt.py` | implemented, updated 2026-06-25 | Clamps `disturb_rad_curriculum` by stage and only advances stage after recent task return and fall-rate gates pass; optional expert filtering uses the same command semantics as AMP hard routing; per-stage gate lists let early stages use looser gates before tightening later; `commands.profile_mixture` can replace rectangular commands with weighted jittered eval-like profiles and now records profile ids so staged gates can monitor named profiles; optional adaptive regression lowers one stage after repeated failed gate windows. |
| `configs/ablation/command_hold_staged_disturb_release.json` | implemented, trained and evaluated in June23 | Fixed command-range anchor with staged disturbance release `0.0 -> 0.25 -> 0.5 -> 0.75 -> 1.0`; stage gate monitors run-routed noise-disturb episodes. |
| `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | implemented, trained and evaluated in June23 | Same staged disturbance release, but command sampling and the stage gate are both biased into the run expert region. |
| `configs/ablation/command_hold_run_recovery_staged_disturb_release.json` | implemented, trained and evaluated in June24 | June23 follow-up: uses a walk-run transition command band, finer staged levels, and per-stage task-return/fall-rate gates that start permissive and tighten toward the original full-disturb target. |
| `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json` | trained/evaluated twice; Jun25_0 rerun failed | June25 follow-up: samples weighted jittered anchors matching the seven fixed `evaluate.py` presets. After the first eval-manifold run improved run but failed stand/jump, the next version monitored all seven named profiles and enabled adaptive stage regression; the Jun25_0 rerun collapsed late and is not a continuation policy. |
| `configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json` | trained and evaluated in Jun25_0 | Adjacent June25 diagnostic: keeps the same eval-manifold profile mixture and all-profile staged monitoring, but caps staged disturbance at `0.75`, uses finer early stages, and lengthens the stage window to test whether full disturbance pressure caused stand/jump collapse and late regression. |

Training experiment pair:

| experiment | config | controlled factor | status |
| --- | --- | --- | --- |
| `command_hold_staged_disturb_release` | `configs/ablation/command_hold_staged_disturb_release.json` | General fixed command range from `scratch_command_hold`; tests whether staged release prevents the full-disturb collapse when stage advancement is gated by run-routed episodes. | evaluated in June23 |
| `command_hold_run_focused_staged_disturb_release` | `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | Run-focused command range with forward speed `1.1-1.6`, gait frequency `2.6-3.2`, narrow lateral/yaw range, foot/body-height caps below the jump-routing thresholds, and run-routed stage monitoring. | evaluated in June23 |
| `command_hold_run_recovery_staged_disturb_release` | `configs/ablation/command_hold_run_recovery_staged_disturb_release.json` | Softer run-recovery follow-up after June23: command sampling spans the walk-run transition, stage levels are smaller at the start, and per-stage gates allow early progress with high but improving run fall rate before tightening later. | evaluated in June24 |
| `command_hold_eval_manifold_staged_disturb_release` | `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json` | Weighted jittered command profiles around `stand`, `walk_slow`, `walk_fast`, `run`, `jump`, `turn_left`, and `strafe_right`; first run used a run-routed staged gate, rerun used all-profile staged monitoring plus adaptive regression. | evaluated in first June25 run and Jun25_0 rerun; Jun25_0 rerun failed |
| `command_hold_eval_manifold_conservative_disturb_release` | `configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json` | Same eval-like profiles and profile-aware gate as the updated eval-manifold config, but conservative staged release `0.0 -> 0.05 -> 0.1 -> 0.18 -> 0.28 -> 0.42 -> 0.6 -> 0.75` with `stage_min_episodes=2048`. | trained and evaluated in Jun25_0 |

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

Run-only disturb sweep result from `scratch_command_hold` `model_8000.pt`, 64 episodes per row:

| disturb ratio | task return | fall rate | survival s | lin rmse | yaw rmse |
|---:|---:|---:|---:|---:|---:|
| 0.0 | -8.18 | 0.922 | 3.28 | 1.166 | 1.153 |
| 0.2 | -6.63 | 0.391 | 7.67 | 0.946 | 1.241 |
| 0.4 | -11.49 | 0.688 | 5.47 | 1.252 | 1.358 |
| 0.6 | -5.44 | 0.422 | 7.19 | 1.012 | 1.222 |
| 0.8 | -17.55 | 0.828 | 5.12 | 1.231 | 1.481 |
| 1.0 | -11.51 | 1.000 | 1.30 | 1.275 | 1.325 |

Facts:

- The pending sweep has now been generated at `outputs/eval/run_disturb_sweep_command_hold_8000`; each ratio directory contains one `run` row in `metrics.csv`, and the helper also wrote `run_disturb_sweep_summary.csv` plus `run_disturb_sweep.png`.
- The zero-disturb row is much worse than the earlier full seven-preset June19 aggregate suggested for `run`, with `fall_rate=0.922` and survival `3.28s` in this fresh fixed-ratio run-only protocol. This means the old command-hold final checkpoint should not be used as a reliable run warm-start anchor without rechecking seed/protocol sensitivity.
- Full disturbance still fails completely (`fall_rate=1.000`, survival `1.30s`), while mid ratios are noisy rather than monotonic. The useful conclusion is not a precise threshold but that `scratch_command_hold` `8000` does not have robust run recovery under the corrected fixed-ratio evaluator.

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

Run-recovery staged disturb release training command from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_run_recovery_staged_disturb_release.json
```

Eval-manifold staged disturb release training command from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_eval_manifold_staged_disturb_release.json
```

Conservative eval-manifold staged disturb release training command from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json
```

Optional focused fine-tuning from the June19 command-hold final checkpoint can use the staged configs if that checkpoint exists on the remote machine:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --resume --load_run Jun19/Jun19_16-09-11_scratch_command_hold --checkpoint=8000 --cfg_override_json configs/ablation/command_hold_staged_disturb_release.json

CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --resume --load_run Jun19/Jun19_16-09-11_scratch_command_hold --checkpoint=8000 --cfg_override_json configs/ablation/command_hold_run_focused_staged_disturb_release.json
```

Current status:

- `run-only disturb sweep`: completed on 2026-06-30 at `outputs/eval/run_disturb_sweep_command_hold_8000`; `scratch_command_hold` `8000` is not a robust run warm-start anchor under this protocol.
- `command_hold_staged_disturb_release`: trained and evaluated in June23.
- `command_hold_run_focused_staged_disturb_release`: trained and evaluated in June23.
- `command_hold_run_recovery_staged_disturb_release`: trained and evaluated in June24.
- `command_hold_eval_manifold_staged_disturb_release`: trained and evaluated in the first June25 run and again in Jun25_0; the Jun25_0 rerun collapsed late and is not a continuation policy.
- `command_hold_eval_manifold_conservative_disturb_release`: trained and evaluated in Jun25_0; it improves final fixed-preset fall rate but produces poor task return and remains diagnostic rather than a final policy.
- Follow-up rule: stop adding from-scratch staged-gate variants until the train/eval objective mismatch is isolated. The evidence now points to two separate problems: all-profile staged monitoring can still destabilize training from scratch, while conservative staged disturbance can produce survival without useful task performance. The next code-level diagnostic is implemented through `evaluate.py --record_reward_terms`.

## June23 Batch

Hypothesis: staged disturbance release should avoid the sudden full-disturb collapse seen in June20/June21, while `stage_monitor_expert="run"` should stop aggregate walk/stand success from hiding the known run failure.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\June23
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June23_command_hold_staged_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June23_command_hold_staged_disturb_release_8000
E:\codebase\VR_Teleoperation\outputs\eval\June23_command_hold_run_focused_staged_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June23_command_hold_run_focused_staged_disturb_release_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | final staged level / stage | final staged window fall rate | best train task reward | status |
|---|---|---|---|---:|---:|---|---:|---:|---|
| `command_hold_staged_disturb_release` | `configs/ablation/command_hold_staged_disturb_release.json` | `logs/r2_amp/Jun23/Jun23_03-38-06_command_hold_staged_disturb_release` | `1315`, `1331`, `1705` | `7.36` | `0.9944` | `1.0000 / 4` | `0.1174` | `26.76` | evaluated |
| `command_hold_run_focused_staged_disturb_release` | `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | `logs/r2_amp/Jun23/Jun23_14-58-32_command_hold_run_focused_staged_disturb_release` | `4221`, `4294`, `7112` | `8.14` | `0.0000` | `0.0000 / 0` | `0.6372` | `12.08` | evaluated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | survival s | lin rmse | yaw rmse | height viol | roll/pitch viol | style reward | policy logit | disc gap | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_staged_disturb_release` | `best` | -14.39 | 0.540 | 282.9 | 5.66 | 0.569 | 0.664 | 0.0028 | 0.0007 | 0.0037 | -0.807 | 1.673 | 21912 | 3.113 | 320423 |
| `command_hold_staged_disturb_release` | `8000` | -41.51 | 0.219 | 411.7 | 8.23 | 0.405 | 0.488 | 0.0005 | 0.0003 | 0.0056 | -0.641 | 1.522 | 35707 | 6.296 | 310039 |
| `command_hold_run_focused_staged_disturb_release` | `best` | -7.91 | 0.978 | 58.3 | 1.17 | 0.856 | 1.170 | 0.0147 | 0.0038 | 0.0076 | -0.517 | 1.195 | 19599 | 4.700 | 431233 |
| `command_hold_run_focused_staged_disturb_release` | `8000` | -8.46 | 1.000 | 37.5 | 0.75 | 0.883 | 1.198 | 0.0143 | 0.0031 | 0.0071 | -0.520 | 1.203 | 25466 | 9.140 | 460219 |

### Preset-Level Failure Notes

| experiment | checkpoint | worst-return preset | worst fall preset |
|---|---:|---|---|
| `command_hold_staged_disturb_release` | `best` | `turn_left`, task `-23.01`, fall `0.297` | `run`, fall `1.000` |
| `command_hold_staged_disturb_release` | `8000` | `walk_fast`, task `-50.78`, fall `0.281` | `run`, fall `0.656` |
| `command_hold_run_focused_staged_disturb_release` | `best` | `walk_fast`, task `-8.68`, fall `0.938` | `stand`, `walk_slow`, `jump`, `turn_left`, `strafe_right`, fall `1.000` |
| `command_hold_run_focused_staged_disturb_release` | `8000` | `run`, task `-10.10`, fall `1.000` | all seven presets, fall `1.000` |

### Run Preset Diagnostic

| experiment | checkpoint | run task return | run fall rate | run length steps | run survival s | run lin rmse | run yaw rmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_staged_disturb_release` | `best` | -7.15 | 1.000 | 28.2 | 0.56 | 1.345 | 1.025 |
| `command_hold_staged_disturb_release` | `8000` | -35.17 | 0.656 | 196.7 | 3.93 | 1.057 | 0.942 |
| `command_hold_run_focused_staged_disturb_release` | `best` | -8.56 | 0.906 | 101.5 | 2.03 | 1.469 | 1.076 |
| `command_hold_run_focused_staged_disturb_release` | `8000` | -10.10 | 1.000 | 51.0 | 1.02 | 1.530 | 1.162 |

### Supported Conclusion

The June23 result does not support either staged-release run as a direct continuation policy.

Facts:

- `command_hold_staged_disturb_release` reached full staged disturbance release in training (`staged_disturb_level=1.0000`, `staged_disturb_stage=4`, `disturb_curriculum=0.9944`). The final checkpoint reduced average fall rate relative to best (`0.219` vs `0.540`) and improved average survival (`8.23s` vs `5.66s`), but average task return degraded from `-14.39` to `-41.51`, and the `run` preset still had `fall_rate=0.656`.
- `command_hold_staged_disturb_release` best checkpoint had a strong average task return (`-14.39`) but failed `run` completely (`fall_rate=1.000`), so the average is not robust enough for continuation.
- `command_hold_run_focused_staged_disturb_release` did not advance beyond stage 0 in training (`staged_disturb_level=0.0000`, `staged_disturb_stage=0`, `disturb_curriculum=0.0000`) and still failed the fixed-preset evaluation. Its best checkpoint had average fall rate `0.978`; its final checkpoint had `fall_rate=1.000` on all seven presets.
- The apparently good average task return for the run-focused run (`-7.91` best, `-8.46` final) is misleading because episodes terminate very early: average survival is only `1.17s` and `0.75s`.

Interpretation:

- Staged release is better evidence than full immediate disturbance release because the general staged run can reach full disturbance without complete aggregate collapse. However, it still does not solve run robustness.
- The run-focused JSON is too unstable as a from-scratch setting under the current gate. It either needs a warm start from a stronger command-hold checkpoint, a less severe run-only curriculum, or a lower initial gate burden before it can test staged disturbance meaningfully.
- The later run-only disturbance sweep shows that `scratch_command_hold` `8000` is not a reliable run warm-start anchor under fixed-ratio evaluation. Do not use it as the next continuation target without additional seed/protocol checks.

## Follow-up Implementation - 2026-06-24

Hypothesis: the June23 run-focused setting was too harsh from scratch because it combined a run-only command band with a strict global staged gate. A softer follow-up should keep run-routed monitoring, but start from a walk-run transition command band and use per-stage gates that tighten as disturbance increases.

Code change:

```text
legged_gym/envs/r2/r2interrupt_config.py
legged_gym/envs/r2/r2interrupt.py
tests/test_amp_training_contracts.py
```

`stage_min_task_return` and `stage_max_fall_rate` remain backward-compatible scalar config fields. They may now also be lists with the same length as `stage_levels`; `_expand_staged_disturb_gate_values()` validates this contract, and `_current_staged_disturb_gate()` selects the gate for the current stage. The training log now also exposes `staged_disturb_gate_min_task_return` and `staged_disturb_gate_max_fall_rate` in episode extras.

New config:

```text
configs/ablation/command_hold_run_recovery_staged_disturb_release.json
```

Key settings:

| field | value | reason |
|---|---|---|
| `commands.ranges.lin_vel_x` | `[0.8, 1.35]` | Covers the walk-run transition instead of forcing all commands into the harsher `1.1-1.6` run-only band. |
| `commands.ranges.gait_frequency` | `[1.8, 2.7]` | Lets some samples remain near the run-routing boundary while still producing run-routed monitor episodes. |
| `disturb.stage_levels` | `[0.0, 0.15, 0.3, 0.5, 0.75, 1.0]` | Uses smaller early disturbance increments than the June23 `0.25` jump. |
| `disturb.stage_min_task_return` | `[4.0, 8.0, 12.0, 16.0, 20.0, 20.0]` | Starts below the June23 final run-focused window return and tightens toward the original target. |
| `disturb.stage_max_fall_rate` | `[0.7, 0.55, 0.4, 0.25, 0.15, 0.1]` | Allows stage 0 to progress despite imperfect run recovery, then progressively requires robustness. |
| `disturb.stage_monitor_expert` | `run` | Keeps the June21/June23 diagnosis that run-routed episodes must drive the gate. |

Training command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_run_recovery_staged_disturb_release.json
```

Current status:

- `command_hold_run_recovery_staged_disturb_release`: trained and evaluated in June24; see the June24 batch below.
- Required evaluation after training: completed with the same WSL CPU fixed-preset protocol as June23, evaluating both `model_best_task.pt` and `model_8000.pt`.

## June24 Batch

Hypothesis: a softer walk-run transition command band plus finer per-stage disturbance gates should recover some run stability without forcing the from-scratch policy into the harsher run-only regime that failed in June23.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June24_command_hold_run_recovery_staged_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June24_command_hold_run_recovery_staged_disturb_release_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | final staged level / stage | final staged window task return | final staged window fall rate | best train task reward | status |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| `command_hold_run_recovery_staged_disturb_release` | `configs/ablation/command_hold_run_recovery_staged_disturb_release.json` | `logs/r2_amp/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release` | `6996`, `7752`, `7845` | `19.90` | `0.1651` | `0.3000 / 2` | `17.8358` | `0.4707` | `22.36` | evaluated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | survival s | lin rmse | yaw rmse | height viol | roll/pitch viol | style reward | policy logit | disc gap | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_run_recovery_staged_disturb_release` | `best` | -8.56 | 0.920 | 77.7 | 1.55 | 0.584 | 0.936 | 0.0047 | 0.0029 | 0.0042 | -0.728 | 1.530 | 25481 | 9.406 | 401193 |
| `command_hold_run_recovery_staged_disturb_release` | `8000` | -8.49 | 0.933 | 71.4 | 1.43 | 0.615 | 0.888 | 0.0048 | 0.0039 | 0.0040 | -0.743 | 1.542 | 24938 | 8.070 | 387569 |

### Preset-Level Failure Notes

| experiment | checkpoint | worst-return preset | worst fall preset |
|---|---:|---|---|
| `command_hold_run_recovery_staged_disturb_release` | `best` | `run`, task `-11.54`, fall `0.703` | `stand`, `walk_slow`, `jump`, `turn_left`, `strafe_right`, fall `1.000` |
| `command_hold_run_recovery_staged_disturb_release` | `8000` | `jump`, task `-9.95`, fall `1.000` | `stand`, `walk_slow`, `jump`, `turn_left`, `strafe_right`, fall `1.000`; `run` nearly fails at `0.984` |

### Run Preset Diagnostic

| experiment | checkpoint | run task return | run fall rate | run length steps | run survival s | run lin rmse | run yaw rmse |
|---|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_run_recovery_staged_disturb_release` | `best` | -11.54 | 0.703 | 175.7 | 3.51 | 0.899 | 0.812 |
| `command_hold_run_recovery_staged_disturb_release` | `8000` | -9.92 | 0.984 | 45.2 | 0.90 | 1.212 | 0.839 |

### Supported Conclusion

The June24 run-recovery setting is useful diagnostic evidence, but it is not a direct continuation policy.

Facts:

- Training advanced farther than the June23 run-focused run, reaching `staged_disturb_stage=2` and `staged_disturb_level=0.3000` instead of staying at stage 0. However, it did not reach full staged disturbance release; the tail still had `disturb_curriculum=0.1651` and staged window fall rate `0.4707`, above the current stage gate target `0.4000`.
- Fixed-preset evaluation still fails broadly. The best checkpoint has average fall rate `0.920` and average survival only `1.55s`; the final checkpoint has average fall rate `0.933` and average survival `1.43s`.
- The `run` preset improved relative to the June23 run-focused final checkpoint, but remains unusable: best has `fall_rate=0.703`, while final regresses to `0.984`.
- The high-looking average task return (`-8.56` best, `-8.49` final) is misleading for the same reason as the June23 run-focused result: many episodes terminate early, so survival and fall-rate metrics must dominate the conclusion.

Interpretation:

- The softer walk-run band and permissive early gates helped training pass the first two staged levels, so this change diagnosed that the previous run-only setting was too harsh.
- It did not solve robustness. The policy still collapses under fixed-preset evaluation, including non-run presets that were not supposed to be the primary target.
- The next step should not be another from-scratch run-focused staged variant. The now-completed command-hold run-only disturb sweep weakens the case for warm-starting from `scratch_command_hold` `8000`; later conservative `8000` diagnostics are the stronger local anchor.

## Follow-up Implementation - 2026-06-25

Hypothesis: the June24 run-recovery config improved staged training progress but still failed fixed-preset evaluation because the training command distribution did not cover the evaluation manifold. The next code change should therefore make the command sampler optionally sample around the seven fixed `evaluate.py` presets instead of only widening or narrowing rectangular command ranges.

Code change:

```text
legged_gym/envs/r2/r2_config.py
legged_gym/envs/r2/r2interrupt.py
tests/test_amp_training_contracts.py
```

`R2Cfg.commands.profile_mixture` is default-off (`None`) to preserve all existing configs. When a JSON supplies profiles, `R2InterruptRobot._apply_command_profile_mixture()` samples one profile per reset environment with `torch.multinomial`, writes `command + uniform_jitter`, records `command_profile_ids`, clips the result to `command_ranges`, and recomputes `velocity_level`. The sampler clears `standing_envs_mask` by default because `evaluate.py` clears it for fixed-preset evaluation; a profile must explicitly set `standing=true` to opt back into standing-specific rewards.

Update after evaluating the first eval-manifold run: `stage_monitor_expert="run"` improved the run preset but did not protect stand and jump. The config now sets `stage_monitor_expert=null`, monitors all seven `stage_monitor_profiles`, increases stand/jump sampling weights, and enables `stage_regress_on_failure=true` with `stage_regress_patience=2`.

New config:

```text
configs/ablation/command_hold_eval_manifold_staged_disturb_release.json
```

Key settings:

| field | value | reason |
|---|---|---|
| `commands.profile_mixture` | seven weighted profiles: `stand`, `walk_slow`, `walk_fast`, `run`, `jump`, `turn_left`, `strafe_right`; next-run weights emphasize `stand` and `jump` more than the first run | Matches the fixed evaluation presets while retaining small jitter for local robustness; the first run still failed stand/jump. |
| `commands.ranges.lin_vel_x` | `[0.0, 1.7]` | Covers stationary, walking, and run fixed-preset speeds. |
| `commands.ranges.lin_vel_y` | `[-0.35, 0.35]` | Covers the `strafe_right` preset and its jitter. |
| `commands.ranges.ang_vel_yaw` | `[-0.65, 0.65]` | Covers the `turn_left` preset and leaves symmetric support. |
| `disturb.stage_levels` | `[0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]` | Uses smaller early disturbance increments than June24 so command coverage and disturbance release do not both change harshly. |
| `disturb.stage_monitor_expert` | `null` | Stops the staged gate from filtering out non-run profiles after stand/jump failed in fixed-preset evaluation. |
| `disturb.stage_monitor_profiles` | all seven eval profiles | Makes the staged window explicitly profile-aware so no single AMP expert can hide a weak preset. |
| `disturb.stage_regress_on_failure` / `stage_regress_patience` | `true` / `2` | Backs off one stage after repeated failed windows, addressing the observed post-best regression by adapting difficulty to demonstrated competence. |

Training command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_eval_manifold_staged_disturb_release.json
```

Adjacent conservative config:

```text
configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json
```

This config is not a replacement for the updated eval-manifold run. It is a diagnostic variant for the specific failure mode observed in the first June25 run: stand/jump failed and the final checkpoint regressed after training reached full disturbance. It keeps the same seven eval-like profiles, all-profile staged monitoring, and adaptive regression, but caps staged disturbance at `0.75`, uses smaller early increments, and raises `stage_min_episodes` to `2048`. If this variant improves stand/jump while the full-disturb variant does not, full disturbance pressure is likely the main driver; if it still fails, the next target should be profile-specific reward/termination semantics rather than only disturb release.

Training command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --cfg_override_json configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json
```

Current status:

- `command_hold_eval_manifold_staged_disturb_release`: first run trained/evaluated from `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release`; Jun25_0 rerun trained/evaluated from `logs/r2_amp/Jun25_0/Jun25_05-00-11_command_hold_eval_manifold_staged_disturb_release`.
- `command_hold_eval_manifold_conservative_disturb_release`: Jun25_0 run trained/evaluated from `logs/r2_amp/Jun25_0/Jun25_04-43-45_command_hold_eval_manifold_conservative_disturb_release`.
- First-run evaluation outputs: `outputs/eval/June25_command_hold_eval_manifold_staged_disturb_release_best` and `outputs/eval/June25_command_hold_eval_manifold_staged_disturb_release_8000`.
- Jun25_0 evaluation outputs: `outputs/eval/June29_Jun25_0_eval_manifold_conservative_best`, `outputs/eval/June29_Jun25_0_eval_manifold_conservative_8000`, `outputs/eval/June29_Jun25_0_eval_manifold_staged_best`, and `outputs/eval/June29_Jun25_0_eval_manifold_staged_8000`.
- Next required work: do not continue either Jun25_0 checkpoint as a policy. Use the Jun25_0 result to design a smaller diagnostic around reward/termination and profile sampling rather than another broad staged-release rerun.

## June25 Eval-Manifold Batch

Hypothesis: sampling jittered command profiles around the seven fixed evaluation presets should reduce the train/eval command-manifold mismatch that made the June24 run-recovery policy collapse broadly under fixed-preset evaluation.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release
```

Evaluation outputs:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June25_command_hold_eval_manifold_staged_disturb_release_best
E:\codebase\VR_Teleoperation\outputs\eval\June25_command_hold_eval_manifold_staged_disturb_release_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed because `--compute_dtw` was not passed.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final disturb curriculum | final staged level / stage | final staged window task return | final staged window fall rate | best train task reward | status |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| `command_hold_eval_manifold_staged_disturb_release` | `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json` | `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release` | `1490`, `1517`, `1518` | `5.37` | `0.9972` | `1.0000 / 6` | `6.8213` | `0.0571` | `33.92` | evaluated; config updated |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_eval_manifold_staged_disturb_release` | `best` | -7.20 | 0.496 | 322.3 | 6.45 | 0.495 | 0.710 | 0.0040 | -0.805 | 1.677 |
| `command_hold_eval_manifold_staged_disturb_release` | `8000` | -35.71 | 0.533 | 315.3 | 6.31 | 0.542 | 0.662 | 0.0022 | -0.890 | 1.798 |

### Preset-Level Failure Notes

| checkpoint | worst-return preset | worst fall preset |
|---:|---|---|
| `best` | `jump`, task `-10.21`, fall `1.000`; `stand`, task `-10.01`, fall `1.000` | `stand` and `jump`, fall `1.000` |
| `8000` | `turn_left`, task `-50.62`, fall `0.375`; `run`, task `-45.48`, fall `0.328` | `stand` and `jump`, fall `1.000` |

### Run Preset Diagnostic

| checkpoint | run task return | run fall rate | run length steps | run survival s | run lin rmse | run yaw rmse |
|---:|---:|---:|---:|---:|---:|---:|
| `best` | -8.06 | 0.250 | 388.4 | 7.77 | 0.659 | 0.723 |
| `8000` | -45.48 | 0.328 | 363.9 | 7.28 | 0.734 | 0.735 |

### Supported Conclusion

The eval-manifold change is useful but incomplete.

Facts:

- The best checkpoint is a large improvement over June24 run-recovery on aggregate fixed-preset evaluation: avg fall rate improved from `0.920` to `0.496`, avg survival from `1.55s` to `6.45s`, and the run preset fall rate from `0.703` to `0.250`.
- The final checkpoint regressed badly in task return: avg task return fell from `-7.20` at best to `-35.71` at final, matching the training-log drop from best train task reward `33.92` near iteration `1517` to final train task reward `5.37`.
- The run-only staged gate missed non-run failures. `stand` and `jump` both have `fall_rate=1.000` at best and final, even though training reached `staged_disturb_level=1.0000`.

Interpretation:

- Keep eval-manifold sampling; it improved the exact failure that motivated it, especially run robustness and survival.
- Do not continue the first eval-manifold checkpoint as the final policy because stand/jump are unusable and late training regresses.
- The next version should monitor all named eval profiles, not only the run expert, and should allow staged disturbance to back off after repeated failed windows. This is now implemented in `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json`, `legged_gym/envs/r2/r2interrupt_config.py`, and `legged_gym/envs/r2/r2interrupt.py`.

## Jun25_0 Eval-Manifold Rerun Batch

Hypothesis: profile-aware staged monitoring plus adaptive stage regression should stop non-run profiles from being hidden by the stage gate, while the conservative variant should test whether capping disturbance at `0.75` avoids the first eval-manifold run's stand/jump collapse and late regression.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jun25_0
```

Evaluation outputs generated on 2026-06-29:

```text
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_conservative_best
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_conservative_8000
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_conservative_best_corrected
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_conservative_8000_corrected
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_staged_best
E:\codebase\VR_Teleoperation\outputs\eval\June29_Jun25_0_eval_manifold_staged_8000
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per fixed preset. DTW was not computed because `--compute_dtw` was not passed. The `*_corrected` conservative outputs use the fixed default evaluator that clears applied interrupt masks but preserves `R2InterruptRobot` reward masking semantics.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | final train task reward | final episode length | final disturb curriculum | final staged level / stage | final staged window task return | final staged window fall rate | best train task reward | status |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| `command_hold_eval_manifold_conservative_disturb_release` | `configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json` | `logs/r2_amp/Jun25_0/Jun25_04-43-45_command_hold_eval_manifold_conservative_disturb_release` | `2637`, `2646`, `2677` | `21.04` | `932.12` | `0.3736` | `0.7500 / 7` | `25.2394` | `0.0731` | `31.20` | evaluated |
| `command_hold_eval_manifold_staged_disturb_release` | `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json` | `logs/r2_amp/Jun25_0/Jun25_05-00-11_command_hold_eval_manifold_staged_disturb_release` | `268`, `277`, `280` | `-2183.01` | `2.00` | `0.0000` | `0.0000 / 0` | `-1621.1836` | `0.9999` | `6.45` | evaluated; collapsed |

### Aggregate Evaluation

Higher `avg task return` is better. Lower `avg fall rate` is better.

| experiment | checkpoint | avg task return | avg fall rate | avg length steps | survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | torque L2 | action-rate L2 | dof-acc L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `command_hold_eval_manifold_conservative_disturb_release` | `best` | -8.50 | 0.888 | 159.0 | 3.18 | 0.672 | 0.874 | 0.0050 | -0.741 | 1.626 | 25276 | 3.559 | 391820 |
| `command_hold_eval_manifold_conservative_disturb_release` | `8000` | -45.43 | 0.143 | 451.5 | 9.03 | 0.314 | 0.405 | 0.0043 | -0.758 | 1.654 | 32340 | 2.614 | 231309 |
| `command_hold_eval_manifold_conservative_disturb_release` | `best_corrected` | 1.94 | 0.886 | 155.9 | 3.12 | 0.578 | 0.814 | 0.0049 | -0.747 | 1.632 | 25717 | 3.611 | 384494 |
| `command_hold_eval_manifold_conservative_disturb_release` | `8000_corrected` | 30.02 | 0.167 | 443.0 | 8.86 | 0.317 | 0.413 | 0.0043 | -0.760 | 1.657 | 32088 | 2.634 | 240126 |
| `command_hold_eval_manifold_staged_disturb_release` | `best` | -6.75 | 0.737 | 203.9 | 4.08 | 0.605 | 0.859 | 0.0035 | -0.823 | 1.587 | 12236 | 1.387 | 245486 |
| `command_hold_eval_manifold_staged_disturb_release` | `8000` | -993.79 | 1.000 | 3.1 | 0.06 | 1.761 | 3.201 | 0.0065 | -0.159 | 1.059 | 224595 | 3482843 | 7026349 |

### Preset-Level Failure Notes

| experiment | checkpoint | worst-return preset | worst fall preset |
|---|---:|---|---|
| `command_hold_eval_manifold_conservative_disturb_release` | `best` | `run`, task `-11.07`, fall `1.000`, survival `1.31s` | `stand`, `run`, `jump`, `turn_left`, and `strafe_right`, fall `1.000` |
| `command_hold_eval_manifold_conservative_disturb_release` | `8000` | `jump`, task `-49.98`, fall `0.406`, survival `7.61s` | `jump`, fall `0.406`; `stand`, fall `0.281` |
| `command_hold_eval_manifold_conservative_disturb_release` | `best_corrected` | `jump`, task `-1.87`, fall `1.000`, survival `2.20s` | `stand`, `walk_slow`, `jump`, `turn_left`, and `strafe_right`, fall `1.000` |
| `command_hold_eval_manifold_conservative_disturb_release` | `8000_corrected` | `jump`, task `20.86`, fall `0.391`, survival `7.99s` | `jump`, fall `0.391`; `stand`, fall `0.219`; `run`, fall `0.203` |
| `command_hold_eval_manifold_staged_disturb_release` | `best` | `strafe_right`, task `-9.08`, fall `1.000`, survival `3.00s` | `stand`, `walk_slow`, `jump`, `turn_left`, and `strafe_right`, fall `1.000` |
| `command_hold_eval_manifold_staged_disturb_release` | `8000` | `walk_fast`, task `-1089.95`, fall `1.000`, survival `0.04s` | all seven presets, fall `1.000` |

### Supported Conclusion

The corrected Jun25_0 results change the conservative-run interpretation: `model_8000.pt` is the only useful continuation candidate in this batch, but it is not a final policy because `jump` still has high fall rate. The staged rerun remains a failure.

Facts:

- The conservative run reached the intended disturbance cap (`staged_disturb_level=0.7500`) and had a healthy-looking training tail (`task reward=21.04`, `episode length=932.12`, staged-window fall rate `0.0731`). Under the corrected evaluator, `model_8000.pt` has avg task return `30.02`, avg fall rate `0.167`, and survival `8.86s`; the old `-45.43` task return was a reward-semantics artifact.
- `model_best_task.pt` remains unusable after correction: avg task return is only `1.94`, avg fall rate is `0.886`, and five presets still have `fall_rate=1.000`.
- `model_8000.pt` is still not solved. Its worst preset is `jump`, with task `20.86`, fall `0.391`, and survival `7.99s`; `stand` and `run` also remain above a `0.20` fall rate in this corrected 64-episode run.
- The staged rerun is a clear failure. It briefly reached stage 1 around training iteration `604`, regressed back to stage 0 around iteration `642`, and ended with `episode length=2.00`, `task reward=-2183.01`, and fixed-preset `fall_rate=1.000` for all seven presets.
- The first eval-manifold run remains the best evidence that matching the evaluation command manifold helps run/walk survival, but the Jun25_0 rerun shows that all-profile staged monitoring plus adaptive regression is not stable enough from scratch.

Interpretation:

- Keep eval-manifold sampling as a useful diagnostic idea, but do not treat the current staged-gate implementation as solved.
- Treat conservative `8000` as the current best Jun25_0 checkpoint for local diagnostics. It is much stronger than the old table implied, but the remaining failure is robustness, especially `jump`, not a broad negative-return objective collapse.
- Do not continue `model_best_task.pt`; the training best-task checkpoint is not the fixed-preset best policy here.
- The next useful step is not another broader from-scratch staged-release JSON. First run local corrected-policy diagnostics around `jump` stability and no-disturb-to-disturb robustness.

Recommended next work:

1. Use `command_hold_eval_manifold_conservative_disturb_release` `model_8000.pt` as the corrected local diagnostic anchor, not `model_best_task.pt`.
2. Completed on 2026-06-29: corrected `jump` reward-term diagnostic separated termination, gait/clearance, smoothness, and tracking contributions under the corrected task-return semantics.
3. Completed on 2026-06-29: focused robustness sweep from `model_8000.pt` with `--eval_disturb_ratio` on `jump` and `run`.
4. If another training run is needed, warm-start from conservative `8000` only after visual/play checks; do not start another all-profile staged run from scratch with the current gate.

### Follow-up Diagnostic Implementation - 2026-06-29

Code change:

```text
legged_gym/envs/r2/r2.py
legged_gym/scripts/evaluate.py
legged_gym/utils/helpers.py
tests/test_amp_training_contracts.py
```

`--record_reward_terms` is now a default-off evaluation flag. When enabled, `evaluate.py` sets `env.record_reward_terms=True`; `R2Robot.compute_reward()` then caches the current step's scaled reward terms in `last_reward_terms` before reset handling can clear `episode_sums`. The evaluator accumulates those terms per completed episode and writes `reward_terms.csv` plus `reward_terms.json` next to the normal `metrics.csv/json`.

The default fixed-preset output schema is unchanged unless the flag is passed.

Recommended first diagnostic command:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /mnt/e/codebase/VR_Teleoperation -- env PATH=/opt/miniconda3/envs/r2gym/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONPATH=/mnt/e/codebase/VR_Teleoperation:/mnt/e/codebase/VR_Teleoperation/rsl_rl LD_LIBRARY_PATH=/opt/miniconda3/envs/r2gym/lib:/mnt/e/wsl/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64 /opt/miniconda3/envs/r2gym/bin/python legged_gym/scripts/evaluate.py --task=r2amp --headless --sim_device=cpu --rl_device=cpu --num_envs=64 --load_run Jun25_0/Jun25_04-43-45_command_hold_eval_manifold_conservative_disturb_release --checkpoint=8000 --cfg_override_json configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json --num_episodes=64 --episode_seconds=10 --preset stand --preset jump --preset run --record_reward_terms --output_dir outputs/eval/June29_Jun25_0_conservative_8000_reward_terms
```

Expected additional outputs:

```text
outputs/eval/June29_Jun25_0_conservative_8000_reward_terms/reward_terms.csv
outputs/eval/June29_Jun25_0_conservative_8000_reward_terms/reward_terms.json
```

### Local Reward-Semantics Diagnostic - 2026-06-29

Hypothesis: the conservative `8000` checkpoint's negative fixed-preset task return is an evaluation reward-semantics artifact: default `evaluate.py` disabled `env.use_disturb`, which also disabled `R2InterruptRobot`'s training-time reward masking for interrupt arm joints.

Local diagnostic outputs:

```text
outputs/eval/June29_Jun25_0_conservative_8000_reward_terms
outputs/eval/June29_Jun25_0_conservative_8000_joint_limit_probe
outputs/eval/June29_Jun25_0_conservative_8000_reward_terms_evaldisturb0
```

Facts:

| protocol | preset | task return | fall rate | survival s | dof_pos_limits | dof_vel_limits | tracking lin | tracking yaw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old default no-disturb semantics | `stand` | -45.14 | 0.281 | 8.31 | -63.73 | -9.21 | 13.04 | 20.84 |
| old default no-disturb semantics | `jump` | -51.47 | 0.453 | 7.65 | -62.12 | -8.77 | 12.71 | 19.26 |
| old default no-disturb semantics | `run` | -46.03 | 0.047 | 9.58 | -63.22 | -12.94 | 15.97 | 21.99 |
| `--eval_disturb_ratio=0.0` | `stand` | 33.27 | 0.141 | 9.00 | -0.19 | -0.06 | 14.71 | 23.24 |
| `--eval_disturb_ratio=0.0` | `jump` | 21.78 | 0.328 | 8.18 | -0.20 | -0.15 | 13.38 | 20.55 |
| `--eval_disturb_ratio=0.0` | `run` | 25.47 | 0.172 | 8.41 | -0.28 | -0.14 | 14.09 | 19.16 |

The joint-limit probe shows the old default `dof_pos_limits` penalty came almost entirely from interrupt arm joints:

| preset | top soft-limit contributors |
|---|---|
| `stand` | `right_shoulder_pitch_joint` share `0.363`, `right_arm_yaw_joint` `0.257`, `right_shoulder_roll_joint` `0.154`, `left_shoulder_roll_joint` `0.125`, `left_arm_pitch_joint` `0.097` |
| `jump` | `right_shoulder_pitch_joint` share `0.325`, `right_arm_yaw_joint` `0.236`, `left_arm_pitch_joint` `0.162`, `right_shoulder_roll_joint` `0.143`, `left_shoulder_roll_joint` `0.131` |
| `run` | `right_shoulder_pitch_joint` share `0.405`, `right_arm_yaw_joint` `0.289`, `right_shoulder_roll_joint` `0.174`, `left_arm_pitch_joint` `0.074`, `left_shoulder_roll_joint` `0.055` |

Code conclusion:

- `evaluate.py` default no-disturb evaluation now clears `disturb_masks` / `interrupt_mask` and sets disturbance strength to zero, but no longer flips `env.use_disturb=False`.
- This keeps rollout disturbance-free while preserving `R2InterruptRobot`'s training reward contract for interrupt arm joints.
- The corrected evaluator writes interrupt buffers under `torch.inference_mode()` because `env.reset()` runs in inference mode and may make `disturb_actions` an inference tensor.
- Completed next step: full seven-preset conservative `8000` and `best` evaluations were rerun in `outputs/eval/June29_Jun25_0_eval_manifold_conservative_8000_corrected` and `outputs/eval/June29_Jun25_0_eval_manifold_conservative_best_corrected`; the aggregate tables above now use those rows for the corrected interpretation.

### Jump Reward-Term and Disturbance Sweep - 2026-06-29

Hypothesis: the conservative `model_8000.pt` checkpoint is mostly command-capable under corrected no-disturb evaluation, but its remaining `jump` weakness is a stability / jump-shape problem that should become worse under fixed disturbance ratios.

Local diagnostic outputs:

```text
outputs/eval/June29_Jun25_0_conservative_8000_jump_reward_terms_corrected
outputs/eval/June29_Jun25_0_conservative_8000_disturb_sweep
outputs/eval/June29_Jun25_0_conservative_8000_disturb_sweep/summary_metrics.csv
```

The `jump_reward_terms_corrected` directory contains `metrics.csv`, `metrics.json`, `reward_terms.csv`, and `reward_terms.json`. Each disturbance-sweep subdirectory contains `metrics.csv`, `metrics.json`, and the preserved `eval.log`.

Reward-term facts for corrected no-disturb `jump`, 64 episodes:

| metric | value |
|---|---:|
| task return | 15.75 |
| fall rate | 0.578 |
| survival s | 6.50 |
| lin rmse | 0.305 |
| yaw rmse | 0.429 |
| style reward | 0.00540 |
| policy logit | -0.653 |
| disc gap | 1.534 |
| torque L2 | 31322 |
| action-rate L2 | 3.284 |
| dof-acc L2 | 240472 |

Largest negative `jump` reward terms by mean episode return:

| reward term | return contribution |
|---|---:|
| `hopping_symmetry` | -2.379 |
| `termination` | -2.312 |
| `feet_clearance_cmd_linear` | -2.167 |
| `torques` | -1.001 |
| `tracking_contacts_shaped_vel` | -0.730 |
| `stand_still` | -0.711 |
| `waist_control` | -0.698 |
| `ang_vel_xy` | -0.569 |
| `base_height` | -0.552 |

Largest positive `jump` reward terms:

| reward term | return contribution |
|---|---:|
| `tracking_ang_vel` | 16.073 |
| `tracking_lin_vel` | 10.834 |
| `alive` | 1.300 |

Focused disturbance sweep from conservative `8000`, 64 episodes per row:

| preset | disturb ratio | task return | fall rate | survival s | lin rmse | yaw rmse | style reward | disc gap | torque L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 0.00 | 20.03 | 0.406 | 7.42 | 0.279 | 0.396 | 0.00558 | 1.535 | 31025 |
| `jump` | 0.25 | 17.74 | 0.375 | 7.61 | 0.389 | 0.446 | 0.00527 | 1.562 | 30203 |
| `jump` | 0.50 | 13.77 | 0.469 | 7.22 | 0.440 | 0.502 | 0.00511 | 1.569 | 29662 |
| `jump` | 0.75 | 13.99 | 0.484 | 6.94 | 0.448 | 0.530 | 0.00499 | 1.573 | 28571 |
| `jump` | 1.00 | -8.39 | 1.000 | 2.49 | 0.722 | 1.221 | 0.00485 | 1.585 | 26665 |
| `run` | 0.00 | 20.23 | 0.234 | 7.79 | 0.602 | 0.682 | 0.00313 | 1.718 | 38004 |
| `run` | 0.25 | 23.04 | 0.172 | 8.67 | 0.701 | 0.643 | 0.00334 | 1.715 | 37445 |
| `run` | 0.50 | 18.39 | 0.266 | 7.94 | 0.825 | 0.749 | 0.00301 | 1.725 | 35978 |
| `run` | 0.75 | 22.65 | 0.188 | 8.41 | 0.735 | 0.605 | 0.00332 | 1.712 | 36515 |
| `run` | 1.00 | -2.11 | 0.750 | 3.50 | 1.487 | 1.425 | 0.00280 | 1.719 | 30393 |

Facts:

- `jump` remains weak even without applied disturbance in this 64-episode diagnostic: fall rate is `0.578` in the reward-term run and `0.406` in the separate sweep baseline. The two runs differ by random rollout sampling, but both identify `jump` as the weakest preset.
- `jump` failure is not primarily a failure to receive tracking reward. `tracking_ang_vel` and `tracking_lin_vel` are the largest positive terms, while the main negative terms are `hopping_symmetry`, `termination`, `feet_clearance_cmd_linear`, torque, contact velocity shaping, and posture/control penalties.
- Fixed disturbance confirms a threshold effect. `jump` degrades gradually through ratios `0.25` to `0.75`, then collapses at `1.00` with `fall_rate=1.000` and survival `2.49s`.
- `run` is more tolerant than `jump` through ratio `0.75`, but also collapses under full disturbance: fall rate rises to `0.750`, survival drops to `3.50s`, and lin/yaw RMSE roughly doubles.
- AMP discriminator metrics do not explain the collapse by themselves. `disc_gap` changes only mildly across the sweep; the sharper signal is task robustness and termination under disturbance.

Interpretation:

- Do not start another broad from-scratch staged-release run yet. The current evidence says the model needs a profile-specific stability fix, especially for jump shape/termination, before full disturbance pressure is meaningful.
- The next code/config change should be smaller than a new full training recipe: either add a jump-focused evaluation/play diagnostic with visual rollout capture, or warm-start conservative `8000` with a narrow jump-stability objective and disturbance capped below `1.0`.
- A defensible warm-start candidate would keep eval-manifold sampling, cap disturbance at `0.75` or lower initially, and specifically reduce `jump` termination / clearance / hopping-symmetry failures before attempting full disturbance again.

### Termination-Reason Diagnostic - 2026-06-29

Hypothesis: the `jump` and full-disturb failures should be traceable to concrete termination buffers, not only aggregate fall rate.

Code change:

```text
legged_gym/scripts/evaluate.py
legged_gym/utils/helpers.py
tests/test_amp_training_contracts.py
```

`evaluate.py` now supports the default-off flag `--record_termination_reasons`. When enabled, completed episodes are classified as `timeout`, `contact`, `orientation`, `base_height`, or `unknown`; contact terminations also record the contact body in `termination_detail`. The evaluator writes `termination_reasons.csv` and `termination_reasons.json` next to the normal metrics. This is a diagnostic export only and does not change the default `metrics.csv/json` schema.

Local diagnostic outputs:

```text
outputs/eval/June29_Jun25_0_conservative_8000_termination_reasons_corrected
outputs/eval/June29_Jun25_0_conservative_8000_termination_reasons_disturb100
```

Both runs used `model_8000.pt`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`, and presets `jump` plus `run`. The `disturb100` run additionally used `--eval_disturb_ratio=1.0`.

Termination facts:

| protocol | preset | task return | fall rate | survival s | termination reason | detail | count | rate | mean survival s |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| corrected no-disturb | `jump` | 15.75 | 0.578 | 6.50 | contact | `base_link` | 37 | 0.578 | 3.94 |
| corrected no-disturb | `jump` | 15.75 | 0.578 | 6.50 | timeout | - | 27 | 0.422 | 10.02 |
| corrected no-disturb | `run` | 24.28 | 0.094 | 9.14 | contact | `base_link` | 6 | 0.094 | 0.59 |
| corrected no-disturb | `run` | 24.28 | 0.094 | 9.14 | timeout | - | 58 | 0.906 | 10.02 |
| full disturb ratio 1.0 | `jump` | -8.39 | 1.000 | 2.49 | contact | `base_link` | 51 | 0.797 | 2.95 |
| full disturb ratio 1.0 | `jump` | -8.39 | 1.000 | 2.49 | orientation | `roll_pitch` | 13 | 0.203 | 0.70 |
| full disturb ratio 1.0 | `run` | -11.74 | 1.000 | 1.51 | contact | `base_link` | 42 | 0.656 | 1.66 |
| full disturb ratio 1.0 | `run` | -11.74 | 1.000 | 1.51 | orientation | `roll_pitch` | 22 | 0.344 | 1.23 |

Facts:

- Corrected no-disturb `jump` failures are entirely `base_link` contact terminations in this 64-episode run; the successful episodes time out at the 10s horizon.
- Corrected no-disturb `run` is mostly stable in this run (`timeout` rate `0.906`), but the small failure set is also `base_link` contact.
- Under full disturbance, both `jump` and `run` fail every episode. `base_link` contact remains the dominant failure mode, with secondary roll/pitch orientation failure.
- This supports the reward-term diagnosis: `jump` does not primarily need more tracking reward; it needs base-contact / body-stability improvement before full disturbance pressure is meaningful.

Interpretation:

- The next local step should be a targeted visual/play check or a headless state trace around base height, roll/pitch, and contact timing for `jump`; changing AMP weight or discriminator capacity is not the first lever suggested by these data.
- If starting a warm-start training follow-up, constrain it to conservative `8000`, keep disturbance below full ratio initially, and target base-link contact reduction plus roll/pitch stability for `jump` before attempting full-disturb release.

### Pre-Reset State-Trace Diagnostic - 2026-06-30

Hypothesis: aggregate termination reasons identify `base_link` contact and roll/pitch failure, but the next decision needs to know whether the terminal contact is a sudden reset-step event or a multi-step loss of height/attitude before termination.

Code change:

```text
legged_gym/envs/r2/r2.py
legged_gym/scripts/evaluate.py
legged_gym/utils/helpers.py
tests/test_amp_training_contracts.py
```

`evaluate.py` now supports the default-off flag `--record_state_trace`. When enabled, `R2Robot.post_physics_step()` caches a pre-reset terminal snapshot before `reset_idx()`, because `evaluate.py` reads diagnostics after `env.step()` returns and done environments have already been reset. The evaluator then writes `state_trace.csv` and `state_trace.json` with a fixed tail window per completed episode. This export records base height, roll/pitch/yaw, base linear/yaw velocity, command tracking error, max termination contact force/body, and `steps_until_done`; it does not change default `metrics.csv/json` behavior.

Local diagnostic outputs:

```text
outputs/eval/June30_Jun25_0_conservative_8000_state_trace_corrected
outputs/eval/June30_Jun25_0_conservative_8000_state_trace_disturb100
outputs/eval/June30_Jun25_0_conservative_8000_state_trace_summary.csv
```

Both runs used `model_8000.pt`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`, presets `jump` plus `run`, `--record_termination_reasons`, `--record_state_trace`, and `--state_trace_window_steps=50`. The `disturb100` run additionally used `--eval_disturb_ratio=1.0`. An earlier post-step trace attempt was discarded because terminal rows showed reset-state `base_z=0.800`; the committed diagnostic uses the pre-reset snapshot instead.

State-trace facts:

| protocol | preset | reason | detail | n | final z | min z | max abs roll | max abs pitch | final lin err | final yaw err | max contact | contact lead mean | contact lead max |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected no-disturb | `jump` | contact | `base_link` | 37 | 0.722 | 0.694 | 0.171 | 0.185 | 0.637 | 0.872 | 451.8 | 0.0 | 0 |
| corrected no-disturb | `jump` | timeout | - | 27 | 0.720 | 0.698 | 0.024 | 0.041 | 0.651 | 0.379 | 0.0 | - | - |
| corrected no-disturb | `run` | contact | `base_link` | 6 | 0.786 | 0.763 | 0.143 | 0.405 | 1.651 | 1.086 | 839.2 | 0.0 | 0 |
| corrected no-disturb | `run` | timeout | - | 58 | 0.759 | 0.751 | 0.032 | 0.057 | 0.808 | 0.605 | 0.0 | - | - |
| full disturb ratio 1.0 | `jump` | contact | `base_link` | 51 | 0.565 | 0.564 | 0.735 | 0.750 | 1.687 | 1.666 | 1302.8 | 24.0 | 49 |
| full disturb ratio 1.0 | `jump` | orientation | `roll_pitch` | 13 | 0.663 | 0.659 | 0.323 | 0.951 | 1.409 | 2.885 | 218.9 | 19.2 | 29 |
| full disturb ratio 1.0 | `run` | contact | `base_link` | 42 | 0.487 | 0.479 | 0.541 | 0.949 | 1.605 | 1.651 | 1887.8 | 19.5 | 49 |
| full disturb ratio 1.0 | `run` | orientation | `roll_pitch` | 22 | 0.511 | 0.501 | 0.629 | 0.978 | 2.379 | 1.713 | 1550.1 | 22.7 | 45 |

Metrics for the same traced runs:

| protocol | preset | task return | fall rate | survival s | lin rmse | yaw rmse |
|---|---|---:|---:|---:|---:|---:|
| corrected no-disturb | `jump` | 15.75 | 0.578 | 6.50 | 0.305 | 0.429 |
| corrected no-disturb | `run` | 24.28 | 0.094 | 9.14 | 0.441 | 0.582 |
| full disturb ratio 1.0 | `jump` | -8.39 | 1.000 | 2.49 | 0.722 | 1.221 |
| full disturb ratio 1.0 | `run` | -11.74 | 1.000 | 1.51 | 1.323 | 1.506 |

Facts:

- Corrected no-disturb `jump` contact failures do not show a long pre-terminal contact build-up in the 50-step tail window: `contact_lead_mean=0.0`, and roll/pitch remain modest. The terminal base height is lower than the timeout set but not a progressive full-body collapse.
- Corrected no-disturb `run` mostly times out; its small failure set has higher final speed/yaw error and one larger pitch excursion, but also no multi-step contact lead.
- Full disturbance changes the failure shape. Contact appears roughly 19-24 steps before termination on average, base height drops much lower, and roll/pitch approaches the same thresholds used by `check_termination()`.
- The dominant full-disturb failure is not an AMP discriminator issue in this diagnostic; it is an applied-disturbance robustness problem that manifests as base-link contact plus roll/pitch loss.

Interpretation:

- The next training change should not be another broad from-scratch staged recipe. The local evidence supports a warm-start, narrow robustness run from conservative `8000`.
- The warm-start should initially cap disturbance below full ratio, keep eval-manifold profile sampling, and focus on base height / roll-pitch stability under `jump` and `run` before reopening full disturbance.
- For no-disturb `jump`, the sudden `base_link` contact suggests checking jump clearance/body-height/contact-threshold behavior with visual play or a small jump-focused reward/config probe before changing AMP style weight or discriminator capacity.

### Short Play Diagnostic - 2026-06-30

Hypothesis: before any warm-start training, the conservative `8000` checkpoint should at least be replayable through the `walk -> jump -> run` demo sequence, and the local machine should report whether true visual MP4 capture is available.

Code change:

```text
legged_gym/scripts/play.py
legged_gym/utils/helpers.py
tests/test_amp_training_contracts.py
```

`play.py` now supports default-off diagnostic parameters `--play_seconds` and `--record_seconds`. If omitted, it keeps the legacy long-running viewer behavior; if supplied, it exits after the requested demo duration and limits the recording window. This makes local play checks reproducible without relying on an external timeout to kill the process.

Local diagnostic outputs:

```text
outputs/eval/June30_Jun25_0_conservative_8000_play_diagnostic/headless_play.log
outputs/eval/June30_Jun25_0_conservative_8000_play_diagnostic/xvfb_record_attempt.log
```

Commands used the conservative `model_8000.pt` checkpoint:

```text
--load_run Jun25_0/Jun25_04-43-45_command_hold_eval_manifold_conservative_disturb_release
--checkpoint 8000
--cfg_override_json configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json
--play_seconds 10.5
```

Facts:

- True MP4 capture is not available on this local WSL/Xvfb path. The `xvfb_record_attempt.log` run reaches Isaac Gym initialization with PhysX CPU and disabled GPU pipeline, then exits with `Segmentation fault (core dumped)` before policy loading. No `.mp4` was produced.
- Headless play succeeds with the same checkpoint and finite duration. It loads `model_8000.pt`, reports `viewer unavailable, skip mp4 recording`, and executes the sequence `walk -> jump -> run`.
- The deterministic reset check in `headless_play.log` reports `base_z=0.8000m`, `left_foot_z=0.0530m`, and `right_foot_z=0.0530m`.
- The jump segment is reached: at step `400`, phase is `jump`, command is `[0. 0. 0.]`, base linear velocity is approximately `[0.0357, 0.0226, -0.0651]`, and action norm is `25.869`.

Interpretation:

- This local machine cannot currently provide the requested visual MP4 evidence through Isaac Gym viewer recording; this is a graphics/runtime limitation, not evidence that the policy failed to load.
- The finite headless play check does prove that conservative `8000` can be loaded and driven through the demo command sequence, including the jump phase. It is weaker than visual inspection and should not replace quantitative `evaluate.py` metrics.
- If a real viewer is needed before training, run the same `play.py --play_seconds 10.5 --record_seconds 10.5` command on a machine with a working Isaac Gym viewer stack. Locally, the next useful non-training step remains headless reward/state diagnostics or a small warm-start config design, not more attempts to record through Xvfb.

### Checkpoint Evaluation Coverage Audit - 2026-06-30

Hypothesis: before starting another evaluation or training branch, verify from the filesystem that there are no recent model-bearing R2 AMP runs silently missing evaluation.

Local audit outputs:

```text
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/summary.json
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/checkpoint_eval_coverage.csv
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/checkpoint_eval_coverage.json
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/transient_log_dirs.csv
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/top_level_model_artifacts.csv
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/documented_eval_output_paths.csv
outputs/eval/June30_r2_amp_checkpoint_eval_coverage/documented_eval_output_paths_summary.json
```

Coverage summary:

| item | count | interpretation |
|---|---:|---|
| model-bearing `logs/r2_amp` run directories | 25 | Direct checkpoint directories found under the local E-checkout. |
| covered current Jun19-Jun25 runs | 16 | Current main ablation line has fixed-preset or focused diagnostic evidence in `outputs/eval`. |
| legacy Jun17 runs with combined eval | 3 | Evaluated through older `June17_manual_eval` / `June17_multi_expert_eval` combined outputs. |
| legacy runs needing manual checkpoint normalization | 6 | Pre-Jun17/Jun15/Jun10/Apr17 archive runs use non-current checkpoint naming or older task assumptions. |
| transient Jun29-Jun30 log dirs without checkpoints | 44 | These contain only `train.log`; they are local failed/smoke invocations and have no `.pt` to evaluate. |
| missing evidence paths for current mapped runs | 0 | Every mapped current evidence path exists on disk. |

Facts:

- The recent top-level `Jun29_*` and `Jun30_*` directories are not un-evaluated trained policies: each inspected directory contains `train.log` only and no `.pt` checkpoint.
- The current Jun19-Jun25 model-bearing experiment line is covered through the documented fixed-preset outputs and the later focused diagnostics. This includes Jun25_0 conservative/staged best and final checkpoints.
- The only remaining model-bearing directories outside that covered line are archival runs: `Apr17_15-18-11_r2v2_amp_version4`, `eval_style0_jun10_30000`, `Jun10`, `Jun10/sw1`, `Jun15/sw05`, and `Jun15/sw1`.
- Those archive runs are not ready for the current `evaluate.py --checkpoint` path as-is because several checkpoint filenames are non-standard for `get_load_path()`, for example `model_best_task(3).pt`, `model_30000(6).pt`, `mixed_30000.pt`, and `style0_30000.pt`.
- The top-level `logs/r2_amp/model*.pt` files are not additional unevaluated runs: `top_level_model_artifacts.csv` shows seven are byte-identical duplicates of already evaluated Jun24 artifacts, while `logs/r2_amp/model_top_task_1518.pt` is an invalid artifact; current WSL `torch.load(..., map_location="cpu")` fails with `UnpicklingError: invalid load key, '#'`.

Interpretation:

- For the current Jun25_0 decision, there is no hidden recent checkpoint left to evaluate locally. The next model-development decision should be based on conservative `8000` diagnostics, not on unevaluated Jun29/Jun30 transient logs.
- If "all evaluations" is extended to archival pre-Jun17 runs, the next step is a separate compatibility/normalization task: map each non-standard checkpoint filename to an explicit load path or create a temporary normalized copy/symlink, then run the current fixed-preset CPU protocol. That is separate from the Jun25_0 continuation decision.

### Archival Checkpoint Compatibility Evaluation - 2026-06-30

Hypothesis: the remaining pre-Jun17 archival R2 AMP checkpoints should be evaluated only after proving that their saved policy architecture and checkpoint naming are compatible with the current `evaluate.py` loader.

Compatibility facts:

- `logs/r2_amp/Apr17_15-18-11_r2v2_amp_version4/model_best_mixed.pt` is not directly comparable with the current evaluator. Its checkpoint has `std.shape=(24,)` and first actor layer shape `(256, 124)`, while current R2 policy checkpoints use `std.shape=(26,)` and first actor layer shape `(256, 131)`.
- The Jun10 and Jun15 archival checkpoints listed below all use the current 26-action policy shape and can be loaded by the current evaluator after filename normalization.
- `logs/r2_amp/eval_style0_jun10_30000/model_30000.pt` and `logs/r2_amp/Jun10/style0_30000.pt` have the same file hash, so only the normalized `Jun10_style0` evaluation is kept as the formal row.
- Temporary hard-link load directories were created under ignored local output state:

```text
logs/r2_amp/_archive_eval_compat/Jun10_style0
logs/r2_amp/_archive_eval_compat/Jun10_mixed
logs/r2_amp/_archive_eval_compat/Jun10_mixed2
logs/r2_amp/_archive_eval_compat/Jun10_walk
logs/r2_amp/_archive_eval_compat/Jun10_sw1
logs/r2_amp/_archive_eval_compat/Jun15_sw05
logs/r2_amp/_archive_eval_compat/Jun15_sw1
```

Local evaluation outputs:

```text
outputs/eval/June30_archive_Jun10_style0_30000
outputs/eval/June30_archive_Jun10_style0_best
outputs/eval/June30_archive_Jun10_mixed_30000
outputs/eval/June30_archive_Jun10_mixed_best_mixed
outputs/eval/June30_archive_Jun10_mixed2_30000
outputs/eval/June30_archive_Jun10_walk_30000
outputs/eval/June30_archive_Jun10_sw1_30000
outputs/eval/June30_archive_Jun10_sw1_best
outputs/eval/June30_archive_Jun15_sw05_30000
outputs/eval/June30_archive_Jun15_sw05_best
outputs/eval/June30_archive_Jun15_sw1_30000
outputs/eval/June30_archive_Jun15_sw1_best
outputs/eval/June30_archive_eval_summary/archive_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides were matched to the archival intent where available: `style0.json`, `sw1.json`, `sw005.json`, and `motion_walk.json`.

Aggregate evaluation:

| archive eval | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | torque L2 | action-rate L2 | worst preset | worst return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `Jun10_mixed_best_mixed` | 7 | 5.27 | 0.208 | 8.52 | 0.435 | 0.509 | 0.01161 | -0.266 | 0.307 | 22647 | 2.580 | `run` | -1.96 |
| `Jun10_style0_best` | 7 | 5.02 | 0.446 | 6.28 | 0.602 | 0.635 | 0.00000 | -0.384 | 0.424 | 16802 | 3.539 | `run` | -9.73 |
| `Jun10_sw1_best` | 7 | 0.05 | 0.219 | 8.37 | 0.427 | 0.532 | 0.01195 | -0.228 | 0.263 | 29265 | 3.282 | `run` | -9.64 |
| `Jun10_mixed2_30000` | 7 | -0.22 | 0.384 | 6.63 | 0.486 | 0.694 | 0.01125 | -0.292 | 0.337 | 33970 | 91.120 | `jump` | -11.39 |
| `Jun10_walk_30000` | 7 | -3.72 | 0.462 | 6.46 | 0.515 | 0.723 | 0.01109 | -0.294 | 0.328 | 32618 | 21.766 | `stand` | -13.39 |
| `Jun10_mixed_30000` | 7 | -3.72 | 0.462 | 6.46 | 0.515 | 0.723 | 0.01109 | -0.294 | 0.328 | 32618 | 21.766 | `stand` | -13.39 |
| `Jun15_sw05_best` | 7 | -6.47 | 1.000 | 1.09 | 1.102 | 0.418 | 0.00064 | -0.201 | 0.124 | 3178 | 0.005 | `jump` | -10.15 |
| `Jun15_sw1_best` | 7 | -6.50 | 1.000 | 1.09 | 1.095 | 0.427 | 0.01270 | -0.201 | 0.123 | 3161 | 0.005 | `jump` | -10.28 |
| `Jun15_sw05_30000` | 7 | -8.58 | 1.000 | 0.86 | 0.999 | 1.245 | 0.00060 | -0.227 | 0.224 | 53223 | 339.257 | `jump` | -11.09 |
| `Jun15_sw1_30000` | 7 | -9.57 | 1.000 | 1.35 | 0.962 | 1.132 | 0.01266 | -0.185 | 0.174 | 44781 | 6.598 | `jump` | -11.92 |
| `Jun10_style0_30000` | 7 | -12.31 | 1.000 | 0.98 | 0.954 | 1.924 | 0.00000 | -0.386 | 0.433 | 44206 | 283.808 | `stand` | -17.64 |
| `Jun10_sw1_30000` | 7 | -14.00 | 1.000 | 1.03 | 0.893 | 1.823 | 0.01120 | -0.306 | 0.338 | 48254 | 412.549 | `stand` | -19.07 |

Facts:

- Every formal archival output has 7 preset rows in `metrics.csv`.
- The strongest archival row is `Jun10_mixed_best_mixed`, but it is still much weaker than the current conservative Jun25_0 `8000` checkpoint on the corrected fixed-preset protocol. `Jun10_mixed_best_mixed` has avg task return `5.27` and avg fall rate `0.208`; Jun25_0 conservative `8000` corrected eval was `18.91` and `0.170`.
- Jun10 `best` checkpoints retain some usable behavior, especially `mixed_best_mixed`, `style0_best`, and `sw1_best`; the corresponding 30000 checkpoints are substantially worse or fully unstable.
- Jun15 `sw05` and `sw1` are not useful continuation targets under the current protocol. Both final checkpoints have `fall_rate=1.000` across all presets, and the `model_best_task(...)` files also show `fall_rate=1.000`. Those two best files also carry checkpoint `iter=0`, so they are recorded as archive evidence rather than serious trained-policy candidates.
- `Jun10_mixed_30000` and `Jun10_walk_30000` produced identical aggregate metrics in this protocol, which suggests either duplicated policy content or equivalent loaded behavior. They are not competitive with current Jun25_0 evidence.

Interpretation:

- The archival compatibility task is now closed for the model-bearing pre-Jun17 checkpoints that can be loaded by the current 26-action evaluator.
- These archive results do not change the current development decision. The best available evidence still points to continuing from `Jun25_0` conservative `8000`, with a warm-start, narrow robustness run focused on jump/run base-contact and roll/pitch stability.
- The next non-training local diagnostic, if needed before warm-starting, should be a narrow jump/run reward or state-trace probe around base height, roll/pitch, and contact timing; more broad archival evaluation is unlikely to improve the decision.

### Jun25_0 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: before treating conservative `8000` as the best Jun25_0 checkpoint, evaluate the saved `model_top_task_*` files because `model_best_task.pt` covers only one top-k slot and the other top-k files may contain a better fixed-preset policy.

Compatibility setup:

- Temporary hard-link load directories were created under ignored local output state:

```text
logs/r2_amp/_topk_eval_compat/Jun25_0/conservative_top_2637
logs/r2_amp/_topk_eval_compat/Jun25_0/conservative_top_2646
logs/r2_amp/_topk_eval_compat/Jun25_0/conservative_top_2677
logs/r2_amp/_topk_eval_compat/Jun25_0/staged_top_268
logs/r2_amp/_topk_eval_compat/Jun25_0/staged_top_277
logs/r2_amp/_topk_eval_compat/Jun25_0/staged_top_280
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun25_0_conservative_top_task_2637_corrected
outputs/eval/June30_Jun25_0_conservative_top_task_2646_corrected
outputs/eval/June30_Jun25_0_conservative_top_task_2677_corrected
outputs/eval/June30_Jun25_0_staged_top_task_268
outputs/eval/June30_Jun25_0_staged_top_task_277
outputs/eval/June30_Jun25_0_staged_top_task_280
outputs/eval/June30_Jun25_0_top_task_eval_summary/top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Conservative top-k used `configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json`; staged top-k used `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json`.

Aggregate comparison:

| eval | checkpoint | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `conservative_8000_corrected` | `8000` | 7 | 30.02 | 0.167 | 8.86 | 0.317 | 0.413 | 0.00425 | -0.760 | 1.657 | `jump` | 20.86 | `jump` | 0.391 |
| `conservative_top_2677` | `2677` | 7 | 4.78 | 0.812 | 3.67 | 0.531 | 0.767 | 0.00508 | -0.734 | 1.621 | `turn_left` | -1.66 | `strafe_right` | 1.000 |
| `staged_top_280` | `280` | 7 | 2.89 | 0.728 | 4.14 | 0.549 | 0.858 | 0.00360 | -0.821 | 1.590 | `jump` | -3.71 | `turn_left` | 1.000 |
| `staged_top_277` | `277` | 7 | 2.80 | 0.743 | 3.95 | 0.549 | 0.869 | 0.00358 | -0.822 | 1.586 | `jump` | -3.78 | `strafe_right` | 1.000 |
| `conservative_top_2637` | `2637` | 7 | 1.94 | 0.886 | 3.12 | 0.578 | 0.814 | 0.00491 | -0.747 | 1.632 | `jump` | -1.87 | `strafe_right` | 1.000 |
| `conservative_best_corrected` | `best` | 7 | 1.94 | 0.886 | 3.12 | 0.578 | 0.814 | 0.00491 | -0.747 | 1.632 | `jump` | -1.87 | `strafe_right` | 1.000 |
| `staged_top_268` | `268` | 7 | 1.83 | 0.739 | 3.72 | 0.567 | 0.904 | 0.00361 | -0.817 | 1.579 | `jump` | -3.81 | `strafe_right` | 1.000 |
| `conservative_top_2646` | `2646` | 7 | 1.00 | 0.915 | 2.80 | 0.610 | 0.827 | 0.00487 | -0.746 | 1.629 | `run` | -6.69 | `jump` | 1.000 |
| `staged_best` | `best` | 7 | -6.75 | 0.737 | 4.08 | 0.605 | 0.859 | 0.00354 | -0.823 | 1.587 | `strafe_right` | -9.08 | `strafe_right` | 1.000 |
| `staged_8000` | `8000` | 7 | -993.79 | 1.000 | 0.06 | 1.761 | 3.201 | 0.00652 | -0.159 | 1.059 | `walk_fast` | -1089.95 | `jump` | 1.000 |

Facts:

- Every Jun25_0 top-k output has 7 preset rows in `metrics.csv`.
- Conservative `model_top_task_2637.pt` and the corrected `model_best_task.pt` produce identical fixed-preset metrics, even though their checkpoint file hashes differ. The likely policy weights are behaviorally equivalent under this protocol.
- The strongest top-k row is conservative `model_top_task_2677.pt`, but it is far below conservative `model_8000.pt`: avg task return `4.78` vs `30.02`, avg fall rate `0.812` vs `0.167`.
- Staged top-k checkpoints are better than the staged `8000` collapse, but all remain weak: avg task return `1.83` to `2.89`, and at least one preset has `fall_rate=1.000` in each row.
- A secondary filesystem audit found root-level `logs/r2_amp/model_*.pt` files. Most are duplicate hashes of `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release/*`; root-level `logs/r2_amp/model_top_task_1518.pt` is not a valid torch checkpoint (`invalid load key, '#'`) and is not an evaluation target.

Interpretation:

- The Jun25_0 top-k gap is now closed. No saved top-k checkpoint improves on conservative `8000`.
- The current checkpoint choice is still conservative `model_8000.pt`.
- The next useful local evaluation should target the proposed warm-start mechanism itself, not more checkpoint fishing: for example a short smoke once the warm-start config exists, then fixed-preset / jump-run disturbance diagnostics on its resulting checkpoint.

### Jun17 Current-Protocol Fixed-Preset Evaluation - 2026-06-30

Hypothesis: the Jun17 archive runs were previously covered only by older combined eval outputs with `num_episodes=8`, so they should be rerun with the current 64-episode fixed-preset protocol before declaring historical evaluation coverage complete.

Existing older outputs:

```text
outputs/eval/June17_manual_eval/combined_metrics.csv
outputs/eval/June17_multi_expert_eval/combined_metrics.csv
```

Those combined outputs contain 42 rows each, but each row uses `num_episodes=8`. They remain useful as old smoke evidence, not as the current main comparison protocol.

Compatibility facts:

- The six evaluated Jun17 checkpoints all load with the current 26-action actor shape: `std.shape=(26,)`, first actor layer `(256, 131)`.
- The `model_best_task.pt` internal iterations are:
  - `sw1_dt_warmup`: `1549`
  - `expert_hard_gate_no_style_warmup`: `2126`
  - `expert_hard_gate_selective_walk`: `4000`
- The final checkpoint for `expert_hard_gate_selective_walk` is stored as `model_30000.pt` but has internal `iter=26000`; it is still the final model file present for that run.

Local evaluation outputs:

```text
outputs/eval/June30_Jun17_fixed_sw1_dt_warmup_best
outputs/eval/June30_Jun17_fixed_sw1_dt_warmup_final
outputs/eval/June30_Jun17_fixed_no_style_best
outputs/eval/June30_Jun17_fixed_no_style_final
outputs/eval/June30_Jun17_fixed_selective_walk_best
outputs/eval/June30_Jun17_fixed_selective_walk_final
outputs/eval/June30_Jun17_fixed_eval_summary/jun17_fixed_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/sw1_dt_warmup.json`
  - `configs/ablation/expert_hard_gate_no_style_warmup.json`
  - `configs/ablation/expert_hard_gate_selective_walk.json`

Aggregate evaluation:

| eval | config | checkpoint | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `selective_walk_best` | `expert_hard_gate_selective_walk` | `best` | 7 | 29.79 | 0.036 | 9.78 | 0.311 | 0.453 | 0.00040 | -0.806 | 1.698 | `run` | 20.03 | `stand` | 0.141 |
| `no_style_best` | `expert_hard_gate_no_style_warmup` | `best` | 7 | 24.27 | 0.094 | 9.28 | 0.352 | 0.537 | 0.00556 | -0.666 | 1.539 | `run` | 6.91 | `run` | 0.234 |
| `sw1_dt_warmup_best` | `sw1_dt_warmup` | `best` | 7 | 20.93 | 0.250 | 7.99 | 0.407 | 0.569 | 0.00413 | -0.793 | 1.656 | `run` | -0.50 | `jump` | 0.922 |
| `selective_walk_final` | `expert_hard_gate_selective_walk` | `30000` | 7 | -2.56 | 0.804 | 2.55 | 0.865 | 2.670 | 0.00098 | -0.589 | 1.479 | `jump` | -13.80 | `turn_left` | 1.000 |
| `no_style_final` | `expert_hard_gate_no_style_warmup` | `30000` | 7 | -3.62 | 0.846 | 2.30 | 0.754 | 1.849 | 0.00595 | -0.654 | 1.528 | `turn_left` | -14.25 | `turn_left` | 1.000 |
| `sw1_dt_warmup_final` | `sw1_dt_warmup` | `30000` | 7 | -9.94 | 1.000 | 0.46 | 0.725 | 2.676 | 0.00405 | -0.595 | 1.472 | `jump` | -11.73 | `jump` | 1.000 |

Selected per-preset facts:

- `selective_walk_best` is strong across all seven presets in this no-disturb fixed-preset protocol. Its worst task row is `run` with task return `20.03` and fall rate `0.031`; `jump` has task return `30.82` and fall rate `0.031`.
- `no_style_best` is also usable but weaker on `run`: task return `6.91`, fall rate `0.234`.
- `sw1_dt_warmup_best` is weaker because `jump` has fall rate `0.922` and `run` task return is slightly negative.
- All three final checkpoints show late collapse relative to their best checkpoint. This supports the broader pattern seen in later runs: early/best checkpoints can be useful, while final checkpoints often regress.

Interpretation:

- The Jun17 current-protocol gap for best/final checkpoints is now closed.
- `expert_hard_gate_selective_walk` best is now a strong historical reference: it nearly matches Jun25_0 conservative `8000` on avg task return (`29.79` vs `30.02`) and has lower no-disturb fixed-preset fall rate (`0.036` vs `0.167`).
- It should not automatically replace Jun25_0 conservative `8000` as the warm-start source, because it has not yet received the same corrected disturbance sweep, termination-reason, state-trace, and play diagnostics. It is, however, now the most important historical control to compare against.
- The next non-training evaluation step should be a focused robustness diagnostic for `expert_hard_gate_selective_walk` best using the same jump/run disturbance and termination/state-trace tools already applied to Jun25_0 conservative `8000`.

### Jun17 Selective-Walk Best Robustness Diagnostic - 2026-06-30

Hypothesis: because `expert_hard_gate_selective_walk` best nearly matches Jun25_0 conservative `8000` on no-disturb fixed-preset aggregate metrics, it should receive the same focused jump/run robustness diagnostics before deciding which checkpoint is the better warm-start/control source.

Local diagnostic outputs:

```text
outputs/eval/June30_Jun17_selective_walk_best_disturb_sweep
outputs/eval/June30_Jun17_selective_walk_best_disturb_sweep/summary_metrics.csv
outputs/eval/June30_Jun17_selective_walk_best_state_trace_corrected
outputs/eval/June30_Jun17_selective_walk_best_state_trace_disturb100
outputs/eval/June30_Jun17_selective_walk_best_state_trace_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`.
- Config: `configs/ablation/expert_hard_gate_selective_walk.json`.
- Disturbance sweep: `jump` and `run`, ratios `0.0`, `0.25`, `0.5`, `0.75`, `1.0`, `64` episodes per row.
- State trace: `jump` and `run`, `64` episodes per row, `--record_termination_reasons`, `--record_state_trace`, `--state_trace_window_steps=50`, once with default corrected no-disturb evaluation and once with `--eval_disturb_ratio=1.0`.

Focused disturbance sweep:

| preset | disturb ratio | task return | fall rate | survival s | lin rmse | yaw rmse | style reward | disc gap | torque L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 0.00 | 31.42 | 0.047 | 9.66 | 0.198 | 0.330 | 0.00000 | 1.672 | 23328 |
| `jump` | 0.25 | 31.48 | 0.031 | 9.87 | 0.214 | 0.337 | 0.00000 | 1.676 | 20540 |
| `jump` | 0.50 | 31.29 | 0.016 | 9.88 | 0.213 | 0.346 | 0.00000 | 1.681 | 16011 |
| `jump` | 0.75 | 30.55 | 0.016 | 9.97 | 0.212 | 0.355 | 0.00000 | 1.702 | 10461 |
| `jump` | 1.00 | -19.55 | 1.000 | 2.56 | 0.983 | 1.323 | 0.00000 | 1.694 | 23373 |
| `run` | 0.00 | 16.10 | 0.047 | 9.59 | 0.633 | 0.573 | 0.00000 | 1.651 | 37565 |
| `run` | 0.25 | 18.01 | 0.031 | 9.77 | 0.624 | 0.575 | 0.00000 | 1.692 | 35957 |
| `run` | 0.50 | 15.95 | 0.016 | 9.89 | 0.650 | 0.570 | 0.00000 | 1.732 | 28904 |
| `run` | 0.75 | 11.57 | 0.016 | 9.89 | 0.705 | 0.660 | 0.00000 | 1.738 | 22405 |
| `run` | 1.00 | -22.61 | 1.000 | 3.33 | 1.666 | 1.932 | 0.00000 | 1.668 | 36020 |

Termination facts:

| protocol | preset | task return | fall rate | survival s | termination reason | detail | count | rate | mean survival s |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| corrected no-disturb | `jump` | 30.48 | 0.078 | 9.54 | contact | `base_link` | 5 | 0.078 | 3.82 |
| corrected no-disturb | `jump` | 30.48 | 0.078 | 9.54 | timeout | - | 59 | 0.922 | 10.02 |
| corrected no-disturb | `run` | 16.28 | 0.063 | 9.44 | contact | `base_link` | 4 | 0.063 | 0.69 |
| corrected no-disturb | `run` | 16.28 | 0.063 | 9.44 | timeout | - | 60 | 0.938 | 10.02 |
| full disturb ratio 1.0 | `jump` | -19.55 | 1.000 | 2.56 | contact | `base_link` | 53 | 0.828 | 2.56 |
| full disturb ratio 1.0 | `jump` | -19.55 | 1.000 | 2.56 | orientation | `roll_pitch` | 11 | 0.172 | 2.57 |
| full disturb ratio 1.0 | `run` | -22.58 | 1.000 | 3.49 | contact | `base_link` | 42 | 0.656 | 3.24 |
| full disturb ratio 1.0 | `run` | -22.58 | 1.000 | 3.49 | orientation | `roll_pitch` | 22 | 0.344 | 3.97 |

State-trace facts:

| protocol | preset | reason | detail | n | final z | min z | max abs roll | max abs pitch | final lin err | final yaw err | max contact | contact lead mean | contact lead max |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected no-disturb | `jump` | contact | `base_link` | 5 | 0.698 | 0.655 | 0.133 | 0.215 | 0.757 | 0.612 | 466.6 | 0.0 | 0 |
| corrected no-disturb | `jump` | timeout | - | 59 | 0.755 | 0.736 | 0.031 | 0.117 | 0.554 | -0.165 | 0.0 | - | - |
| corrected no-disturb | `run` | contact | `base_link` | 4 | 0.642 | 0.635 | 0.141 | 0.390 | 2.308 | -0.392 | 142.5 | 0.0 | 0 |
| corrected no-disturb | `run` | timeout | - | 60 | 0.754 | 0.742 | 0.036 | 0.138 | 0.934 | 0.497 | 0.0 | - | - |
| full disturb ratio 1.0 | `jump` | contact | `base_link` | 53 | 0.393 | 0.393 | 0.775 | 0.625 | 2.431 | -0.905 | 1361.2 | 24.4 | 49 |
| full disturb ratio 1.0 | `jump` | orientation | `roll_pitch` | 11 | 0.470 | 0.470 | 0.763 | 0.436 | 2.255 | -2.250 | 934.7 | 20.0 | 41 |
| full disturb ratio 1.0 | `run` | contact | `base_link` | 42 | 0.423 | 0.421 | 0.785 | 0.793 | 2.913 | -1.960 | 1728.4 | 32.5 | 49 |
| full disturb ratio 1.0 | `run` | orientation | `roll_pitch` | 22 | 0.506 | 0.495 | 0.747 | 0.648 | 2.515 | -0.720 | 1950.8 | 31.0 | 49 |

Facts:

- Compared with Jun25_0 conservative `8000`, `expert_hard_gate_selective_walk` best is much stronger under partial disturbance. At `jump` ratio `0.75`, fall rate is `0.016` versus Jun25_0 conservative `0.484`; at `run` ratio `0.75`, fall rate is `0.016` versus Jun25_0 conservative `0.188`.
- Full disturbance still breaks the policy. Both `jump` and `run` reach `fall_rate=1.000` at ratio `1.0`.
- Under corrected no-disturb trace, failures are rare and mostly immediate `base_link` contact events with `contact_lead_mean=0.0`.
- Under full disturbance, the failure shape matches the Jun25_0 diagnosis: base height drops, contact appears roughly 20-33 steps before termination, and roll/pitch becomes a secondary termination source.
- Style reward is effectively zero in this selective-walk checkpoint for these `jump`/`run` diagnostics, consistent with the config disabling run/jump style contribution while still routing expert metadata.

Interpretation:

- `expert_hard_gate_selective_walk` best is now the strongest robustness reference among evaluated historical checkpoints for no-disturb and partial-disturb `jump`/`run`.
- It still does not solve full disturbance. The remaining full-disturb failure mode is the same broad class as Jun25_0 conservative `8000`: base-link contact plus roll/pitch loss after sustained disturbance.
- The next training decision should compare two possible warm-start sources, not only one: Jun25_0 conservative `8000` has the best current avg task return and existing diagnostic trail, while Jun17 selective-walk best has clearly better partial-disturb robustness and lower no-disturb fall rate.
- A defensible next run would warm-start from the stronger robustness reference or run a small paired warm-start smoke from both sources, with disturbance capped at `0.75` before attempting full ratio `1.0`.

### Jun17 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: after `expert_hard_gate_selective_walk` best emerged as a strong robustness reference, evaluate the saved Jun17 `model_top_task_*` checkpoints to ensure no adjacent early checkpoint is stronger under the current 64-episode fixed-preset protocol.

Compatibility facts:

- Valid top-task checkpoints were hard-linked into ignored local load directories under:

```text
logs/r2_amp/_topk_eval_compat/Jun17
```

- Seven Jun17 top-task files are valid current-shape torch checkpoints:
  - `sw1_dt_warmup`: `model_top_task_1549.pt`, `model_top_task_1679.pt`, `model_top_task_1689.pt`
  - `expert_hard_gate_no_style_warmup`: `model_top_task_2003.pt`, `model_top_task_2126.pt`, `model_top_task_2130.pt`
  - `expert_hard_gate_selective_walk`: `model_top_task_1274.pt`
- `expert_hard_gate_selective_walk/model_top_task_1461.pt` and `model_top_task_1464.pt` are not valid torch checkpoints in this local tree. They raise `UnpicklingError` during `torch.load()` and are not evaluation targets.
- `expert_hard_gate_selective_walk/model_top_task_1274.pt` loads successfully but reports internal `iter=12000`; the evaluation keeps the filename-based checkpoint label `1274` and records this mismatch as an archive artifact issue.

Local evaluation outputs:

```text
outputs/eval/June30_Jun17_top_sw1_dt_warmup_1549
outputs/eval/June30_Jun17_top_sw1_dt_warmup_1679
outputs/eval/June30_Jun17_top_sw1_dt_warmup_1689
outputs/eval/June30_Jun17_top_no_style_2003
outputs/eval/June30_Jun17_top_no_style_2126
outputs/eval/June30_Jun17_top_no_style_2130
outputs/eval/June30_Jun17_top_selective_walk_1274
outputs/eval/June30_Jun17_top_task_eval_summary/jun17_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/sw1_dt_warmup.json`
  - `configs/ablation/expert_hard_gate_no_style_warmup.json`
  - `configs/ablation/expert_hard_gate_selective_walk.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `selective_walk_best` | `expert_hard_gate_selective_walk` | `best` | `best_task` | 7 | 29.79 | 0.036 | 9.78 | 0.311 | 0.453 | 0.00040 | -0.806 | 1.698 | `run` | 20.03 | `stand` | 0.141 |
| `no_style_top_2126` | `expert_hard_gate_no_style_warmup` | `2126` | `top_task` | 7 | 24.27 | 0.094 | 9.28 | 0.352 | 0.537 | 0.00556 | -0.666 | 1.539 | `run` | 6.91 | `run` | 0.234 |
| `no_style_best` | `expert_hard_gate_no_style_warmup` | `best` | `best_task` | 7 | 24.27 | 0.094 | 9.28 | 0.352 | 0.537 | 0.00556 | -0.666 | 1.539 | `run` | 6.91 | `run` | 0.234 |
| `no_style_top_2003` | `expert_hard_gate_no_style_warmup` | `2003` | `top_task` | 7 | 24.22 | 0.085 | 9.34 | 0.366 | 0.533 | 0.00548 | -0.676 | 1.550 | `run` | 7.22 | `run` | 0.203 |
| `no_style_top_2130` | `expert_hard_gate_no_style_warmup` | `2130` | `top_task` | 7 | 23.99 | 0.094 | 9.27 | 0.368 | 0.544 | 0.00542 | -0.676 | 1.542 | `run` | 6.18 | `run` | 0.297 |
| `selective_walk_top_1274` | `expert_hard_gate_selective_walk` | `1274` | `top_task` | 7 | 23.92 | 0.176 | 8.37 | 0.405 | 0.566 | 0.00025 | -0.898 | 1.805 | `run` | 0.54 | `run` | 0.703 |
| `sw1_dt_warmup_top_1679` | `sw1_dt_warmup` | `1679` | `top_task` | 7 | 21.84 | 0.230 | 8.14 | 0.432 | 0.547 | 0.00437 | -0.781 | 1.656 | `run` | -2.22 | `run` | 0.641 |
| `sw1_dt_warmup_best` | `sw1_dt_warmup` | `best` | `best_task` | 7 | 20.93 | 0.250 | 7.99 | 0.407 | 0.569 | 0.00413 | -0.793 | 1.656 | `run` | -0.50 | `jump` | 0.922 |
| `sw1_dt_warmup_top_1549` | `sw1_dt_warmup` | `1549` | `top_task` | 7 | 20.93 | 0.250 | 7.99 | 0.407 | 0.569 | 0.00413 | -0.793 | 1.656 | `run` | -0.50 | `jump` | 0.922 |
| `sw1_dt_warmup_top_1689` | `sw1_dt_warmup` | `1689` | `top_task` | 7 | 20.78 | 0.239 | 8.06 | 0.435 | 0.570 | 0.00408 | -0.796 | 1.661 | `run` | -4.21 | `run` | 0.703 |

Facts:

- Every valid Jun17 top-task output has 7 preset rows in `metrics.csv`.
- No Jun17 top-task checkpoint beats `expert_hard_gate_selective_walk` `model_best_task.pt` under the current no-disturb fixed-preset protocol.
- `expert_hard_gate_no_style_warmup/model_top_task_2126.pt` is behaviorally identical to `model_best_task.pt` under this protocol; `model_top_task_2003.pt` has slightly lower avg fall rate (`0.085` vs `0.094`) but nearly the same task return.
- `sw1_dt_warmup/model_top_task_1679.pt` improves over the `sw1_dt_warmup` best aggregate (`21.84` vs `20.93`) but remains weaker than the no-style and selective-walk best checkpoints, mainly because `run` and `jump` remain weak.
- `selective_walk_top_1274` is much weaker than `selective_walk_best`, especially on `run` (`fall_rate=0.703`).

Interpretation:

- The Jun17 top-task checkpoint gap is now closed for all valid saved top-k files.
- The strongest Jun17 checkpoint remains `expert_hard_gate_selective_walk/model_best_task.pt`, not an adjacent top-task file.
- The invalid top-task files should remain documented as archive corruption / non-checkpoint artifacts, not silently ignored.

### Jun24 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: the Jun24 staged-release runs were previously represented mostly by best/final checkpoints, so the saved `model_top_task_*` files should be evaluated before using those runs as negative evidence.

Compatibility facts:

- Six run-local Jun24 top-task checkpoints load as current-shape torch checkpoints:
  - `logs/r2_amp/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/model_top_task_6996.pt`
  - `logs/r2_amp/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/model_top_task_7752.pt`
  - `logs/r2_amp/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/Jun24_07-02-24_command_hold_run_recovery_staged_disturb_release/model_top_task_7845.pt`
  - `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release/model_top_task_1490.pt`
  - `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release/model_top_task_1517.pt`
  - `logs/r2_amp/Jun24_16-51-59_command_hold_eval_manifold_staged_disturb_release/model_top_task_1518.pt`
- Local compatibility load directories were created under:

```text
logs/r2_amp/_topk_eval_compat/Jun24
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun24_top_run_recovery_6996
outputs/eval/June30_Jun24_top_run_recovery_7752
outputs/eval/June30_Jun24_top_run_recovery_7845
outputs/eval/June30_Jun24_top_eval_manifold_staged_1490
outputs/eval/June30_Jun24_top_eval_manifold_staged_1517
outputs/eval/June30_Jun24_top_eval_manifold_staged_1518
outputs/eval/June30_Jun24_top_task_eval_summary/jun24_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/command_hold_run_recovery_staged_disturb_release.json`
  - `configs/ablation/command_hold_eval_manifold_staged_disturb_release.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| `eval_manifold_staged_top_1518` | `command_hold_eval_manifold_staged_disturb_release` | `1518` | `top_task` | 7 | 17.34 | 0.435 | 6.70 | `jump` | -2.72 | `jump` | 1.000 |
| `eval_manifold_staged_top_1490` | `command_hold_eval_manifold_staged_disturb_release` | `1490` | `top_task` | 7 | 15.16 | 0.491 | 6.09 | `jump` | -2.25 | `jump` | 1.000 |
| `eval_manifold_staged_top_1517` | `command_hold_eval_manifold_staged_disturb_release` | `1517` | `top_task` | 7 | 15.15 | 0.500 | 6.31 | `jump` | -2.34 | `jump` | 1.000 |
| `run_recovery_top_7845` | `command_hold_run_recovery_staged_disturb_release` | `7845` | `top_task` | 7 | -2.56 | 0.938 | 1.41 | `jump` | -4.71 | `strafe_right` | 1.000 |
| `run_recovery_top_6996` | `command_hold_run_recovery_staged_disturb_release` | `6996` | `top_task` | 7 | -2.87 | 0.960 | 1.17 | `jump` | -4.57 | `strafe_right` | 1.000 |
| `run_recovery_top_7752` | `command_hold_run_recovery_staged_disturb_release` | `7752` | `top_task` | 7 | -4.24 | 1.000 | 0.69 | `run` | -5.27 | `jump` | 1.000 |
| `eval_manifold_staged_best` | `command_hold_eval_manifold_staged_disturb_release` | `best` | `best_task` | 7 | -7.20 | 0.496 | 6.45 | `jump` | -10.21 | `jump` | 1.000 |
| `run_recovery_8000` | `command_hold_run_recovery_staged_disturb_release` | `8000` | `final` | 7 | -8.49 | 0.933 | 1.43 | `jump` | -9.95 | `strafe_right` | 1.000 |
| `run_recovery_best` | `command_hold_run_recovery_staged_disturb_release` | `best` | `best_task` | 7 | -8.56 | 0.920 | 1.55 | `run` | -11.54 | `strafe_right` | 1.000 |
| `eval_manifold_staged_8000` | `command_hold_eval_manifold_staged_disturb_release` | `8000` | `final` | 7 | -35.71 | 0.533 | 6.31 | `turn_left` | -50.62 | `jump` | 1.000 |

Selected per-preset facts:

- `eval_manifold_staged_top_1518` is the strongest Jun24 top-task checkpoint in this batch. It is good on `walk_fast` (task return `29.93`, fall rate `0.047`), `strafe_right` (`27.38`, `0.219`), `walk_slow` (`25.66`, `0.266`), and `run` (`20.77`, `0.203`).
- The same `eval_manifold_staged_top_1518` checkpoint still fully fails `jump` (`task_return=-2.72`, `fall_rate=1.000`) and `stand` (`task_return=0.45`, `fall_rate=1.000`), so it is not a robust replacement for the Jun17 selective-walk best checkpoint.
- `run_recovery_top_7845` improves over the Jun24 run-recovery best/final aggregate, but it remains unstable: six of seven presets have `fall_rate=1.000`, while only `run` is partly usable (`task_return=5.68`, `fall_rate=0.562`).

Interpretation:

- The Jun24 top-task gap is now closed for `command_hold_run_recovery_staged_disturb_release` and `command_hold_eval_manifold_staged_disturb_release`.
- `eval_manifold_staged_top_1518` changes the historical read of the Jun24 eval-manifold run: the run did contain a much better intermediate checkpoint than its best/final archive suggested.
- It still does not change the current training recommendation. The best evaluated control remains Jun17 `expert_hard_gate_selective_walk/model_best_task.pt`, because that checkpoint has far lower no-disturb fall rate and already survived partial `jump`/`run` disturbance up to ratio `0.75`.
- The next local no-training step is to continue the same top-task coverage audit backward through Jun23, Jun21, Jun20, and Jun19 before finalizing the historical checkpoint ranking.

### Jun23 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: the Jun23 staged-release runs had saved top-task checkpoints that were not covered by the existing best/final fixed-preset evaluations, so they should be evaluated before closing the historical checkpoint ranking.

Compatibility facts:

- Six Jun23 top-task checkpoints load successfully in the current `r2gym` evaluation environment, and their internal `iter` values match the filename labels:
  - `logs/r2_amp/Jun23/Jun23_03-38-06_command_hold_staged_disturb_release/model_top_task_1315.pt`
  - `logs/r2_amp/Jun23/Jun23_03-38-06_command_hold_staged_disturb_release/model_top_task_1331.pt`
  - `logs/r2_amp/Jun23/Jun23_03-38-06_command_hold_staged_disturb_release/model_top_task_1705.pt`
  - `logs/r2_amp/Jun23/Jun23_14-58-32_command_hold_run_focused_staged_disturb_release/model_top_task_4221.pt`
  - `logs/r2_amp/Jun23/Jun23_14-58-32_command_hold_run_focused_staged_disturb_release/model_top_task_4294.pt`
  - `logs/r2_amp/Jun23/Jun23_14-58-32_command_hold_run_focused_staged_disturb_release/model_top_task_7112.pt`
- Local compatibility load directories were created under:

```text
logs/r2_amp/_topk_eval_compat/Jun23
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun23_top_staged_1315
outputs/eval/June30_Jun23_top_staged_1331
outputs/eval/June30_Jun23_top_staged_1705
outputs/eval/June30_Jun23_top_run_focused_4221
outputs/eval/June30_Jun23_top_run_focused_4294
outputs/eval/June30_Jun23_top_run_focused_7112
outputs/eval/June30_Jun23_top_task_eval_summary/jun23_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/command_hold_staged_disturb_release.json`
  - `configs/ablation/command_hold_run_focused_staged_disturb_release.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `staged_top_1331` | `command_hold_staged_disturb_release` | `1331` | `top_task` | 7 | 15.04 | 0.431 | 6.65 | 0.501 | 0.628 | `run` | -4.86 | `run` | 1.000 |
| `staged_top_1315` | `command_hold_staged_disturb_release` | `1315` | `top_task` | 7 | 12.40 | 0.513 | 5.86 | 0.555 | 0.699 | `run` | -4.96 | `run` | 1.000 |
| `staged_top_1705` | `command_hold_staged_disturb_release` | `1705` | `top_task` | 7 | 7.20 | 0.699 | 4.24 | 0.636 | 0.737 | `run` | -4.75 | `run` | 1.000 |
| `run_focused_top_4221` | `command_hold_run_focused_staged_disturb_release` | `4221` | `top_task` | 7 | -0.81 | 0.888 | 1.92 | 0.687 | 0.968 | `jump` | -4.53 | `jump` | 1.000 |
| `run_focused_top_4294` | `command_hold_run_focused_staged_disturb_release` | `4294` | `top_task` | 7 | -1.80 | 0.915 | 1.67 | 0.720 | 0.976 | `jump` | -4.89 | `jump` | 1.000 |
| `run_focused_top_7112` | `command_hold_run_focused_staged_disturb_release` | `7112` | `top_task` | 7 | -3.49 | 0.953 | 1.24 | 0.791 | 1.083 | `run` | -6.55 | `run` | 1.000 |
| `run_focused_best` | `command_hold_run_focused_staged_disturb_release` | `best` | `best_task` | 7 | -7.91 | 0.978 | 1.17 | 0.856 | 1.170 | `walk_fast` | -8.68 | `jump` | 1.000 |
| `run_focused_8000` | `command_hold_run_focused_staged_disturb_release` | `8000` | `final` | 7 | -8.46 | 1.000 | 0.75 | 0.883 | 1.198 | `run` | -10.10 | `run` | 1.000 |
| `staged_best` | `command_hold_staged_disturb_release` | `best` | `best_task` | 7 | -14.39 | 0.540 | 5.66 | 0.569 | 0.664 | `turn_left` | -23.01 | `run` | 1.000 |
| `staged_8000` | `command_hold_staged_disturb_release` | `8000` | `final` | 7 | -41.51 | 0.219 | 8.23 | 0.405 | 0.488 | `walk_fast` | -50.78 | `run` | 0.656 |

Selected per-preset facts:

- `staged_top_1331` is the strongest Jun23 checkpoint in this batch. It is reasonably good on `strafe_right` (task return `29.66`, fall rate `0.172`), `walk_slow` (`26.46`, `0.156`), and `turn_left` (`20.93`, `0.141`).
- The same `staged_top_1331` checkpoint fully fails `run` (`task_return=-4.86`, `fall_rate=1.000`) and is still weak on `stand` (`task_return=5.85`, `fall_rate=0.828`). This keeps it below the Jun24 `eval_manifold_staged_top_1518` checkpoint and far below the Jun17 selective-walk best reference.
- `run_focused_top_4221` is the best run-focused top-task checkpoint but remains broadly unstable: `jump`, `stand`, `turn_left`, `walk_slow`, and `strafe_right` all have `fall_rate=1.000`; only `run` (`task_return=9.32`, `fall_rate=0.562`) and `walk_fast` (`5.88`, `0.656`) are partly usable.

Interpretation:

- The Jun23 top-task checkpoint gap is now closed for both staged-release runs.
- Jun23 contains useful evidence that early top-task snapshots can be much better than archived best/final checkpoints, especially for `command_hold_staged_disturb_release`; however, the absolute robustness level is still insufficient.
- No Jun23 checkpoint should be used as a warm-start source. The current candidate hierarchy remains Jun17 `expert_hard_gate_selective_walk/model_best_task.pt` first, Jun25_0 conservative `8000` as a diagnostic anchor, and Jun24 `eval_manifold_staged_top_1518` only as historical evidence.
- The next local no-training step is to continue the same top-task coverage audit through Jun21, Jun20, and Jun19.

### Jun21 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: the June21 rerun had poor best/final task-return numbers for `command_hold_conservative_penalty_ramp`, but its saved top-task checkpoints may contain a better intermediate policy that the best/final archive missed.

Compatibility facts:

- Six Jun21 top-task checkpoints load successfully in the current `r2gym` evaluation environment, and their internal `iter` values match the filename labels:
  - `logs/r2_amp/Jun21/Jun21_12-28-33_command_hold_conservative_penalty_ramp/model_top_task_7075.pt`
  - `logs/r2_amp/Jun21/Jun21_12-28-33_command_hold_conservative_penalty_ramp/model_top_task_7654.pt`
  - `logs/r2_amp/Jun21/Jun21_12-28-33_command_hold_conservative_penalty_ramp/model_top_task_7657.pt`
  - `logs/r2_amp/Jun21/Jun21_12-28-55_command_hold_controlled_disturb_release/model_top_task_1450.pt`
  - `logs/r2_amp/Jun21/Jun21_12-28-55_command_hold_controlled_disturb_release/model_top_task_1498.pt`
  - `logs/r2_amp/Jun21/Jun21_12-28-55_command_hold_controlled_disturb_release/model_top_task_1608.pt`
- Local compatibility load directories were created under:

```text
logs/r2_amp/_topk_eval_compat/Jun21
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun21_top_conservative_penalty_ramp_7075
outputs/eval/June30_Jun21_top_conservative_penalty_ramp_7654
outputs/eval/June30_Jun21_top_conservative_penalty_ramp_7657
outputs/eval/June30_Jun21_top_controlled_disturb_1450
outputs/eval/June30_Jun21_top_controlled_disturb_1498
outputs/eval/June30_Jun21_top_controlled_disturb_1608
outputs/eval/June30_Jun21_top_task_eval_summary/jun21_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/command_hold_conservative_penalty_ramp.json`
  - `configs/ablation/command_hold_controlled_disturb_release.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `conservative_penalty_top_7654` | `command_hold_conservative_penalty_ramp` | `7654` | `top_task` | 7 | 29.16 | 0.054 | 9.59 | 0.328 | 0.400 | 0.00548 | -0.687 | 1.575 | `run` | 7.06 | `run` | 0.156 |
| `conservative_penalty_top_7075` | `command_hold_conservative_penalty_ramp` | `7075` | `top_task` | 7 | 28.46 | 0.069 | 9.47 | 0.339 | 0.407 | 0.00551 | -0.684 | 1.574 | `run` | 6.74 | `stand` | 0.250 |
| `conservative_penalty_top_7657` | `command_hold_conservative_penalty_ramp` | `7657` | `top_task` | 7 | 27.21 | 0.107 | 9.17 | 0.362 | 0.430 | 0.00534 | -0.688 | 1.575 | `run` | 3.12 | `run` | 0.297 |
| `controlled_disturb_top_1450` | `command_hold_controlled_disturb_release` | `1450` | `top_task` | 7 | 10.30 | 0.605 | 5.12 | 0.603 | 0.710 | 0.00453 | -0.768 | 1.631 | `walk_fast` | -5.04 | `walk_fast` | 1.000 |
| `controlled_disturb_top_1498` | `command_hold_controlled_disturb_release` | `1498` | `top_task` | 7 | 8.66 | 0.650 | 4.73 | 0.608 | 0.740 | 0.00449 | -0.770 | 1.637 | `walk_fast` | -5.02 | `walk_fast` | 1.000 |
| `controlled_disturb_top_1608` | `command_hold_controlled_disturb_release` | `1608` | `top_task` | 7 | 6.99 | 0.730 | 3.90 | 0.606 | 0.753 | 0.00442 | -0.775 | 1.641 | `run` | -5.05 | `run` | 1.000 |
| `controlled_disturb_best` | `command_hold_controlled_disturb_release` | `best` | `best_task` | 7 | -9.98 | 0.676 | 4.73 | 0.641 | 0.749 | 0.00438 | -0.774 | 1.640 | `jump` | -14.95 | `stand` | 1.000 |
| `conservative_penalty_8000` | `command_hold_conservative_penalty_ramp` | `8000` | `final` | 7 | -33.57 | 0.045 | 9.68 | 0.359 | 0.406 | 0.00541 | -0.692 | 1.577 | `run` | -46.21 | `stand` | 0.188 |
| `controlled_disturb_8000` | `command_hold_controlled_disturb_release` | `8000` | `final` | 7 | -33.83 | 0.498 | 6.19 | 0.634 | 0.614 | 0.00229 | -0.863 | 1.768 | `walk_fast` | -46.18 | `run` | 1.000 |
| `conservative_penalty_best` | `command_hold_conservative_penalty_ramp` | `best` | `best_task` | 7 | -34.71 | 0.051 | 9.68 | 0.340 | 0.399 | 0.00554 | -0.686 | 1.574 | `run` | -54.38 | `run` | 0.109 |

Selected per-preset facts:

- `conservative_penalty_top_7654` is a newly important historical candidate. It is strong on `strafe_right` (task return `40.42`, fall rate `0.000`), `walk_slow` (`38.24`, `0.016`), `turn_left` (`34.58`, `0.016`), `stand` (`33.37`, `0.125`), and `jump` (`31.73`, `0.047`).
- Its main weakness is still `run`: `task_return=7.06`, `fall_rate=0.156`, `survival=8.57s`. This is much weaker than Jun17 selective-walk best on the same no-disturb fixed-preset protocol (`run` task return `20.03`, fall rate `0.031`).
- `command_hold_conservative_penalty_ramp` best/final were misleading for task return. Both kept low fall rates but produced very negative `run` returns (`best=-54.38`, `8000=-46.21`), while top-task `7654` retains low fall rate and much higher task return.
- `controlled_disturb_top_1450` improves over the controlled-disturb best/final task return, but it is not a candidate because `walk_fast` and `run` both have `fall_rate=1.000`.

Interpretation:

- The Jun21 top-task gap is now closed, and it materially changes the historical ranking.
- `conservative_penalty_top_7654` is now the second-best no-disturb fixed-preset historical checkpoint by aggregate task return among the evaluated top-task batch, close to Jun17 selective-walk best (`29.16` vs `29.79`) but still weaker on the critical `run` preset.
- It has now received the same focused `jump`/`run` disturbance and termination/state-trace diagnostics below.
- The next local no-training step is to continue the remaining top-task coverage audit through Jun20 and Jun19.

### Jun21 Conservative-Penalty Top-7654 Robustness Diagnostic - 2026-06-30

Hypothesis: because Jun21 `command_hold_conservative_penalty_ramp/model_top_task_7654.pt` nearly matches Jun17 selective-walk best on no-disturb fixed-preset aggregate metrics, it needs the same `jump`/`run` disturbance and state-trace diagnostic before ranking it as a warm-start/control candidate.

Local diagnostic outputs:

```text
outputs/eval/June30_Jun21_conservative_penalty_top_7654_disturb_sweep
outputs/eval/June30_Jun21_conservative_penalty_top_7654_disturb_sweep/summary_metrics.csv
outputs/eval/June30_Jun21_conservative_penalty_top_7654_state_trace_corrected
outputs/eval/June30_Jun21_conservative_penalty_top_7654_state_trace_disturb100
outputs/eval/June30_Jun21_conservative_penalty_top_7654_state_trace_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/_topk_eval_compat/Jun21/conservative_penalty_ramp_top_7654/model_7654.pt`, hard-linked from `logs/r2_amp/Jun21/Jun21_12-28-33_command_hold_conservative_penalty_ramp/model_top_task_7654.pt`.
- Config: `configs/ablation/command_hold_conservative_penalty_ramp.json`.
- Disturbance sweep: `jump` and `run`, ratios `0.0`, `0.25`, `0.5`, `0.75`, `1.0`, `64` episodes per row.
- State trace: `jump` and `run`, `64` episodes per row, `--record_termination_reasons`, `--record_state_trace`, `--state_trace_window_steps=50`, once with default corrected no-disturb evaluation and once with `--eval_disturb_ratio=1.0`.

Focused disturbance sweep:

| preset | disturb ratio | task return | fall rate | survival s | lin rmse | yaw rmse | style reward | disc gap | torque L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 0.00 | 31.39 | 0.047 | 9.66 | 0.203 | 0.295 | 0.00834 | 1.350 | 19949 |
| `jump` | 0.25 | 30.88 | 0.062 | 9.58 | 0.237 | 0.296 | 0.00835 | 1.344 | 18312 |
| `jump` | 0.50 | 29.18 | 0.062 | 9.56 | 0.246 | 0.331 | 0.00816 | 1.359 | 16129 |
| `jump` | 0.75 | 6.83 | 0.500 | 6.94 | 0.519 | 0.596 | 0.00289 | 1.714 | 14785 |
| `jump` | 1.00 | -13.18 | 1.000 | 1.34 | 0.755 | 0.927 | 0.00391 | 1.651 | 13660 |
| `run` | 0.00 | 3.23 | 0.188 | 8.33 | 0.801 | 0.760 | 0.00521 | 1.580 | 40841 |
| `run` | 0.25 | 5.32 | 0.031 | 9.78 | 0.697 | 0.647 | 0.00560 | 1.577 | 38167 |
| `run` | 0.50 | 0.77 | 0.078 | 9.56 | 0.754 | 0.775 | 0.00443 | 1.641 | 35051 |
| `run` | 0.75 | -3.56 | 0.250 | 8.19 | 0.861 | 1.068 | 0.00161 | 1.806 | 29771 |
| `run` | 1.00 | -18.41 | 1.000 | 2.44 | 1.307 | 1.820 | 0.00148 | 1.817 | 33311 |

Termination facts:

| protocol | preset | task return | fall rate | survival s | termination reason | detail | count | rate | mean survival s |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| corrected no-disturb | `jump` | 29.37 | 0.109 | 9.40 | contact | `base_link` | 7 | 0.109 | 4.32 |
| corrected no-disturb | `jump` | 29.37 | 0.109 | 9.40 | timeout | - | 57 | 0.891 | 10.02 |
| corrected no-disturb | `run` | 7.64 | 0.172 | 8.64 | contact | `base_link` | 11 | 0.172 | 1.97 |
| corrected no-disturb | `run` | 7.64 | 0.172 | 8.64 | timeout | - | 53 | 0.828 | 10.02 |
| full disturb ratio 1.0 | `jump` | -13.18 | 1.000 | 1.34 | contact | `base_link` | 62 | 0.969 | 1.34 |
| full disturb ratio 1.0 | `jump` | -13.18 | 1.000 | 1.34 | orientation | `roll_pitch` | 2 | 0.031 | 1.41 |
| full disturb ratio 1.0 | `run` | -17.11 | 1.000 | 2.26 | contact | `base_link` | 52 | 0.812 | 2.29 |
| full disturb ratio 1.0 | `run` | -17.11 | 1.000 | 2.26 | orientation | `roll_pitch` | 12 | 0.188 | 2.15 |

State-trace facts:

| protocol | preset | reason | detail | n | final z | min z | max abs roll | max abs pitch | final lin err | final yaw err | max contact |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected no-disturb | `jump` | contact | `base_link` | 7 | 0.741 | 0.673 | 0.291 | 0.735 | 1.001 | 0.831 | 1295.4 |
| corrected no-disturb | `jump` | timeout | - | 57 | 0.745 | 0.731 | 0.051 | 0.215 | 0.518 | -0.096 | 0.0 |
| corrected no-disturb | `run` | contact | `base_link` | 11 | 0.697 | 0.644 | 0.272 | 0.456 | 1.297 | 0.753 | 797.4 |
| corrected no-disturb | `run` | timeout | - | 53 | 0.736 | 0.715 | 0.116 | 0.128 | 0.933 | -0.491 | 0.0 |
| full disturb ratio 1.0 | `jump` | contact | `base_link` | 62 | 0.557 | 0.278 | 0.916 | 1.080 | 1.759 | 0.139 | 3980.6 |
| full disturb ratio 1.0 | `jump` | orientation | `roll_pitch` | 2 | 0.642 | 0.559 | 0.563 | 1.018 | 1.524 | -0.221 | 0.0 |
| full disturb ratio 1.0 | `run` | contact | `base_link` | 52 | 0.550 | 0.364 | 0.873 | 1.081 | 1.619 | 0.117 | 5859.1 |
| full disturb ratio 1.0 | `run` | orientation | `roll_pitch` | 12 | 0.543 | 0.265 | 0.826 | 1.047 | 2.210 | -0.556 | 0.0 |

Facts:

- Jun21 `top_7654` is competitive on no-disturb `jump`, but weaker than Jun17 selective-walk best under partial disturbance. At `jump` ratio `0.75`, Jun21 fall rate is `0.500` while Jun17 selective-walk best is `0.016`.
- `run` is the decisive weakness. At no-disturb and ratio `0.75`, Jun21 `top_7654` has task returns `3.23` and `-3.56`; Jun17 selective-walk best has `16.10` and `11.57`.
- Full disturbance breaks Jun21 `top_7654` in both `jump` and `run`, with `fall_rate=1.000`.
- Full-disturb failures are mostly `base_link` contact, with roll/pitch orientation as a secondary termination path. This is the same broad failure class as Jun17 selective-walk best and Jun25_0 conservative `8000`, but the Jun21 checkpoint reaches the failure region earlier at ratio `0.75`.

Interpretation:

- Jun21 `conservative_penalty_top_7654` remains an important historical control, but it should not replace Jun17 selective-walk best as the leading warm-start/control source.
- The positive signal from Jun21 is not full robustness; it is that a non-selective-style conservative-penalty run can produce a strong intermediate no-disturb checkpoint before best/final task returns collapse.
- The next local no-training step should return to top-task coverage for Jun20 and Jun19. If a later training run needs a second warm-start comparison, Jun21 `top_7654` is the best secondary candidate after Jun17 selective-walk best.

### Jun20 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: the June20 best/final evaluations made the batch look weak, but the saved `model_top_task_*` checkpoints may contain high-quality intermediate policies before late task-return collapse.

Compatibility facts:

- Twelve Jun20 top-task checkpoints load successfully in the current `r2gym` evaluation environment, and their internal `iter` values match the filename labels:
  - `command_hold_conservative_penalty_ramp`: `model_top_task_5818.pt`, `model_top_task_7663.pt`, `model_top_task_7930.pt`
  - `command_hold_controlled_disturb_release`: `model_top_task_1166.pt`, `model_top_task_1706.pt`, `model_top_task_1944.pt`
  - `command_hold_no_push`: `model_top_task_6059.pt`, `model_top_task_6973.pt`, `model_top_task_7440.pt`
  - `command_hold_style_lowcap`: `model_top_task_7439.pt`, `model_top_task_7600.pt`, `model_top_task_7937.pt`
- Local compatibility load directories were created under:

```text
logs/r2_amp/_topk_eval_compat/Jun20
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun20_top_conservative_penalty_ramp_5818
outputs/eval/June30_Jun20_top_conservative_penalty_ramp_7663
outputs/eval/June30_Jun20_top_conservative_penalty_ramp_7930
outputs/eval/June30_Jun20_top_controlled_disturb_1166
outputs/eval/June30_Jun20_top_controlled_disturb_1706
outputs/eval/June30_Jun20_top_controlled_disturb_1944
outputs/eval/June30_Jun20_top_no_push_6059
outputs/eval/June30_Jun20_top_no_push_6973
outputs/eval/June30_Jun20_top_no_push_7440
outputs/eval/June30_Jun20_top_style_lowcap_7439
outputs/eval/June30_Jun20_top_style_lowcap_7600
outputs/eval/June30_Jun20_top_style_lowcap_7937
outputs/eval/June30_Jun20_top_task_eval_summary/jun20_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/command_hold_conservative_penalty_ramp.json`
  - `configs/ablation/command_hold_controlled_disturb_release.json`
  - `configs/ablation/command_hold_no_push.json`
  - `configs/ablation/command_hold_style_lowcap.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `conservative_penalty_top_5818` | `command_hold_conservative_penalty_ramp` | `5818` | `top_task` | 7 | 30.28 | 0.062 | 9.63 | 0.289 | 0.379 | 0.00475 | -0.739 | 1.630 | `run` | 11.41 | `stand` | 0.172 |
| `conservative_penalty_top_7663` | `command_hold_conservative_penalty_ramp` | `7663` | `top_task` | 7 | 29.44 | 0.080 | 9.55 | 0.290 | 0.414 | 0.00553 | -0.694 | 1.586 | `run` | 9.66 | `jump` | 0.312 |
| `conservative_penalty_top_7930` | `command_hold_conservative_penalty_ramp` | `7930` | `top_task` | 7 | 29.19 | 0.071 | 9.64 | 0.291 | 0.431 | 0.00541 | -0.703 | 1.596 | `run` | 10.94 | `jump` | 0.203 |
| `style_lowcap_top_7439` | `command_hold_style_lowcap` | `7439` | `top_task` | 7 | 28.64 | 0.056 | 9.68 | 0.295 | 0.440 | 0.00538 | -0.699 | 1.568 | `run` | 3.92 | `run` | 0.266 |
| `style_lowcap_top_7600` | `command_hold_style_lowcap` | `7600` | `top_task` | 7 | 27.95 | 0.132 | 9.12 | 0.332 | 0.467 | 0.00522 | -0.704 | 1.570 | `run` | -0.85 | `run` | 0.703 |
| `no_push_top_6973` | `command_hold_no_push` | `6973` | `top_task` | 7 | 26.68 | 0.205 | 8.24 | 0.389 | 0.530 | 0.00278 | -0.847 | 1.733 | `run` | -6.78 | `run` | 0.984 |
| `style_lowcap_top_7937` | `command_hold_style_lowcap` | `7937` | `top_task` | 7 | 26.23 | 0.188 | 8.58 | 0.364 | 0.506 | 0.00476 | -0.729 | 1.597 | `run` | -5.12 | `run` | 1.000 |
| `no_push_top_7440` | `command_hold_no_push` | `7440` | `top_task` | 7 | 24.92 | 0.250 | 7.84 | 0.402 | 0.521 | 0.00272 | -0.848 | 1.732 | `run` | -5.65 | `run` | 1.000 |
| `no_push_top_6059` | `command_hold_no_push` | `6059` | `top_task` | 7 | 24.58 | 0.199 | 8.39 | 0.401 | 0.567 | 0.00266 | -0.846 | 1.739 | `run` | 2.82 | `run` | 0.578 |
| `controlled_disturb_top_1706` | `command_hold_controlled_disturb_release` | `1706` | `top_task` | 7 | 20.24 | 0.290 | 7.47 | 0.416 | 0.576 | 0.00431 | -0.768 | 1.631 | `jump` | -4.67 | `jump` | 1.000 |
| `controlled_disturb_top_1944` | `command_hold_controlled_disturb_release` | `1944` | `top_task` | 7 | 17.01 | 0.357 | 6.92 | 0.455 | 0.608 | 0.00437 | -0.761 | 1.624 | `jump` | -4.68 | `jump` | 1.000 |
| `controlled_disturb_top_1166` | `command_hold_controlled_disturb_release` | `1166` | `top_task` | 7 | 16.33 | 0.429 | 6.55 | 0.479 | 0.633 | 0.00418 | -0.781 | 1.637 | `run` | -4.60 | `jump` | 1.000 |
| `controlled_disturb_best` | `command_hold_controlled_disturb_release` | `best` | `best_task` | 7 | -9.44 | 0.435 | - | 0.536 | 0.651 | 0.00418 | -0.782 | 1.638 | `turn_left` | -13.58 | `run` | 1.000 |
| `no_push_best` | `command_hold_no_push` | `best` | `best_task` | 7 | -22.34 | 0.205 | - | 0.441 | 0.604 | 0.00269 | -0.845 | 1.738 | `jump` | -28.75 | `run` | 0.609 |
| `no_push_8000` | `command_hold_no_push` | `8000` | `final` | 7 | -31.86 | 0.143 | - | 0.346 | 0.520 | 0.00264 | -0.853 | 1.734 | `jump` | -41.82 | `run` | 0.500 |
| `style_lowcap_8000` | `command_hold_style_lowcap` | `8000` | `final` | 7 | -44.22 | 0.167 | - | 0.368 | 0.489 | 0.00476 | -0.728 | 1.594 | `walk_fast` | -61.55 | `run` | 1.000 |
| `style_lowcap_best` | `command_hold_style_lowcap` | `best` | `best_task` | 7 | -45.33 | 0.154 | - | 0.358 | 0.498 | 0.00482 | -0.724 | 1.593 | `walk_fast` | -61.84 | `run` | 1.000 |
| `controlled_disturb_8000` | `command_hold_controlled_disturb_release` | `8000` | `final` | 7 | -67.70 | 0.049 | - | 0.318 | 0.466 | 0.00585 | -0.670 | 1.547 | `run` | -94.09 | `jump` | 0.172 |
| `conservative_penalty_best` | `command_hold_conservative_penalty_ramp` | `best` | `best_task` | 7 | -79.37 | 0.087 | - | 0.317 | 0.426 | 0.00537 | -0.704 | 1.597 | `turn_left` | -93.46 | `run` | 0.188 |
| `conservative_penalty_8000` | `command_hold_conservative_penalty_ramp` | `8000` | `final` | 7 | -80.78 | 0.080 | - | 0.298 | 0.438 | 0.00544 | -0.701 | 1.590 | `turn_left` | -91.15 | `jump` | 0.391 |

Selected per-preset facts:

- `conservative_penalty_top_5818` is the strongest no-disturb fixed-preset checkpoint found so far by aggregate task return (`30.28`), narrowly above Jun17 selective-walk best (`29.79`).
- Its weakest preset is still `run`: task return `11.41`, fall rate `0.125`, survival `9.20s`. This is better than Jun21 `top_7654` on `run` but still weaker than Jun17 selective-walk best (`run` task `20.03`, fall `0.031`).
- `style_lowcap_top_7439` is also strong by aggregate fall rate (`0.056`) and task return (`28.64`), but its `run` row is much weaker (`task_return=3.92`, `fall_rate=0.266`).
- The best/final rows for June20 are misleading for task quality because several final checkpoints keep low fall rates while task return collapses to very negative values.

Interpretation:

- The Jun20 top-task checkpoint gap is now closed for all four Jun20 runs.
- Jun20 materially changes the candidate ranking: `command_hold_conservative_penalty_ramp/model_top_task_5818.pt` is now the strongest no-disturb aggregate checkpoint, but it needs focused robustness evidence before replacing Jun17 selective-walk best as the leading candidate.
- The high top-task scores support a repeated pattern: intermediate snapshots before late collapse are more informative than the archived best/final names for this experiment line.

### Jun20 Conservative-Penalty Top-5818 Robustness Diagnostic - 2026-06-30

Hypothesis: because Jun20 `command_hold_conservative_penalty_ramp/model_top_task_5818.pt` is now the strongest no-disturb aggregate checkpoint, it needs the same `jump`/`run` disturbance and state-trace diagnostic used for Jun17 selective-walk best and Jun21 `top_7654`.

Local diagnostic outputs:

```text
outputs/eval/June30_Jun20_conservative_penalty_top_5818_disturb_sweep
outputs/eval/June30_Jun20_conservative_penalty_top_5818_disturb_sweep/summary_metrics.csv
outputs/eval/June30_Jun20_conservative_penalty_top_5818_state_trace_corrected
outputs/eval/June30_Jun20_conservative_penalty_top_5818_state_trace_disturb100
outputs/eval/June30_Jun20_conservative_penalty_top_5818_state_trace_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/_topk_eval_compat/Jun20/conservative_penalty_ramp_top_5818/model_5818.pt`, hard-linked from `logs/r2_amp/Jun20/Jun20_15-18-58_command_hold_conservative_penalty_ramp/model_top_task_5818.pt`.
- Config: `configs/ablation/command_hold_conservative_penalty_ramp.json`.
- Disturbance sweep: `jump` and `run`, ratios `0.0`, `0.25`, `0.5`, `0.75`, `1.0`, `64` episodes per row.
- State trace: `jump` and `run`, `64` episodes per row, `--record_termination_reasons`, `--record_state_trace`, `--state_trace_window_steps=50`, once with default corrected no-disturb evaluation and once with `--eval_disturb_ratio=1.0`.

Focused disturbance sweep:

| preset | disturb ratio | task return | fall rate | survival s | lin rmse | yaw rmse | style reward | disc gap | torque L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 0.00 | 24.76 | 0.203 | 8.45 | 0.278 | 0.380 | 0.00639 | 1.491 | 21345 |
| `jump` | 0.25 | 28.76 | 0.094 | 9.28 | 0.271 | 0.343 | 0.00688 | 1.473 | 15030 |
| `jump` | 0.50 | 29.01 | 0.062 | 9.49 | 0.250 | 0.321 | 0.00427 | 1.641 | 10835 |
| `jump` | 0.75 | 7.30 | 0.453 | 7.26 | 0.477 | 0.718 | 0.00287 | 1.733 | 12984 |
| `jump` | 1.00 | -18.12 | 1.000 | 1.34 | 0.953 | 1.288 | 0.00327 | 1.708 | 16651 |
| `run` | 0.00 | 10.61 | 0.203 | 9.09 | 0.546 | 0.702 | 0.00574 | 1.575 | 29061 |
| `run` | 0.25 | 18.22 | 0.031 | 9.82 | 0.516 | 0.536 | 0.00570 | 1.579 | 22681 |
| `run` | 0.50 | 13.69 | 0.000 | 10.02 | 0.482 | 0.661 | 0.00575 | 1.575 | 21039 |
| `run` | 0.75 | 3.94 | 0.078 | 9.62 | 0.575 | 1.040 | 0.00158 | 1.814 | 23700 |
| `run` | 1.00 | -12.72 | 1.000 | 1.46 | 1.126 | 1.594 | 0.00304 | 1.737 | 24339 |

Termination facts:

| protocol | preset | task return | fall rate | survival s | termination reason | detail | count | rate | mean survival s |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| corrected no-disturb | `jump` | 24.40 | 0.172 | 8.68 | contact | `base_link` | 11 | 0.172 | 2.25 |
| corrected no-disturb | `jump` | 24.40 | 0.172 | 8.68 | timeout | - | 53 | 0.828 | 10.02 |
| corrected no-disturb | `run` | 10.80 | 0.203 | 9.13 | contact | `base_link` | 13 | 0.203 | 5.64 |
| corrected no-disturb | `run` | 10.80 | 0.203 | 9.13 | timeout | - | 51 | 0.797 | 10.02 |
| full disturb ratio 1.0 | `jump` | -18.12 | 1.000 | 1.34 | contact | `base_link` | 53 | 0.828 | 1.32 |
| full disturb ratio 1.0 | `jump` | -18.12 | 1.000 | 1.34 | orientation | `roll_pitch` | 11 | 0.172 | 1.42 |
| full disturb ratio 1.0 | `run` | -12.16 | 1.000 | 1.43 | contact | `base_link` | 41 | 0.641 | 1.51 |
| full disturb ratio 1.0 | `run` | -12.16 | 1.000 | 1.43 | orientation | `roll_pitch` | 23 | 0.359 | 1.30 |

State-trace facts:

| protocol | preset | reason | detail | n | final z | min z | max abs roll | max abs pitch | final lin err | final yaw err | max contact |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected no-disturb | `jump` | contact | `base_link` | 11 | 0.679 | 0.616 | 0.437 | 0.729 | 1.246 | -0.261 | 2106.3 |
| corrected no-disturb | `jump` | timeout | - | 53 | 0.737 | 0.723 | 0.033 | 0.221 | 0.573 | 0.031 | 0.0 |
| corrected no-disturb | `run` | contact | `base_link` | 13 | 0.660 | 0.528 | 0.551 | 0.641 | 1.504 | -0.911 | 1463.4 |
| corrected no-disturb | `run` | timeout | - | 51 | 0.734 | 0.684 | 0.084 | 0.245 | 1.158 | 0.020 | 0.0 |
| full disturb ratio 1.0 | `jump` | contact | `base_link` | 53 | 0.475 | 0.350 | 0.908 | 1.091 | 1.667 | -0.902 | 4423.0 |
| full disturb ratio 1.0 | `jump` | orientation | `roll_pitch` | 11 | 0.543 | 0.437 | 0.922 | 1.067 | 1.695 | -0.323 | 0.0 |
| full disturb ratio 1.0 | `run` | contact | `base_link` | 41 | 0.467 | 0.349 | 0.882 | 1.088 | 1.297 | -0.904 | 4514.2 |
| full disturb ratio 1.0 | `run` | orientation | `roll_pitch` | 23 | 0.471 | 0.389 | 0.885 | 1.092 | 1.404 | -1.956 | 0.0 |

Facts:

- Jun20 `top_5818` is the best no-disturb aggregate checkpoint so far, but not the strongest robustness checkpoint.
- At `run` ratio `0.75`, Jun20 `top_5818` keeps fall rate relatively low but still higher than Jun17 selective-walk best (`0.078` vs `0.016`), and its task return is much lower (`3.94` vs `11.57`).
- At `jump` ratio `0.75`, Jun20 `top_5818` degrades sharply (`task_return=7.30`, `fall_rate=0.453`), while Jun17 selective-walk best remains stable (`task_return=30.55`, `fall_rate=0.016`).
- Full disturbance still breaks the checkpoint in both `jump` and `run`, with `fall_rate=1.000`.
- Failure mode remains dominated by `base_link` contact, with roll/pitch orientation as a secondary full-disturb path.

Interpretation:

- Jun20 `conservative_penalty_top_5818` should be ranked first for no-disturb fixed-preset aggregate quality, but Jun17 selective-walk best remains the better robustness/warm-start candidate because it handles partial `jump` disturbance much better and has stronger `run` task return.
- The next training decision should not use no-disturb aggregate alone. Candidate choice should weight partial-disturb `jump/run` diagnostics above small differences in seven-preset no-disturb average return.
- With the Jun19 top-task audit below, the current saved top-task checkpoint coverage pass for the Jun19-Jun25 line is complete.

### Jun19 Top-Task Checkpoint Evaluation - 2026-06-30

Hypothesis: the first Jun19 ablation batch looked weak under archived best/final checkpoints, but its saved `model_top_task_*` snapshots may contain intermediate policies that close part of the gap to the later Jun20-Jun21 candidates.

Compatibility facts:

- Twelve Jun19 top-task checkpoints load successfully in the current `r2gym` evaluation environment:
  - `scratch_amp_slow_lowcap`: `model_top_task_1214.pt`, `model_top_task_1227.pt`, `model_top_task_1806.pt`
  - `scratch_command_hold`: `model_top_task_7120.pt`, `model_top_task_7219.pt`, `model_top_task_7966.pt`
  - `scratch_no_push`: `model_top_task_1676.pt`, `model_top_task_1685.pt`, `model_top_task_1881.pt`
  - `scratch_slow_penalty_ramp`: `model_top_task_1163.pt`, `model_top_task_1219.pt`, `model_top_task_1222.pt`
- The `scratch_slow_penalty_ramp` directory is under the Jun19 group but has the timestamped path `logs/r2_amp/Jun19/Jun20_04-58-31_scratch_slow_penalty_ramp`; this is treated as part of the Jun19 first ablation batch because it matches the same config family and was already grouped there in the earlier best/final evaluation record.
- Local compatibility load directories were created under:

```text
logs/r2_amp/_topk_eval_compat/Jun19
```

Local evaluation outputs:

```text
outputs/eval/June30_Jun19_top_amp_slow_lowcap_1214
outputs/eval/June30_Jun19_top_amp_slow_lowcap_1227
outputs/eval/June30_Jun19_top_amp_slow_lowcap_1806
outputs/eval/June30_Jun19_top_command_hold_7120
outputs/eval/June30_Jun19_top_command_hold_7219
outputs/eval/June30_Jun19_top_command_hold_7966
outputs/eval/June30_Jun19_top_no_push_1676
outputs/eval/June30_Jun19_top_no_push_1685
outputs/eval/June30_Jun19_top_no_push_1881
outputs/eval/June30_Jun19_top_slow_penalty_ramp_1163
outputs/eval/June30_Jun19_top_slow_penalty_ramp_1219
outputs/eval/June30_Jun19_top_slow_penalty_ramp_1222
outputs/eval/June30_Jun19_top_task_eval_summary/jun19_top_task_eval_summary.csv
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default 7 fixed presets from `evaluate.py`.
- Config overrides:
  - `configs/ablation/scratch_amp_slow_lowcap.json`
  - `configs/ablation/scratch_command_hold.json`
  - `configs/ablation/scratch_no_push.json`
  - `configs/ablation/scratch_slow_penalty_ramp.json`

Aggregate comparison:

| eval | config | checkpoint | source | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `command_hold_top_7966` | `scratch_command_hold` | `7966` | `top_task` | 7 | 23.48 | 0.138 | 9.00 | 0.418 | 0.559 | 0.00466 | -0.726 | 1.620 | `run` | -2.12 | `run` | 0.578 |
| `no_push_top_1881` | `scratch_no_push` | `1881` | `top_task` | 7 | 23.21 | 0.172 | 8.82 | 0.395 | 0.476 | 0.00376 | -0.803 | 1.672 | `run` | -0.27 | `stand` | 0.578 |
| `command_hold_top_7219` | `scratch_command_hold` | `7219` | `top_task` | 7 | 23.06 | 0.188 | 8.62 | 0.430 | 0.593 | 0.00456 | -0.731 | 1.621 | `run` | -7.89 | `run` | 0.938 |
| `slow_penalty_ramp_top_1163` | `scratch_slow_penalty_ramp` | `1163` | `top_task` | 7 | 21.40 | 0.132 | 9.26 | 0.347 | 0.526 | 0.00413 | -0.784 | 1.638 | `run` | 6.53 | `stand` | 0.344 |
| `slow_penalty_ramp_top_1222` | `scratch_slow_penalty_ramp` | `1222` | `top_task` | 7 | 21.26 | 0.152 | 9.09 | 0.362 | 0.505 | 0.00426 | -0.776 | 1.638 | `run` | 1.86 | `jump` | 0.328 |
| `command_hold_top_7120` | `scratch_command_hold` | `7120` | `top_task` | 7 | 20.60 | 0.210 | 8.36 | 0.450 | 0.657 | 0.00418 | -0.753 | 1.645 | `run` | -8.73 | `run` | 1.000 |
| `no_push_top_1685` | `scratch_no_push` | `1685` | `top_task` | 7 | 19.59 | 0.263 | 8.05 | 0.434 | 0.575 | 0.00393 | -0.797 | 1.669 | `run` | -2.60 | `jump` | 1.000 |
| `no_push_top_1676` | `scratch_no_push` | `1676` | `top_task` | 7 | 19.59 | 0.283 | 8.01 | 0.447 | 0.594 | 0.00380 | -0.803 | 1.671 | `run` | -3.50 | `jump` | 0.750 |
| `amp_slow_lowcap_top_1806` | `scratch_amp_slow_lowcap` | `1806` | `top_task` | 7 | 19.07 | 0.203 | 8.53 | 0.415 | 0.544 | 0.00376 | -0.806 | 1.675 | `run` | 6.74 | `jump` | 0.656 |
| `amp_slow_lowcap_top_1214` | `scratch_amp_slow_lowcap` | `1214` | `top_task` | 7 | 18.94 | 0.234 | 8.25 | 0.390 | 0.523 | 0.00392 | -0.796 | 1.648 | `jump` | -1.92 | `jump` | 1.000 |
| `slow_penalty_ramp_top_1219` | `scratch_slow_penalty_ramp` | `1219` | `top_task` | 7 | 18.86 | 0.194 | 8.71 | 0.403 | 0.544 | 0.00391 | -0.797 | 1.647 | `run` | 6.69 | `jump` | 0.531 |
| `amp_slow_lowcap_top_1227` | `scratch_amp_slow_lowcap` | `1227` | `top_task` | 7 | 18.54 | 0.243 | 8.27 | 0.400 | 0.529 | 0.00399 | -0.791 | 1.645 | `jump` | -1.99 | `jump` | 1.000 |

For comparison, the earlier Jun19 best/final rows remain far below the top-task snapshots by task return: `no_push_best=-11.48`, `slow_penalty_ramp_best=-14.85`, `command_hold_best=-21.57`, `slow_penalty_ramp_8000=-21.70`, `amp_slow_lowcap_best=-22.25`, `command_hold_8000=-24.24`, `no_push_8000=-64.47`, and `amp_slow_lowcap_8000=-142.87`.

Selected per-preset facts:

- `command_hold_top_7966` is the strongest Jun19 top-task checkpoint by aggregate task return (`23.48`), but its weakest preset is `run`: task return `-2.12`, fall rate `0.578`, survival `5.50s`.
- `no_push_top_1881` has a less negative `run` return (`-0.27`) and lower `run` fall rate (`0.172`) than `command_hold_top_7966`, but it has weak `stand` and `jump` stability: `stand` fall rate `0.578`, `jump` fall rate `0.359`.
- `slow_penalty_ramp_top_1163` is the cleanest Jun19 snapshot by average fall rate among the better task-return rows (`0.132`), but its aggregate task return (`21.40`) is below `command_hold_top_7966` and far below the Jun20/Jun21 leaders.

Interpretation:

- The Jun19 top-task checkpoint gap is now closed for the first ablation batch.
- The top-task snapshots substantially improve the Jun19 read compared with best/final checkpoint names, confirming the same late-collapse pattern seen in later batches.
- No Jun19 checkpoint changes the warm-start/control hierarchy: Jun20 `conservative_penalty_top_5818` remains the best no-disturb aggregate checkpoint, Jun17 selective-walk best remains the leading robustness/warm-start candidate, and Jun21 `conservative_penalty_top_7654` remains the best secondary control.
- A focused disturbance diagnostic is not justified for Jun19 at this point because even the best Jun19 top-task row has much lower aggregate task return and much worse `run` stability than the already-diagnosed Jun20, Jun21, and Jun17 candidates.

### Next Training Config - Selective-Walk Conservative Eval-Manifold - 2026-06-30

Hypothesis: Jun17 `expert_hard_gate_selective_walk/model_best_task.pt` is the best warm-start/control source because it has the strongest partial-disturb `jump/run` robustness, but it should be combined with the conservative eval-manifold disturbance curriculum rather than full-disturb release.

Config added:

```text
configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json
```

Design:

- Warm-start source: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`.
- Keep the seven-profile eval-manifold command mixture from `command_hold_eval_manifold_conservative_disturb_release.json`.
- Keep conservative staged disturbance release: levels `[0.0, 0.05, 0.1, 0.18, 0.28, 0.42, 0.6, 0.75]`, `stage_min_episodes=2048`, all-profile monitor, adaptive regression.
- Keep three motion experts registered, but use selective-walk style contribution: `walk=true`, `run=false`, `jump=false` in both `env.amp.expert_style_enabled` and `train.amp.expert_style_enabled`.
- Save top task checkpoints and train for `8000` resumed iterations when launched.

Suggested launch command when training budget is available:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py --task=r2amp --headless --seed=0 --resume --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk --checkpoint=-2 --cfg_override_json configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json
```

Status: `not trained`.

Post-training evaluation helpers added:

```text
scripts/plan_selective_walk_followup_eval.py
scripts/plan_selective_walk_followup_train.py
scripts/run_selective_walk_followup_eval_plan.py
scripts/summarize_selective_walk_followup_eval.py
scripts/audit_selective_walk_followup_readiness.py
```

Purpose: once a real `selective_walk_eval_manifold_conservative_disturb_release` training run produces checkpoints, this helper prints the standard WSL CPU evaluation commands instead of requiring manual command reconstruction. For each checkpoint it emits nine commands: no-disturb full7, forced-disturb full7 at `0.75`, `0.9`, `0.925`, `0.95`, and `1.0`, plus termination/state-trace diagnostics at `0.925`, `0.95`, and `1.0`. This is a planning helper only; it does not launch training or evaluation by itself.

The training planner prints the reviewed formal warm-start command for the same follow-up: Jun17 selective-walk `model_best_task.pt` via `--resume --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk --checkpoint=-2`, the follow-up JSON, and the formal run name `selective_walk_eval_manifold_conservative_disturb_release`. The Jun17 source checkpoint was verified with `torch.load(..., map_location="cpu")` to contain `iter=4000`, and `OnPolicyRunner.learn(num_learning_iterations=...)` treats `max_iterations` as additional updates after resume; therefore the planner defaults to `--max_iterations=4000` so the final target checkpoint is `model_8000.pt`. It is also a planner only; it does not launch the long training job.

The companion summarizer reads those nine planned output directories after the WSL CPU jobs finish, writes a compact CSV/JSON summary, and defaults to failing if any planned output is missing. This keeps the formal follow-up checkpoint evaluation from being marked complete from a partial set of `metrics.csv` files.

The readiness audit helper scans `logs/r2_amp` for matching follow-up run directories and checkpoint files, then checks whether the nine planned evaluation outputs exist. This separates transient/no-checkpoint run directories from a real trained checkpoint that is ready for evaluation.

The eval-plan runner consumes the readiness audit JSON. Its default mode is dry-run: it prints the audit's `recommended_eval_plan` commands without launching Isaac Gym. It runs only with explicit `--execute`, and it refuses execution when no real checkpoint exists and `recommended_eval_plan` is empty.

Template:

```powershell
python scripts\plan_selective_walk_followup_train.py
python scripts\plan_selective_walk_followup_eval.py --load_run <new_run_dir_under_logs/r2_amp> --checkpoint <checkpoint_id> --output_prefix <planned_eval_output_prefix> --json
python scripts\audit_selective_walk_followup_readiness.py --output_prefix <planned_eval_output_prefix> --output_json <audit_output_json>
python scripts\run_selective_walk_followup_eval_plan.py --audit_json <audit_output_json>
python scripts\summarize_selective_walk_followup_eval.py --output_prefix <planned_eval_output_prefix> --output_dir <summary_output_dir>
```

Readiness audit output:

```text
outputs/eval/June30_selective_walk_followup_readiness_audit
outputs/eval/June30_selective_walk_followup_readiness_audit/readiness_audit.json
```

Readiness audit result:

| metric | value |
|---|---:|
| matching follow-up run dirs | 15 |
| `load_only_no_training_progress` run dirs | 15 |
| `evaluate_checkpoint_load_log_dir` artifact_source dirs | 15 |
| matching checkpoints | 0 |
| planned eval outputs | 9 |
| present eval outputs | 0 |
| missing eval outputs | 9 |
| recommended checkpoint | `null` |
| recommended load_run | `null` |
| recommended eval commands | 0 |
| ready for evaluation | `false` |
| ready for completion | `false` |

Interpretation: the filesystem currently contains transient `selective_walk_eval_manifold_conservative_disturb_release` run directories, but each one contains only a 265-byte `train.log` with `Loading model from` / `load_path` and no TensorBoard event file, iteration log, or `model*.pt` checkpoint. These directories were created by the previous `evaluate.py` checkpoint-load path initializing `task_registry.make_alg_runner()` with the default runner log root; they are evaluation load artifacts, not failed training runs. The readiness JSON marks all 15 such rows with `artifact_source="evaluate_checkpoint_load_log_dir"` so this source is machine-readable instead of only a prose interpretation. `evaluate.py` now passes `log_root=None` during checkpoint evaluation so future eval jobs should not create new train-style run directories under `logs/r2_amp`. The readiness audit emits `recommended_checkpoint`, `recommended_load_run`, and `recommended_eval_plan` only when a real checkpoint exists; all three are empty/null in the current audit, so no formal follow-up checkpoint evaluation should be launched yet. The next required external step is still a real training run that enters the training loop and produces at least one checkpoint.

Evaluation log-dir regression smoke:

```text
outputs/eval/June30_evaluate_log_root_none_smoke
```

Protocol: WSL CPU PhysX / CPU policy eval, Jun17 selective-walk `model_best_task.pt` via `--checkpoint=-2`, config `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`, preset `stand`, `--num_envs=1`, `--num_episodes=1`, `--episode_seconds=0.2`. This smoke exists only to verify the evaluation toolchain after the `log_root=None` fix; it is not a policy-quality result.

Result: `metrics.csv` and `metrics.json` were written successfully. `metrics.csv` contains one `stand` row with `task_return_mean=0.4041624665`, `fall_rate=0.0`, and `survival_time_mean_s=0.2199999951`. The matching `logs/r2_amp/*selective_walk_eval_manifold_conservative_disturb_release*` directory count remained `15` before and after the smoke, so the fixed evaluation path did not create a new train-style run directory.

Training-entry smoke:

```text
logs/r2_amp/Jun30_19-34-57_smoke_sw_eval_manifold_conservative_disturb_release
outputs/eval/June30_smoke_sw_eval_manifold_conservative_4001_stand
```

Protocol: WSL CPU PhysX / CPU policy, `train.py --task=r2amp --num_envs=4 --max_iterations=1`, `--resume --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk --checkpoint=-2`, config `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`, and override `--run_name smoke_sw_eval_manifold_conservative_disturb_release`. The smoke run name intentionally avoids the full formal `selective_walk_eval_manifold_conservative_disturb_release` substring so readiness audit does not treat it as the real follow-up training run.

Result: the run entered `OnPolicyRunner.learn()` and continued from the resumed checkpoint iteration, logging `Learning iteration 4000/4001`. It wrote TensorBoard events, `train.log`, `model_4000.pt`, `model_4001.pt`, `model_best_task.pt`, and `model_top_task_4000.pt`. The one-iteration window reported `Mean task reward=-4.00`, `Mean mixed reward=-4.00`, `staged_disturb_stage=0.0`, and `staged_disturb_window_fall_rate=1.0000`; this confirms plumbing only and is not a quality signal.

The newly written smoke checkpoint `model_4001.pt` was then loaded through `evaluate.py` with preset `stand`, `--num_envs=1`, `--num_episodes=1`, and `--episode_seconds=0.2`. `outputs/eval/June30_smoke_sw_eval_manifold_conservative_4001_stand/metrics.csv` contains one row with `task_return_mean=0.2805526853`, `fall_rate=0.0`, and `survival_time_mean_s=0.2199999951`. The formal readiness audit still reports `runs_found=15`, `checkpoint_count=0`, `ready_for_evaluation=false`, and `ready_for_completion=false`, proving the smoke artifacts did not pollute the formal follow-up gate.

Compatibility smoke:

```text
outputs/eval/June30_selective_walk_conservative_followup_smoke_jump
outputs/eval/June30_selective_walk_conservative_followup_smoke_run
outputs/eval/June30_selective_walk_conservative_followup_smoke_jump_disturb075
outputs/eval/June30_selective_walk_conservative_followup_smoke_run_disturb075
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`, loaded through `--checkpoint=-2`.
- Config: `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`.
- `--preset jump` and `--preset run`, each as a separate output directory with `--num_envs=8`, `--num_episodes=4`, `--episode_seconds=2`.
- Each preset was run once without forced disturbance and once with `--eval_disturb_ratio=0.75` to exercise the same key partial-disturbance path that motivated this follow-up.
- Purpose: load-chain, config compatibility, and forced-disturbance plumbing smoke only. This is not a formal ranking evaluation because it uses two presets, four short episodes each, and the old checkpoint rather than a newly trained follow-up checkpoint.

Smoke result:

| preset | forced disturb ratio | rows | episodes | seconds | task return | fall rate | survival s | style reward | disc gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 0.00 | 1 | 4 | 2.0 | 5.13 | 0.000 | 2.02 | 0.00002 | 1.673 |
| `run` | 0.00 | 1 | 4 | 2.0 | -0.81 | 0.000 | 2.02 | 0.00005 | 1.591 |
| `jump` | 0.75 | 1 | 4 | 2.0 | 4.69 | 0.000 | 2.02 | 0.00000 | 1.705 |
| `run` | 0.75 | 1 | 4 | 2.0 | -1.38 | 0.000 | 2.02 | 0.00000 | 1.738 |

Fact: the first smoke attempt without CPU device flags reproduced the known local GPU incompatibility (`no kernel image is available for execution on the device` on RTX 5080 Laptop with `torch 2.4.1+cu118`). The CPU reruns with `--sim_device=cpu --rl_device=cpu` loaded the config and checkpoint and wrote `metrics.csv` / `metrics.json` successfully for both `jump` and `run`, including the forced `eval_disturb_ratio=0.75` path.

Full-shape old-checkpoint baseline:

```text
outputs/eval/June30_selective_walk_conservative_followup_baseline_full7
outputs/eval/June30_selective_walk_conservative_followup_baseline_full7/baseline_summary.json
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`, loaded through `--checkpoint=-2`.
- Config: `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`.
- Default seven fixed presets from `evaluate.py`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Purpose: full evaluation-shape compatibility/baseline for the warm-start checkpoint under the new follow-up config. This is still not a formal result for the follow-up policy, because no `selective_walk_eval_manifold_conservative_disturb_release` training checkpoint exists yet.

Baseline result:

| rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | style reward | policy logit | disc gap | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 28.03 | 0.038 | 9.76 | 0.339 | 0.485 | 0.00040 | -0.799 | 1.691 | `run` | 11.80 | `stand` | 0.109 |

Selected per-preset facts:

- `jump`: task return `34.48`, fall rate `0.000`, survival `10.02s`.
- `run`: task return `11.80`, fall rate `0.094`, survival `9.16s`.
- `stand`: task return `21.74`, fall rate `0.109`, survival `9.50s`.

Formal partial-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_jump_run_disturb075_formal
outputs/eval/June30_selective_walk_conservative_followup_jump_run_disturb075_formal/disturb075_summary.json
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`, loaded through `--checkpoint=-2`.
- Config: `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`.
- Presets: `jump` and `run`, `--eval_disturb_ratio=0.75`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Purpose: replace the earlier 4-episode smoke with a formal-size partial-disturbance diagnostic for the key presets that motivated this follow-up. This still uses the old warm-start checkpoint, so it remains diagnostic rather than a trained follow-up result.

Diagnostic result:

| preset | episodes | seconds | task return | fall rate | survival s | lin rmse | yaw rmse | disc gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 64 | 10.0 | 29.44 | 0.031 | 9.75 | 0.256 | 0.397 | 1.680 |
| `run` | 64 | 10.0 | 15.77 | 0.031 | 9.86 | 0.683 | 0.561 | 1.692 |

Aggregate over the two focused presets: avg task return `22.60`, avg fall rate `0.031`, avg survival `9.80s`. This supports the same training direction as the earlier Jun17 diagnostic: partial-disturb `jump/run` robustness is strong enough to justify using the selective-walk checkpoint as the warm-start for the conservative eval-manifold rerun.

Full seven-preset forced-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb075
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb075/full7_disturb075_summary.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb075/full7_disturb075_vs_baseline_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb075/full7_disturb075_vs_baseline_delta.json
```

Protocol:

- WSL CPU PhysX / CPU policy eval.
- Checkpoint: `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`, loaded through `--checkpoint=-2`.
- Config: `configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`.
- Default seven fixed presets from `evaluate.py`, `--eval_disturb_ratio=0.75`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Purpose: check whether the same forced `0.75` disturbance pressure that looks acceptable on `jump/run` creates hidden failures on the remaining evaluation presets. This is still an old-checkpoint diagnostic, not a trained follow-up result.

Diagnostic result:

| rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | base-height violation | roll/pitch violation | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 27.69 | 0.018 | 9.93 | 0.357 | 0.488 | 0.00176 | 0.00011 | `run` | 12.50 | `stand` | 0.047 |

Selected per-preset facts:

- `stand`: task return `25.41`, fall rate `0.047`, survival `9.93s`.
- `walk_slow`: task return `34.59`, fall rate `0.000`, survival `10.02s`.
- `run`: task return `12.50`, fall rate `0.031`, survival `9.76s`.
- `jump`: task return `31.91`, fall rate `0.031`, survival `9.89s`.
- `strafe_right`: task return `34.10`, fall rate `0.000`, survival `10.02s`.

Delta versus the no-forced-disturbance full7 baseline:

| metric | no-disturb baseline | forced 0.75 | delta |
|---|---:|---:|---:|
| avg task return | 28.03 | 27.69 | -0.34 |
| avg fall rate | 0.038 | 0.018 | -0.020 |
| avg survival s | 9.76 | 9.93 | +0.17 |

Largest local changes:

- Largest task-return drop: `strafe_right`, `-3.46`.
- Largest fall-rate increase: `jump`, `+0.031`.
- `run` improves slightly under forced `0.75` in this paired run: task return `+0.70`, fall rate `-0.063`, survival `+0.60s`.

Near-boundary forced-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb090_summary.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb090_vs_baseline_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb090_vs_baseline_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb090_vs_disturb075_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb090_vs_disturb075_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb100_vs_disturb090_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090/full7_disturb100_vs_disturb090_delta.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the forced `0.75` run, but with `--eval_disturb_ratio=0.9`.
- Purpose: locate whether the collapse boundary starts immediately above the conservative `0.75` cap or appears only near full-disturb `1.0`.

Diagnostic result:

| rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | base-height violation | roll/pitch violation | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 25.47 | 0.029 | 9.82 | 0.387 | 0.550 | 0.00343 | 0.00025 | `run` | 15.77 | `run` | 0.063 |

Selected per-preset facts:

- `stand`: task return `16.98`, fall rate `0.047`, survival `9.80s`.
- `walk_slow`: task return `31.01`, fall rate `0.031`, survival `9.78s`.
- `run`: task return `15.77`, fall rate `0.063`, survival `9.49s`.
- `jump`: task return `28.81`, fall rate `0.016`, survival `9.92s`.
- `strafe_right`: task return `32.68`, fall rate `0.016`, survival `9.93s`.

Stable-stress failure diagnostics:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics/termination_reasons.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics/state_trace.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics/failure_diagnostics_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics/failure_diagnostics_summary.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the `0.9` full7 run.
- Presets: `stand`, `run`, and `jump`, to match the onset/collapse diagnostic set while keeping the run focused.
- Flags: `--eval_disturb_ratio=0.9`, `--record_termination_reasons`, and `--record_state_trace`.

Failure summary:

| preset | task return | fall rate | contact `base_link` rate | orientation rate | timeout rate | contact mean survival s | terminal base z | terminal mean abs roll | terminal mean abs pitch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stand` | 16.98 | 0.047 | 0.016 | 0.031 | 0.953 | 5.72 | 0.751 | 0.048 | 0.117 |
| `run` | 16.42 | 0.047 | 0.031 | 0.016 | 0.953 | 3.37 | 0.742 | 0.049 | 0.138 |
| `jump` | 24.11 | 0.078 | 0.078 | 0.000 | 0.922 | 4.88 | 0.723 | 0.081 | 0.115 |

Interpretation: the `0.9` stress point remains stable under focused termination/state tracing. All three presets time out in more than `92%` of episodes, contact remains low-rate (`0.016-0.078`), and terminal base height stays close to the target (`0.72-0.75` versus target `0.78`). This makes `0.9` a useful post-training robustness check, while `0.925` is the first tested ratio with clear onset degradation.

Onset-boundary forced-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_summary.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_baseline_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_baseline_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_disturb075_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_disturb075_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_disturb090_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb0925_vs_disturb090_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb095_vs_disturb0925_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb095_vs_disturb0925_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb100_vs_disturb0925_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb100_vs_disturb0925_delta.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the other boundary runs, but with `--eval_disturb_ratio=0.925`.
- Purpose: check whether degradation starts immediately after `0.9` or closer to `0.95`.

Diagnostic result:

| rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | base-height violation | roll/pitch violation | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 22.18 | 0.054 | 9.68 | 0.432 | 0.619 | 0.00667 | 0.00042 | `run` | 11.42 | `stand` | 0.188 |

Selected per-preset facts:

- `stand`: task return `13.59`, fall rate `0.188`, survival `9.00s`.
- `walk_slow`: task return `25.21`, fall rate `0.000`, survival `10.02s`.
- `run`: task return `11.42`, fall rate `0.094`, survival `9.26s`.
- `jump`: task return `25.97`, fall rate `0.047`, survival `9.71s`.
- `strafe_right`: task return `30.36`, fall rate `0.016`, survival `9.97s`.

Onset-boundary failure diagnostics:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics/termination_reasons.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics/state_trace.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics/failure_diagnostics_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics/failure_diagnostics_summary.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the `0.925` full7 run.
- Presets: `stand`, `run`, and `jump`, because `stand` has the highest fall rate at `0.925`, `run` has the lowest return, and `jump` is the most sensitive preset at full-disturb `1.0`.
- Flags: `--eval_disturb_ratio=0.925`, `--record_termination_reasons`, and `--record_state_trace`.

Failure summary:

| preset | task return | fall rate | contact `base_link` rate | orientation rate | timeout rate | contact mean survival s | terminal base z | terminal mean abs roll | terminal mean abs pitch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stand` | 13.59 | 0.188 | 0.156 | 0.031 | 0.812 | 4.82 | 0.691 | 0.143 | 0.200 |
| `run` | 8.81 | 0.078 | 0.078 | 0.000 | 0.922 | 3.47 | 0.729 | 0.082 | 0.157 |
| `jump` | 24.34 | 0.062 | 0.062 | 0.000 | 0.938 | 3.29 | 0.732 | 0.064 | 0.101 |

Interpretation: the `0.925` onset is already dominated by `base_link` contact, but it is still a low-rate onset rather than a full collapse. Terminal base height stays around `0.69-0.73`, much closer to the `0.78` target than the `1.0` collapse diagnostics (`0.54-0.58`), and roll/pitch is still modest except for rare failures. This supports treating `0.925` as a stress-test threshold, not as a training target.

Transition-zone forced-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_summary.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_baseline_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_baseline_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_disturb075_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_disturb075_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_disturb090_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb095_vs_disturb090_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb100_vs_disturb095_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb100_vs_disturb095_delta.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the forced `0.75` and `0.9` runs, but with `--eval_disturb_ratio=0.95`.
- Purpose: determine whether the collapse transition is gradual between `0.9` and `1.0`, or concentrated at the exact full-disturb setting.

Diagnostic result:

| rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | base-height violation | roll/pitch violation | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 19.46 | 0.098 | 9.35 | 0.483 | 0.704 | 0.01086 | 0.00078 | `stand` | 11.21 | `stand` | 0.219 |

Selected per-preset facts:

- `stand`: task return `11.21`, fall rate `0.219`, survival `8.56s`.
- `walk_slow`: task return `26.42`, fall rate `0.047`, survival `9.63s`.
- `run`: task return `12.28`, fall rate `0.125`, survival `9.04s`.
- `jump`: task return `20.85`, fall rate `0.125`, survival `9.08s`.
- `strafe_right`: task return `25.71`, fall rate `0.094`, survival `9.63s`.

Transition-zone failure diagnostics:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics/termination_reasons.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics/state_trace.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics/failure_diagnostics_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics/failure_diagnostics_summary.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the `0.95` full7 run.
- Presets: `stand`, `run`, `jump`, and `strafe_right`, because these are the visible weak presets at `0.95`.
- Flags: `--eval_disturb_ratio=0.95`, `--record_termination_reasons`, and `--record_state_trace`.

Failure summary:

| preset | task return | fall rate | contact `base_link` rate | orientation rate | timeout rate | contact mean survival s | terminal base z | terminal mean abs roll | terminal mean abs pitch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stand` | 11.21 | 0.219 | 0.188 | 0.031 | 0.781 | 3.37 | 0.682 | 0.164 | 0.221 |
| `run` | 10.88 | 0.125 | 0.062 | 0.062 | 0.875 | 1.85 | 0.721 | 0.119 | 0.174 |
| `jump` | 19.43 | 0.125 | 0.125 | 0.000 | 0.875 | 3.47 | 0.707 | 0.116 | 0.127 |
| `strafe_right` | 28.85 | 0.062 | 0.062 | 0.000 | 0.938 | 4.67 | 0.734 | 0.054 | 0.139 |

Interpretation: the `0.95` transition zone expands the same contact-led failure mode seen at `0.925`: `stand` contact rises from `0.156` to `0.188`, `jump` contact rises from `0.062` to `0.125`, and `strafe_right` begins to show low-rate contact failure. `run` adds a secondary roll/pitch failure path (`0.062`) that is absent at `0.925`. Terminal base height remains `0.68-0.73`, so this still differs from the `1.0` collapse regime where terminal base height drops to `0.54-0.58` and contact rates exceed `0.4-0.5`.

Disturbance boundary comparison:

| metric | forced 0.75 | forced 0.9 | forced 0.925 | forced 0.95 | forced 1.0 |
|---|---:|---:|---:|---:|---:|
| avg task return | 27.69 | 25.47 | 22.18 | 19.46 | 6.03 |
| avg fall rate | 0.018 | 0.029 | 0.054 | 0.098 | 0.444 |
| avg survival s | 9.93 | 9.82 | 9.68 | 9.35 | 6.55 |

Delta versus forced `0.75`: avg task return `-2.21`, avg fall rate `+0.011`, avg survival `-0.12s`. Forced `0.925` is the first tested setting with clear onset degradation: compared with `0.9`, avg task return drops `-3.29`, avg fall rate rises `+0.025`, and avg survival falls `-0.14s`; the largest task-return drop is `walk_slow` (`-5.80`) and the largest fall-rate increase is `stand` (`+0.141`). Forced `0.95` deepens that degradation: compared with `0.925`, avg task return drops another `-2.72`, avg fall rate rises `+0.045`, and avg survival falls `-0.33s`. Delta from forced `0.95` to forced `1.0` remains much larger: avg task return `-13.43`, avg fall rate `+0.346`, avg survival `-2.80s`, with the largest drop and fall-rate increase both on `jump` (`-20.29`, `+0.422`). This places onset around `0.925`, transition around `0.95`, and outright collapse at `1.0`, so `0.75` remains the safe training cap and `0.9/0.925/0.95` are better reserved for post-training stress tests.

Boundary analysis aggregate:

```text
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis/boundary_ratio_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis/boundary_preset_metrics.csv
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis/boundary_failure_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis/boundary_adjacent_deltas.csv
outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis/boundary_analysis_summary.json
```

Aggregate files:

- `boundary_ratio_summary.csv`: 5 rows for forced ratios `0.75`, `0.90`, `0.925`, `0.95`, and `1.00`.
- `boundary_preset_metrics.csv`: 35 rows covering all seven fixed presets across those five ratios.
- `boundary_failure_summary.csv`: 13 rows covering focused failure diagnostics for `0.90`, `0.925`, `0.95`, and `1.00`.
- `boundary_adjacent_deltas.csv`: adjacent transitions `0.75 -> 0.90`, `0.90 -> 0.925`, `0.925 -> 0.95`, and `0.95 -> 1.00`.
- `boundary_analysis_summary.json`: machine-readable summary with the current interpretation: safe training cap `0.75`, stable stress test `0.90`, onset `0.925`, transition `0.95`, collapse `1.00`, primary failure mode `base_link` contact with roll/pitch as a secondary high-ratio path.

Full seven-preset full-disturbance old-checkpoint diagnostic:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100/full7_disturb100_summary.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100/full7_disturb100_vs_baseline_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100/full7_disturb100_vs_baseline_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100/full7_disturb100_vs_disturb075_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100/full7_disturb100_vs_disturb075_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb100_vs_disturb0925_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925/full7_disturb100_vs_disturb0925_delta.json
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb100_vs_disturb095_delta.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095/full7_disturb100_vs_disturb095_delta.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the forced `0.75` run, but with `--eval_disturb_ratio=1.0`.
- Purpose: directly test whether full-disturb pressure is safe enough to train against, or whether the conservative `0.75` cap is a necessary boundary.

Diagnostic result:

| rows | avg task return | avg fall rate | avg survival s | worst task preset | worst return | worst fall preset | worst fall rate |
|---:|---:|---:|---:|---|---:|---|---:|
| 7 | 6.03 | 0.444 | 6.55 | `stand` | -1.12 | `jump` | 0.547 |

Selected per-preset facts:

- `stand`: task return `-1.12`, fall rate `0.531`, survival `5.93s`.
- `walk_slow`: task return `12.04`, fall rate `0.391`, survival `7.17s`.
- `run`: task return `-0.94`, fall rate `0.484`, survival `6.00s`.
- `jump`: task return `0.56`, fall rate `0.547`, survival `5.84s`.
- `strafe_right`: task return `15.76`, fall rate `0.359`, survival `7.27s`.

Delta versus forced `0.75`:

| metric | forced 0.75 | forced 0.9 | forced 0.925 | forced 0.95 | forced 1.0 | delta 0.95 -> 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| avg task return | 27.69 | 25.47 | 22.18 | 19.46 | 6.03 | -13.43 |
| avg fall rate | 0.018 | 0.029 | 0.054 | 0.098 | 0.444 | +0.346 |
| avg survival s | 9.93 | 9.82 | 9.68 | 9.35 | 6.55 | -2.80 |

Largest `0.75 -> 1.0` local changes:

- Largest task-return drop versus forced `0.95`: `jump`, `-20.29`.
- Largest fall-rate increase versus forced `0.95`: `jump`, `+0.422`.

Full-disturb failure diagnostics:

```text
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics/termination_reasons.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics/state_trace.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics/failure_diagnostics_summary.csv
outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics/failure_diagnostics_summary.json
```

Protocol:

- Same WSL CPU / Jun17 selective-walk best checkpoint / new follow-up config protocol as the full-disturb run.
- Presets: `stand`, `run`, and `jump`, because these are the lowest-return / highest-fall presets under full-disturb `1.0`.
- Flags: `--eval_disturb_ratio=1.0`, `--record_termination_reasons`, and `--record_state_trace`.

Failure summary:

| preset | task return | fall rate | contact `base_link` rate | orientation rate | timeout rate | contact mean survival s | terminal base z | terminal mean abs roll | terminal mean abs pitch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stand` | -1.12 | 0.531 | 0.438 | 0.094 | 0.469 | 2.47 | 0.576 | 0.416 | 0.350 |
| `run` | -4.52 | 0.625 | 0.547 | 0.078 | 0.375 | 1.92 | 0.539 | 0.490 | 0.438 |
| `jump` | 3.23 | 0.516 | 0.500 | 0.016 | 0.484 | 2.07 | 0.562 | 0.311 | 0.428 |

Interpretation: forced `0.75` disturbance does not expose a new collapse mode outside `jump/run`; relative to the no-forced-disturbance full7 baseline, the average task return is nearly unchanged and average fall rate is lower. Forced `0.9` has measurable degradation but remains stable by survival and fall-rate evidence. Forced `0.925` is the onset point: fall rate rises to `0.054`, with `stand` already at `0.188`. Forced `0.95` is the transition zone: fall rate rises to `0.098` and `stand/run/jump/strafe_right` all show visible weakness, but survival remains above `9s`; focused diagnostics show contact failures spreading and `run` adding a secondary roll/pitch failure path. Full-disturb `1.0` then causes broad collapse across all seven presets. The focused diagnostics show that the full-disturb collapse is dominated by early `base_link` contact, with terminal base height around `0.54-0.58` versus the configured `0.78` target, and roll/pitch orientation failures as a secondary mode. This keeps the `0.75` cap as the safest next training boundary, while `0.9`, `0.925`, and `0.95` become useful later stress-test targets after a trained follow-up checkpoint exists.

Reason this is the next code/config change:

- Jun20 `conservative_penalty_top_5818` is best by no-disturb aggregate (`30.28`), but at `jump` ratio `0.75` it falls to `task_return=7.30`, `fall_rate=0.453`.
- Jun17 selective-walk best is slightly lower by no-disturb aggregate (`29.79`) but is much stronger under partial disturbance: `jump` ratio `0.75` has `task_return=30.55`, `fall_rate=0.016`; `run` ratio `0.75` has `task_return=11.57`, `fall_rate=0.016`.
- Therefore the next training run should privilege robustness under partial `jump/run` disturbance over a small no-disturb aggregate advantage.

Local candidate decision audit:

```text
outputs/eval/June30_r2_amp_candidate_decision_audit
outputs/eval/June30_r2_amp_candidate_decision_audit/candidate_decision_audit.csv
outputs/eval/June30_r2_amp_candidate_decision_audit/focused_disturbance_risk_audit.csv
outputs/eval/June30_r2_amp_candidate_decision_audit/candidate_decision_audit_summary.json
```

Protocol:

- This is a no-training / no-new-rollout audit over existing summary CSVs only.
- Inputs are the Jun17, Jun19, Jun20, Jun21, Jun23, Jun24, and Jun25_0 top-task summary tables plus focused disturbance summaries for the leading candidates.
- The conservative decision score is an explicit heuristic for ranking candidates when no-disturb task return is close: `avg_task_return - 40*avg_fall_rate + 0.5*avg_survival_s + 0.1*worst_task_return - 20*worst_fall_rate`. It is not a new training reward; it is only an audit lens that penalizes fall rate and weak worst-preset behavior.

Key facts from `candidate_decision_audit_summary.json`:

| item | result |
|---|---|
| candidate rows | `90` |
| missing inputs | `0` |
| best by no-disturb task return | `Jun20:conservative_penalty_top_5818:5818` |
| best by conservative decision score | `Jun17:selective_walk_best:best` |
| Jun25_0 best row by conservative score | `conservative_8000_corrected`, task rank `2`, decision rank `9` |

Top rows by conservative decision score:

| decision rank | candidate | task rank | avg task return | avg fall rate | avg survival s | worst task preset | worst task return | worst fall rate |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 1 | `Jun17:selective_walk_best:best` | 3 | 29.79 | 0.036 | 9.78 | `run` | 20.03 | 0.141 |
| 2 | `Jun20:conservative_penalty_top_5818:5818` | 1 | 30.28 | 0.062 | 9.63 | `run` | 11.41 | 0.172 |
| 3 | `Jun21:conservative_penalty_top_7654:7654` | 6 | 29.16 | 0.054 | 9.59 | `run` | 7.06 | 0.156 |
| 9 | `Jun25_0:conservative_8000_corrected:8000` | 2 | 30.02 | 0.167 | 8.86 | `jump` | 20.86 | 0.391 |

Interpretation: this cross-audit supports the previous qualitative decision. Jun20 conservative top remains the pure no-disturb task-return winner, but the selective-walk best checkpoint remains the stronger warm-start candidate once fall rate, survival, worst-preset return, and focused disturbance evidence are treated as first-class constraints. Jun25_0 conservative final is a strong no-disturb checkpoint but carries higher fall risk (`0.167` average and `0.391` worst-preset fall rate), so it should remain evaluated evidence rather than replacing the selective-walk follow-up plan.

### Local Evaluation Coverage Audit - 2026-06-30

Hypothesis: after the Jun19 top-task pass and archive compatibility evaluations, no additional no-training checkpoint evaluation should remain for the current R2 AMP ranking unless the filesystem contains a loadable checkpoint that lacks a `metrics.csv` output.

Audit commands used current filesystem state under `logs/r2_amp` and `outputs/eval`, excluding generated compatibility hard-link directories from the source checkpoint count. The audit scope is:

- Current experimental line: Jun17, Jun19, Jun20, Jun21, Jun23, Jun24, and Jun25_0 model-bearing runs.
- Archive compatibility line: normalized Jun10/Jun15 checkpoints under `logs/r2_amp/_archive_eval_compat`.
- Evaluation target types: `model_best_task.pt` / `model_best_mixed.pt`, final checkpoints used by the experiment record (`model_8000.pt` or `model_30000.pt`), saved `model_top_task_*.pt`, and focused disturbance/state-trace diagnostics for the leading candidates.
- Routine periodic autosaves such as `model_0.pt`, `model_2000.pt`, `model_4000.pt`, and `model_6000.pt` are not treated as separate ranking targets unless they are also a best/final/top-task checkpoint. They are training autosaves, not selected candidates in the maintained experiment protocol.

Current-line checkpoint inventory:

| group | run groups | top-task files | evaluated top-task files | invalid / excluded top-task files |
|---|---:|---:|---:|---:|
| Jun17 | 3 | 9 | 7 | 2 |
| Jun19 | 4 | 12 | 12 | 0 |
| Jun20 | 4 | 12 | 12 | 0 |
| Jun21 | 2 | 6 | 6 | 0 |
| Jun23 | 2 | 6 | 6 | 0 |
| Jun24 | 2 | 6 | 6 | 0 |
| Jun25_0 | 2 | 6 | 6 | 0 |
| total | 19 | 57 | 55 | 2 |

Invalid top-task files rechecked in the current WSL `r2gym` environment:

| checkpoint | load result |
|---|---|
| `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_top_task_1461.pt` | `torch.load(..., map_location="cpu")` fails with `UnpicklingError: invalid load key, '#'` |
| `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_top_task_1464.pt` | `torch.load(..., map_location="cpu")` fails with `UnpicklingError: invalid load key, 'H'` |

Archive compatibility coverage:

| archive source | target checkpoints | 7-preset outputs |
|---|---:|---:|
| `logs/r2_amp/_archive_eval_compat/Jun10_mixed` | 2 | 2 |
| `logs/r2_amp/_archive_eval_compat/Jun10_mixed2` | 1 | 1 |
| `logs/r2_amp/_archive_eval_compat/Jun10_style0` | 2 | 2 |
| `logs/r2_amp/_archive_eval_compat/Jun10_sw1` | 2 | 2 |
| `logs/r2_amp/_archive_eval_compat/Jun10_walk` | 1 | 1 |
| `logs/r2_amp/_archive_eval_compat/Jun15_sw05` | 2 | 2 |
| `logs/r2_amp/_archive_eval_compat/Jun15_sw1` | 2 | 2 |
| total | 12 | 12 |

Output-shape facts:

- Current `outputs/eval/June30_Jun*` top-task/focused outputs checked in this audit have valid `metrics.csv` row counts: full fixed-preset evaluations have `7` rows; focused single-preset diagnostics have `1` or paired focused summaries have `2` rows.
- No `June30_Jun*` output had an unexpected row count outside `{1, 2, 7}` in the audit.
- The two `June30_archive_smoke_*` outputs have one preset each and remain smoke probes; the archive coverage table above uses the 12 full 7-preset archive outputs as the ranking evidence.
- The top-level `logs/r2_amp/model*.pt` artifacts were rechecked after the follow-up smoke runs. Seven are SHA256 duplicates of already evaluated Jun24 run artifacts; the remaining `logs/r2_amp/model_top_task_1518.pt` is invalid because `torch.load(..., map_location="cpu")` fails with `UnpicklingError: invalid load key, '#'`. The detailed mapping is saved at `outputs/eval/June30_r2_amp_checkpoint_eval_coverage/top_level_model_artifacts.csv`.
- The documented output-path audit extracted 214 concrete-or-pattern `outputs/eval/...` references from this progress document. It found 213 concrete paths, all present on disk, and one intentional wildcard prose reference (`outputs/eval/June30_Jun*`). The ten referenced directories without a top-level `metrics.csv` are expected aggregate/probe directories: they contain `summary_metrics.csv`, `run_disturb_sweep_summary.csv`, `joint_limit_probe.csv`, boundary-analysis CSV/JSON files, candidate-decision/readiness audit CSV/JSON files, or per-ratio child directories with metrics.

Conclusion:

- The current no-training evaluation coverage pass is closed for the maintained R2 AMP ranking scope: all loadable current-line top-task checkpoints have fixed-preset metrics, all current best/final candidates used in the record have metrics, all leading candidates have focused `jump/run` disturbance/state-trace diagnostics, and normalized Jun10/Jun15 archive targets have 7-preset metrics.
- The only current source files without top-task metrics are verified duplicate or non-checkpoint artifacts and should remain excluded rather than forced through `evaluate.py`.
- The remaining work is not another local evaluation run; it is the not-yet-trained follow-up config above, which requires training budget before it can produce new checkpoints to evaluate.

Current open-item ledger after this audit (historical June30 state; the selective-walk follow-up rows are superseded by the July01 formal-run evaluation below):

| item | status after audit | reason |
|---|---|---|
| Old `pending` run-disturb sweep note for `scratch_command_hold/model_8000.pt` | closed | `outputs/eval/run_disturb_sweep_command_hold_8000` exists and is summarized above. |
| Old request for Jun25_0 reward/termination/state-trace diagnostics | closed | Corrected reward/termination/state-trace and focused disturbance outputs exist for Jun25_0 conservative `8000`. |
| Old request for Jun17 selective-walk robustness diagnostics | closed | `outputs/eval/June30_Jun17_selective_walk_best_disturb_sweep`, corrected state trace, and full-disturb state trace exist and are summarized above. |
| Old request for backward top-task coverage through Jun23/Jun21/Jun20/Jun19 | closed | Jun23, Jun21, Jun20, and Jun19 top-task sections now exist with 7-preset outputs. |
| Old request for archival pre-Jun17 compatibility evaluation | closed for normalized Jun10/Jun15 scope | `_archive_eval_compat` contains 12 normalized targets and all 12 have 7-preset outputs. |
| Invalid Jun17 selective-walk `model_top_task_1461.pt` / `1464.pt` | excluded | Current WSL `torch.load` recheck fails with `UnpicklingError`; these are not valid checkpoint targets. |
| Top-level `logs/r2_amp/model*.pt` artifacts | closed / excluded | Seven files are SHA256 duplicates of already evaluated Jun24 artifacts; `logs/r2_amp/model_top_task_1518.pt` fails current WSL `torch.load` with `UnpicklingError: invalid load key, '#'`. |
| Documented `outputs/eval/...` references | closed | `documented_eval_output_paths_summary.json` reports 213 concrete documented paths and 0 missing concrete paths; the only non-concrete reference is the intentional wildcard prose `outputs/eval/June30_Jun*`. |
| Follow-up readiness audit | pending training | `outputs/eval/June30_selective_walk_followup_readiness_audit/readiness_audit.json` finds 15 matching transient run directories, all classified as `load_only_no_training_progress` with 265-byte loading-only `train.log` files, and all 15 carry `artifact_source="evaluate_checkpoint_load_log_dir"`; there are still 0 checkpoints. These are historical `evaluate.py` checkpoint-load artifacts from the old default runner `log_root`, not trained runs. `evaluate.py` now uses `log_root=None` for checkpoint evaluation, and the regression smoke at `outputs/eval/June30_evaluate_log_root_none_smoke` confirmed checkpoint loading still works while the matching transient-directory count remains 15. The audit exposes `recommended_checkpoint`, `recommended_load_run`, and `recommended_eval_plan`; current values are `null`, `null`, and an empty list because no real checkpoint exists. `scripts/plan_selective_walk_followup_train.py` prints the reviewed formal warm-start command with `--max_iterations=4000`, because the source checkpoint is `iter=4000` and the target is `model_8000.pt`; the long formal training job has not been launched in this local no-large-training pass. `scripts/run_selective_walk_followup_eval_plan.py` consumes the audit JSON; current dry-run prints `No recommended eval commands; run readiness audit after a real checkpoint exists.`, and `--execute` refuses for the same reason. Therefore `ready_for_evaluation=false` and `ready_for_completion=false`. |
| `selective_walk_eval_manifold_conservative_disturb_release.json` | config/checkpoint smoke, old-checkpoint full7 baseline, and formal-size forced-disturb diagnostics passed; `not trained` | The CPU smokes at `outputs/eval/June30_selective_walk_conservative_followup_smoke_jump`, `outputs/eval/June30_selective_walk_conservative_followup_smoke_run`, `outputs/eval/June30_selective_walk_conservative_followup_smoke_jump_disturb075`, and `outputs/eval/June30_selective_walk_conservative_followup_smoke_run_disturb075` prove the config and Jun17 best checkpoint load together for both key presets at no forced disturbance and forced `0.75` disturbance. The full-shape baseline at `outputs/eval/June30_selective_walk_conservative_followup_baseline_full7` adds a 7-preset / 64-episode compatibility baseline for the old warm-start checkpoint; `outputs/eval/June30_selective_walk_conservative_followup_jump_run_disturb075_formal` adds 64-episode `jump/run` diagnostics at forced `0.75`; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb075` extends forced `0.75` to all seven fixed presets; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090` shows the same checkpoint is still broadly stable at forced `0.9` and `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb090_failure_diagnostics` confirms low-rate failures with timeout rates above `0.92`; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925` identifies the onset of degradation and `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb0925_failure_diagnostics` attributes onset mainly to low-rate `base_link` contact; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095` identifies the transition zone and `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb095_failure_diagnostics` shows contact spreading plus a secondary `run` orientation path; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100` shows full-disturb `1.0` broadly collapses the same old checkpoint; `outputs/eval/June30_selective_walk_conservative_followup_full7_disturb100_failure_diagnostics` attributes the worst-preset collapse mainly to early `base_link` contact; `outputs/eval/June30_selective_walk_conservative_followup_boundary_analysis` aggregates the ratio, preset, failure, and adjacent-delta evidence into machine-readable CSV/JSON tables. Formal follow-up evaluation still must wait until a training run creates follow-up checkpoints. |

### July01 Formal Selective-Walk Follow-up Training Evaluation

Hypothesis: the selective-walk prior plus eval-manifold command mixture and conservative staged disturbance cap would preserve the Jun17 warm-start checkpoint's partial-disturbance robustness while improving the seven-preset command manifold.

Training artifact:

```text
logs/r2_amp/Jun30_17-05-30_selective_walk_eval_manifold_conservative_disturb_release
configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json
```

Checkpoint facts:

- The run contains `model_4000.pt`, `model_6000.pt`, `model_8000.pt`, `model_10000.pt`, `model_12000.pt`, `model_best_task.pt`, and top-task checkpoints `model_top_task_4088.pt`, `model_top_task_4093.pt`, `model_top_task_4100.pt`.
- This is a real trained run, not one of the older evaluate-only transient directories: `outputs/eval/July01_selective_walk_followup_readiness_audit.json` reports `runs_found=1`, `checkpoint_count=9`, `ready_for_evaluation=true`, and `recommended_checkpoint=8000`.
- The run overshot the intended formal target: the reviewed handoff expected an additional `--max_iterations=4000` from source iteration `4000` to target `model_8000.pt`, but this run produced `model_12000.pt` and `train.log` reaches `Learning iteration 11999/12000`. This means the effective resume budget behaved like an additional `8000` iterations.
- Late training never released staged disturbance: the tail of `train.log` keeps `Mean episode staged_disturb_level: 0.0000`; the final iteration reports `Mean task reward: -4.50`, `staged_disturb_window_task_return: -16.7981`, `staged_disturb_window_fall_rate: 0.1442`, and `Best task reward: 37.14`.

Evaluation protocol:

- WSL CPU PhysX / CPU policy eval through `legged_gym/scripts/evaluate.py`.
- Task/config: `--task=r2amp`, `--cfg_override_json configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json`.
- Default seven fixed presets, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- `model_8000.pt` was also evaluated at forced `--eval_disturb_ratio=0.75`.
- The planned 9-output eval was intentionally stopped after the `8000` baseline and `0.75` full7 evaluation, because those already show broad collapse. Attempting to continue the original batch to `0.9` failed with WSL `exit status 137` after memory/swap exhaustion; WSL was restarted before the subsequent focused baseline checks.

Evaluation outputs:

```text
outputs/eval/July01_selective_walk_followup_best_task_baseline_full7
outputs/eval/July01_selective_walk_followup_baseline_full7
outputs/eval/July01_selective_walk_followup_full7_disturb075
outputs/eval/July01_selective_walk_followup_12000_baseline_full7
outputs/eval/July01_selective_walk_followup_summary
```

Aggregate result:

| checkpoint / reference | protocol | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | action-rate L2 | worst task preset | worst task return | worst fall preset | worst fall rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| Jun17 warm-start reference | baseline full7 | 7 | 28.03 | 0.038 | 9.76 | 0.339 | 0.485 | 2.59 | `run` | 11.80 | `stand` | 0.109 |
| `model_best_task.pt` | baseline full7 | 7 | 22.78 | 0.078 | 9.43 | 0.404 | 0.580 | 2.70 | `run` | 0.46 | `run` | 0.281 |
| `model_8000.pt` | baseline full7 | 7 | 14.01 | 0.406 | 6.45 | 0.432 | 0.631 | 5.77 | `jump` | -3.76 | `stand` | 1.000 |
| `model_8000.pt` | forced disturb `0.75` full7 | 7 | 7.97 | 0.493 | 6.25 | 0.700 | 0.815 | 4.11 | `jump` | -9.71 | `stand` | 1.000 |
| `model_12000.pt` | baseline full7 | 7 | 21.66 | 0.330 | 7.20 | 0.426 | 0.697 | 18.47 | `run` | -6.18 | `run` | 1.000 |

Selected per-preset facts:

- `model_best_task.pt` is already weaker than the Jun17 warm-start reference: `run` drops to `task_return=0.46`, `fall_rate=0.281`, `survival=7.64s`.
- `model_8000.pt` has broad no-forced-disturbance collapse: `stand` has `task_return=-1.31`, `fall_rate=1.000`, `survival=1.90s`; `jump` has `task_return=-3.76`, `fall_rate=1.000`, `survival=0.27s`.
- `model_8000.pt` under forced `0.75` disturbance is worse, not more robust: `stand` remains `fall_rate=1.000`, `jump` remains `fall_rate=1.000`, and average task return falls to `7.97`.
- `model_12000.pt` partially recovers stand/walk/turn/strafe task return, but it is not a usable recovery checkpoint: `run` and `jump` both have `fall_rate=1.000`, and action-rate L2 rises sharply to `18.47`.

Interpretation:

- This formal follow-up did not validate the hypothesis. It underperforms the old Jun17 warm-start checkpoint before forced disturbance, and the planned target `model_8000.pt` fails the basic full7 baseline.
- The failure is not caused by the staged disturbance cap being too aggressive in late training, because the training log reports `staged_disturb_level=0.0000` at the tail. The supported diagnosis is that the fine-tuning objective/command-profile mixture already destabilizes the warm-start policy before staged disturbance ever becomes active.
- The `model_12000.pt` overshoot should not be used as a candidate: it recovers some easy presets but catastrophically fails `run` and `jump` and has much higher action-rate L2.

Decision:

- Do not continue from `model_8000.pt` or `model_12000.pt`.
- Keep the Jun17 selective-walk best checkpoint as the stronger robustness reference.
- The next experiment should be a controlled warm-start fine-tune from the Jun17 checkpoint with a less destabilizing training contract before reintroducing the full eval-manifold disturbance release. The probe budget was later standardized to `--max_iterations=8000` for all three July01 retention configs; because the source checkpoint is internally `iter=4000`, the main terminal checkpoint is expected around `model_12000.pt`.

### July01 Warm-Start Retention Probe Configs

Hypothesis: the Jun30/July01 follow-up failed before staged disturbance release, so the next evidence should isolate whether resume itself, the seven-profile command mixture, or forgetting during fine-tune is the destabilizing factor.

Training decision:

- Use warm-start from `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`.
- Do not train from scratch for this batch. From-scratch training would test a different question and would not isolate preservation of the already-strong Jun17 policy.
- Run three matched 8000-additional-iteration probes. If the null control degrades, the resume/training budget is suspect; if only the profile probes degrade, the command-profile/task objective is suspect; if teacher retention helps, the failure is consistent with fine-tune forgetting. With the Jun17 source checkpoint at internal `iter=4000`, this budget should produce a terminal checkpoint around `model_12000.pt`.

Reference basis:

- PPO remains the optimizer backbone, following the clipped surrogate / multi-epoch minibatch update design in Schulman et al. 2017, `arXiv:1707.06347`.
- The new retention term is a Learning without Forgetting-style preservation constraint, following the idea of using new-task data while preserving old behavior from Li and Hoiem 2016, `arXiv:1606.09282`. In this repo the preserved output is the action mean of the loaded warm-start policy on current rollout mini-batches.
- The failure mode being tested is catastrophic forgetting during sequential fine-tuning; Kirkpatrick et al. 2017, `PNAS 114(13):3521-3526`, is the reference motivation for treating old-skill preservation as a first-class constraint.

Implemented code/config artifacts:

| artifact | status | purpose |
|---|---|---|
| `legged_gym/envs/base/legged_robot_config.py` | implemented | Adds `algorithm.teacher_policy_retention_coef=0.0` to the base PPO config schema so `cfg_override_json` accepts the retention probe JSONs; default 0.0 keeps existing PPO/AMP runs unchanged. |
| `rsl_rl/rsl_rl/algorithms/ppo.py` | implemented | Adds `teacher_policy_retention_coef`, `capture_teacher_policy()`, `_teacher_retention_loss()`, and logs `teacher_policy_retention_loss` / `teacher_policy_retention_skipped`. |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | implemented | Calls `capture_teacher_policy()` immediately after checkpoint `load()`, so the teacher is the loaded Jun17 warm-start policy. |
| `configs/ablation/selective_walk_resume_null_control.json` | trained/evaluated in the Jul01 probe batch | 8000-additional-iteration warm-start null control: selective-walk AMP routing, no `profile_mixture`, no teacher retention. |
| `configs/ablation/selective_walk_profile_task_only_probe.json` | trained/evaluated in the Jul01 probe batch | 8000-additional-iteration warm-start profile probe: seven eval-like `profile_mixture`, `train.amp.style_reward_weight=0.0`, no teacher retention. |
| `configs/ablation/selective_walk_profile_teacher_retention_probe.json` | trained/evaluated in the Jul01 probe batch | Same task-only profile probe plus `teacher_policy_retention_coef=0.25` to test whether teacher retention reduces early fine-tune forgetting. |
| `tests/test_amp_training_contracts.py` | implemented | Contract coverage for teacher-retention hooks, the three 8000-iteration probe JSONs, and the base PPO config schema fields needed by strict JSON merge. |

Training-machine failure note:

- A launch that only had the JSON side of this change failed before environment creation with `AttributeError: Unknown config field 'train.algorithm.teacher_policy_retention_coef'`.
- Root cause: `cfg_override_json` uses strict recursive merge and rejects any key not declared on the config object. `PPO.__init__` already accepted `teacher_policy_retention_coef`, but the base train config schema did not yet expose `algorithm.teacher_policy_retention_coef`.
- Required fix on the training machine: sync the commit that adds `LeggedRobotCfgPPO.algorithm.teacher_policy_retention_coef = 0.0`, or manually add that one default field before launching the retention JSONs.

Recommended training commands:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk \
  --checkpoint=-2 \
  --cfg_override_json configs/ablation/selective_walk_resume_null_control.json \
  --run_name selective_walk_resume_null_control \
  --max_iterations=8000
```

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk \
  --checkpoint=-2 \
  --cfg_override_json configs/ablation/selective_walk_profile_task_only_probe.json \
  --run_name selective_walk_profile_task_only_probe \
  --max_iterations=8000
```

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk \
  --checkpoint=-2 \
  --cfg_override_json configs/ablation/selective_walk_profile_teacher_retention_probe.json \
  --run_name selective_walk_profile_teacher_retention_probe \
  --max_iterations=8000
```

Evaluation plan after each 8000-iteration probe:

- First evaluate `model_best_task.pt` and the expected terminal resumed checkpoint around `model_12000.pt` under the no-forced-disturbance full7 protocol.
- Only if a probe preserves no-disturb full7 performance should it receive forced `0.75` full7 disturbance evaluation.
- Stop the batch early if the null control already falls below the Jun17 warm-start reference (`avg task return 28.03`, `avg fall rate 0.038`) by a large margin; that would make the profile/retention comparison ambiguous until the resume budget and PPO learning-rate settings are rechecked.

### July05 Evaluation of July01 Warm-Start Retention Probes

Hypothesis: if the Jun30/July01 follow-up failed because the profile fine-tune forgot the Jun17 warm-start behavior, then the teacher-retention profile probe should preserve or improve the no-disturb seven-preset baseline better than the task-only profile probe.

Training artifacts:

| experiment | config | run directory | evaluated checkpoints | status |
|---|---|---|---|---|
| `selective_walk_resume_null_control` | `configs/ablation/selective_walk_resume_null_control.json` | `logs/r2_amp/Jul01/Jul01_15-06-31_selective_walk_resume_null_control` | `model_best_task.pt`, `model_12000.pt` | evaluated |
| `selective_walk_profile_teacher_retention_probe` | `configs/ablation/selective_walk_profile_teacher_retention_probe.json` | `logs/r2_amp/Jul01/Jul01_15-14-48_selective_walk_profile_teacher_retention_probe` | `model_best_task.pt`, `model_12000.pt`; `model_12000.pt` also evaluated at forced `0.75` disturbance | evaluated |
| `selective_walk_profile_task_only_probe` | `configs/ablation/selective_walk_profile_task_only_probe.json` | `logs/r2_amp/Jul01/Jul02_07-06-46_selective_walk_profile_task_only_probe` | `model_best_task.pt`, `model_12000.pt` | evaluated |

Evaluation protocol:

- WSL CPU PhysX / CPU policy via `legged_gym/scripts/evaluate.py`.
- `--task=r2amp`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default seven fixed presets; DTW was not enabled.
- Forced-disturbance follow-up was run only for the strongest no-disturb candidate, `selective_walk_profile_teacher_retention_probe/model_12000.pt`, with `--eval_disturb_ratio=0.75`.

Evaluation outputs:

```text
outputs/eval/July01_selective_walk_resume_null_control_best_task_baseline_full7
outputs/eval/July01_selective_walk_resume_null_control_12000_baseline_full7
outputs/eval/July01_selective_walk_profile_teacher_retention_probe_best_task_baseline_full7
outputs/eval/July01_selective_walk_profile_teacher_retention_probe_12000_baseline_full7
outputs/eval/July01_selective_walk_profile_teacher_retention_probe_12000_full7_disturb075
outputs/eval/July01_selective_walk_profile_task_only_probe_best_task_baseline_full7
outputs/eval/July01_selective_walk_profile_task_only_probe_12000_baseline_full7
outputs/eval/July01_selective_walk_probe_summary
```

Aggregate result:

| checkpoint / protocol | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | action-rate L2 | worst task preset | worst task return | worst fall preset | worst fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `resume_null_control/model_best_task.pt`, baseline full7 | 7 | 28.97 | 0.025 | 9.84 | 0.317 | 0.450 | 2.61 | `run` | 14.01 | `run` | 0.047 |
| `resume_null_control/model_12000.pt`, baseline full7 | 7 | 26.01 | 0.103 | 9.13 | 0.355 | 0.519 | 7.30 | `run` | 4.20 | `run` | 0.484 |
| `profile_teacher_retention/model_best_task.pt`, baseline full7 | 7 | 28.72 | 0.042 | 9.77 | 0.313 | 0.466 | 2.71 | `run` | 16.58 | `stand` | 0.156 |
| `profile_teacher_retention/model_12000.pt`, baseline full7 | 7 | 33.46 | 0.022 | 9.89 | 0.277 | 0.370 | 2.37 | `run` | 21.67 | `stand` | 0.078 |
| `profile_teacher_retention/model_12000.pt`, forced `0.75` full7 | 7 | 31.83 | 0.029 | 9.83 | 0.334 | 0.415 | 2.56 | `run` | 18.11 | `run` | 0.078 |
| `profile_task_only/model_best_task.pt`, baseline full7 | 7 | 27.64 | 0.074 | 9.55 | 0.332 | 0.477 | 2.80 | `run` | 18.45 | `stand` | 0.281 |
| `profile_task_only/model_12000.pt`, baseline full7 | 7 | -9.93 | 1.000 | 0.05 | 0.861 | 2.801 | 5100.84 | `strafe_right` | -22.77 | `jump` | 1.000 |

Selected per-preset facts:

- `profile_teacher_retention/model_12000.pt` under forced `0.75` disturbance remained usable across all seven presets: the worst task preset was `run` with `task_return=18.11`, `fall_rate=0.078`, and `survival=9.41s`; `stand` also had `fall_rate=0.078`.
- `profile_task_only/model_12000.pt` collapsed across the entire command manifold: all seven presets had `fall_rate=1.000`, with survival between `0.04s` and `0.09s`.
- `resume_null_control/model_best_task.pt` stayed close to or slightly above the Jun17 warm-start reference (`28.03` avg task return, `0.038` fall rate), but `resume_null_control/model_12000.pt` regressed mainly on `run` (`fall_rate=0.484`).

Interpretation:

- The teacher-retention probe validates the forgetting hypothesis better than the task-only profile probe. Adding `teacher_policy_retention_coef=0.25` preserved the warm-start behavior while still allowing the seven-profile objective to improve the no-disturb full7 aggregate.
- `selective_walk_profile_teacher_retention_probe/model_12000.pt` is the current best candidate in this branch because it improves over the Jun17 warm-start reference under no-disturb evaluation and remains stable under forced `0.75` full7 disturbance.
- The task-only profile objective without teacher retention is not safe at the 12000 resumed checkpoint. Its late collapse is stronger evidence for fine-tune forgetting than for staged-disturbance pressure, because these probes still show `staged_disturb_level=0.0000` in the training-tail logs.
- The null control shows that resume alone can keep a good early checkpoint, but the terminal checkpoint still drifts on `run`; therefore the next useful comparison is not more task-only training, but focused robustness diagnostics and possibly a retention-coefficient sweep around the teacher-retention candidate.

### July05 Next Warm-Start Probe Configs

Hypothesis: after `selective_walk_profile_teacher_retention_probe/model_12000.pt` improved the no-disturb full7 aggregate and stayed stable under forced `0.75` disturbance, the next two experiments should separately test teacher-retention strength and whether a retention-stabilized policy can learn a staged disturbance curriculum during training.

Implemented code/config artifacts:

| artifact | status | purpose |
|---|---|---|
| `configs/ablation/selective_walk_profile_teacher_retention_coef010_probe.json` | trained/evaluated in the Jul05 probe batch | Same seven-profile warm-start probe as the successful `0.25` teacher-retention run, but lowers `teacher_policy_retention_coef` to `0.10` and keeps staged disturbance at `[0.0]`; tests whether weaker Learning without Forgetting action-mean retention is sufficient. |
| `configs/ablation/selective_walk_profile_teacher_retention_disturb075_probe.json` | trained/evaluated in the Jul05 probe batch | Keeps `teacher_policy_retention_coef=0.25`, `style_reward_weight=0.0`, and the same seven profiles, but releases staged disturbance through `0.0 -> 0.15 -> 0.3 -> 0.5 -> 0.75`; tests whether the retention-stabilized policy can learn the disturbance curriculum instead of only surviving forced-disturb evaluation. |
| `tests/test_amp_training_contracts.py` | implemented | Extends the selective-walk retention probe contract so the two new configs must keep the intended run names, retention coefficients, style weight, staged disturbance levels, and all-profile staged monitoring. |

Training decision:

- Both configs are warm-start probes from `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`; do not train them from scratch.
- Both should use `--max_iterations=8000`; because the Jun17 source checkpoint is internally `iter=4000`, the main terminal checkpoint should again be around `model_12000.pt`.
- Evaluate `model_best_task.pt` and `model_12000.pt` with the no-disturb full7 protocol first. Run forced `0.75` full7 evaluation only for checkpoints that preserve no-disturb full7 quality.

Recommended training commands:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk \
  --checkpoint=-2 \
  --cfg_override_json configs/ablation/selective_walk_profile_teacher_retention_coef010_probe.json \
  --run_name selective_walk_profile_teacher_retention_coef010_probe \
  --max_iterations=8000
```

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk \
  --checkpoint=-2 \
  --cfg_override_json configs/ablation/selective_walk_profile_teacher_retention_disturb075_probe.json \
  --run_name selective_walk_profile_teacher_retention_disturb075_probe \
  --max_iterations=8000
```

### July06 Evaluation of Jul05 Warm-Start Probes

Hypothesis: if `teacher_policy_retention_coef=0.25` was only stronger than needed, the `0.10` probe should still preserve the seven-preset baseline; if the retention-stabilized policy can learn staged disturbance during training, the `disturb075` probe should reach `staged_disturb_level=0.75` without losing no-disturb or forced-disturbance full7 performance.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jul05
```

Training artifacts:

| experiment | config | run directory | evaluated checkpoints | tail staged disturbance | status |
|---|---|---|---|---:|---|
| `selective_walk_profile_teacher_retention_coef010_probe` | `configs/ablation/selective_walk_profile_teacher_retention_coef010_probe.json` | `logs/r2_amp/Jul05/Jul05_15-54-09_selective_walk_profile_teacher_retention_coef010_probe` | `model_best_task.pt`, `model_12000.pt`; `model_12000.pt` also evaluated at forced `0.75` disturbance | `0.0000` | evaluated |
| `selective_walk_profile_teacher_retention_disturb075_probe` | `configs/ablation/selective_walk_profile_teacher_retention_disturb075_probe.json` | `logs/r2_amp/Jul05/Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe` | `model_best_task.pt`, `model_12000.pt`; `model_12000.pt` also evaluated at forced `0.75` disturbance | `0.7500` | evaluated |

Training-tail facts from `train.log`:

- `coef010` ended at learning iteration `11999/12000` with `Mean task reward: 39.35`, `Best task reward: 43.96`, `staged_disturb_level=0.0000`, `staged_disturb_window_task_return=42.4119`, and `staged_disturb_window_fall_rate=0.0125`.
- `disturb075` ended at learning iteration `11999/12000` with `Mean task reward: 37.22`, `Best task reward: 42.62`, `staged_disturb_level=0.7500`, `staged_disturb_window_task_return=35.8273`, and `staged_disturb_window_fall_rate=0.1092`.

Evaluation protocol:

- WSL CPU PhysX / CPU policy via `legged_gym/scripts/evaluate.py`.
- `--task=r2amp`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default seven fixed presets; DTW was not enabled.
- Forced-disturbance follow-up was run for both `model_12000.pt` checkpoints because both preserved no-disturb full7 quality.

Evaluation outputs:

```text
outputs/eval/July05_selective_walk_profile_teacher_retention_coef010_probe_best_task_baseline_full7
outputs/eval/July05_selective_walk_profile_teacher_retention_coef010_probe_12000_baseline_full7
outputs/eval/July05_selective_walk_profile_teacher_retention_coef010_probe_12000_full7_disturb075
outputs/eval/July05_selective_walk_profile_teacher_retention_disturb075_probe_best_task_baseline_full7
outputs/eval/July05_selective_walk_profile_teacher_retention_disturb075_probe_12000_baseline_full7
outputs/eval/July05_selective_walk_profile_teacher_retention_disturb075_probe_12000_full7_disturb075
```

Aggregate result:

| checkpoint / protocol | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | action-rate L2 | worst task preset | worst task return | worst fall preset | worst fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `coef010/model_best_task.pt`, baseline full7 | 7 | 29.16 | 0.025 | 9.88 | 0.301 | 0.458 | 2.68 | `run` | 19.31 | `stand` | 0.156 |
| `coef010/model_12000.pt`, baseline full7 | 7 | 35.00 | 0.058 | 9.70 | 0.268 | 0.335 | 2.51 | `jump` | 29.04 | `jump` | 0.219 |
| `coef010/model_12000.pt`, forced `0.75` full7 | 7 | 32.57 | 0.051 | 9.70 | 0.322 | 0.426 | 2.65 | `jump` | 24.46 | `jump` | 0.219 |
| `disturb075/model_best_task.pt`, baseline full7 | 7 | 30.57 | 0.022 | 9.87 | 0.290 | 0.434 | 2.67 | `run` | 23.12 | `stand` | 0.109 |
| `disturb075/model_12000.pt`, baseline full7 | 7 | 34.53 | 0.049 | 9.65 | 0.287 | 0.339 | 2.46 | `run` | 22.07 | `run` | 0.109 |
| `disturb075/model_12000.pt`, forced `0.75` full7 | 7 | 34.11 | 0.025 | 9.88 | 0.305 | 0.376 | 2.51 | `run` | 26.43 | `stand` | 0.141 |

Selected per-preset facts:

- `coef010/model_12000.pt` improved the no-disturb aggregate over its early best checkpoint (`35.00` vs `29.16` avg task return), and forced `0.75` disturbance still kept avg task return at `32.57`; its weakest preset in both 12000 evaluations was `jump`, with forced-disturbance `task_return=24.46` and `fall_rate=0.219`.
- `disturb075/model_12000.pt` reached the trained staged-disturbance target (`staged_disturb_level=0.7500`) and preserved performance: no-disturb avg task return was `34.53`, forced `0.75` avg task return was `34.11`, and the forced-disturbance worst task preset was `run` at `26.43`.
- Both Jul05 `model_12000.pt` checkpoints remain above the prior Jun17 warm-start reference (`28.03` avg task return, `0.038` fall rate) and the July05 `0.25` teacher-retention reference under forced `0.75` (`31.83` avg task return, `0.029` fall rate), although `coef010` has a higher worst-preset fall rate on `jump`.

Interpretation:

- The `0.10` retention coefficient is sufficient for this batch: it preserved and improved no-disturb full7 performance and survived forced `0.75` disturbance. This weakens the idea that the earlier success required a very strong `0.25` retention penalty.
- The `disturb075` probe is the cleaner continuation candidate because it learned the staged disturbance curriculum during training and had the strongest forced `0.75` aggregate among the Jul05 probes (`34.11` avg task return, `0.025` avg fall rate).
- The next useful evaluation should stress `disturb075/model_12000.pt` at higher forced-disturbance ratios or with targeted diagnostics on the remaining weakest presets, rather than returning to task-only profile fine-tuning.

### July06 Next Disturbance-Continuation Config

Hypothesis: because `selective_walk_profile_teacher_retention_disturb075_probe/model_12000.pt` already learned the staged disturbance curriculum up to `0.75` and stayed stable under forced `0.75`, the next training should continue from that checkpoint and push the staged cap toward `1.0` without changing teacher retention or the command-profile manifold.

Implemented code/config artifacts:

| artifact | status | purpose |
|---|---|---|
| `legged_gym/envs/r2/r2interrupt_config.py` | implemented | Adds `disturb.stage_init_curriculum_to_level=False` as a default-off resume-only schema field, so existing staged curricula still initialize `disturb_rad_curriculum` at `0.0`. |
| `legged_gym/envs/r2/r2interrupt.py` | implemented | Consumes `stage_init_curriculum_to_level`; when a staged resume config sets it to `true`, `R2InterruptRobot.__init__()` initializes `disturb_rad_curriculum` to the current stage cap instead of zero. This is needed because `stage_levels` are caps, not saved environment state. |
| `configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json` | implemented; Jul06 run evaluated with source-checkpoint mismatch | Intended to continue from the Jul05 `disturb075/model_12000.pt` checkpoint, keep `teacher_policy_retention_coef=0.25`, seven eval-like profiles, and `style_reward_weight=0.0`, then train staged disturbance through `0.75 -> 0.85 -> 0.925 -> 1.0`. The Jul06 run below did not follow that intended source. |
| `tests/test_amp_training_contracts.py` | implemented | Locks the new resume-only initialization switch and the `disturb100` JSON contract, including the continuation checkpoint note, stage gates, all-profile monitoring, and 4000-additional-iteration budget. |

Training decision:

- Resume from `logs/r2_amp/Jul05/Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe/model_12000.pt`, not from the old Jun17 warm-start.
- Keep teacher retention at `0.25` and AMP style reward at `0.0`, so the only material training change is higher staged disturbance.
- Use `--max_iterations=4000`; because the source checkpoint is internally `iter=12000`, the expected terminal checkpoint is `model_16000.pt`.
- Evaluate `model_best_task.pt` and `model_16000.pt` with no-disturb full7 first. If the no-disturb full7 aggregate is preserved, run forced `0.75`, `0.925`, and `1.0` full7 diagnostics.

Recommended training command:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jul05/Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe \
  --checkpoint=12000 \
  --cfg_override_json configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json \
  --run_name selective_walk_profile_teacher_retention_disturb100_probe \
  --max_iterations=4000
```

### July08 Evaluation of Jul06 Disturb100 Run

Hypothesis under test: pushing the retention-stabilized disturbance curriculum beyond `0.75` should only continue if the policy preserves the no-disturb seven-preset baseline before any higher forced-disturbance diagnostics.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jul06
```

Training artifact:

| experiment | config | run directory | evaluated checkpoints | tail staged disturbance | status |
|---|---|---|---|---:|---|
| `selective_walk_profile_teacher_retention_disturb100_probe` | `configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json` | `logs/r2_amp/Jul06/Jul06_13-53-29_selective_walk_profile_teacher_retention_disturb100_probe` | `model_best_task.pt`, `model_8000.pt` | `0.7500` | evaluated; source-checkpoint mismatch |

Training-tail and source facts from `train.log`:

- The run loaded `logs/r2_amp/Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk/model_best_task.pt`, not the intended `logs/r2_amp/Jul05/Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe/model_12000.pt`.
- The checkpoint sequence ended at `model_8000.pt`, not the planned `model_16000.pt`.
- The tail entry ended at learning iteration `7999/8000` with `Mean task reward: -7.72`, `Best task reward: 40.98`, `staged_disturb_level=0.7500`, `staged_disturb_window_task_return=-43.4630`, and `staged_disturb_window_fall_rate=0.8436`.

Evaluation protocol:

- WSL CPU PhysX / CPU policy via `legged_gym/scripts/evaluate.py`.
- `--task=r2amp`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default seven fixed presets; DTW was not enabled.
- Forced-disturbance follow-up was not run because `model_8000.pt` failed the no-disturb full7 preservation gate.

Evaluation outputs:

```text
outputs/eval/July06_selective_walk_profile_teacher_retention_disturb100_probe_best_task_baseline_full7
outputs/eval/July06_selective_walk_profile_teacher_retention_disturb100_probe_8000_baseline_full7
```

Aggregate result:

| checkpoint / protocol | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | action-rate L2 | worst task preset | worst task return | worst fall preset | worst fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `disturb100/model_best_task.pt`, baseline full7 | 7 | 29.94 | 0.022 | 9.88 | 0.307 | 0.437 | 2.52 | `run` | 20.03 | `stand` | 0.078 |
| `disturb100/model_8000.pt`, baseline full7 | 7 | 19.87 | 0.330 | 7.49 | 0.402 | 0.539 | 3.72 | `jump` | -1.97 | `jump` | 1.000 |

Selected per-preset facts:

- `model_best_task.pt` remained usable under no-disturb full7: all seven presets had `fall_rate <= 0.078`; the weakest task preset was `run` with `task_return=20.03` and `fall_rate=0.016`.
- `model_8000.pt` regressed sharply: `jump` had `task_return=-1.97`, `fall_rate=1.000`, and `survival=1.20s`; `stand` also rose to `fall_rate=0.484`.
- Because the terminal checkpoint fails no-disturb full7, higher forced-disturbance tests at `0.75`, `0.925`, or `1.0` would not isolate disturbance robustness; they would mostly measure an already-regressed policy.

Interpretation:

- This Jul06 run should not be treated as a valid test of the planned Jul05-to-1.0 continuation, because its `train.log` shows it resumed from Jun17 rather than from the Jul05 `disturb075/model_12000.pt` source.
- The early best checkpoint is a usable recovery point, but the terminal `model_8000.pt` is not a continuation candidate.
- The next clean action is to rerun the intended command from the previous section, verify the new `train.log` loads the Jul05 `model_12000.pt` source before training proceeds, and expect the terminal checkpoint to align with `model_16000.pt`.

### July10 Evaluation of Jul08_12 Disturb100 Continuation

Hypothesis under test: if the `disturb100` continuation is resumed from the correct Jul05 `disturb075/model_12000.pt` source, the terminal `model_16000.pt` should preserve the no-disturb seven-preset baseline and then reveal the usable forced-disturbance boundary between `0.75` and `1.0`.

Training root:

```text
E:\codebase\VR_Teleoperation\logs\r2_amp\Jul08_12
```

Training artifact:

| experiment | config | run directory | evaluated checkpoints | tail staged disturbance | status |
|---|---|---|---|---:|---|
| `selective_walk_profile_teacher_retention_disturb100_probe` | `configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json` | `logs/r2_amp/Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe` | `model_best_task.pt`, `model_16000.pt`; `model_16000.pt` also evaluated at forced `0.75`, `0.925`, and `1.0` disturbance | `1.0000` | evaluated; correct Jul05 source |

Training-tail and source facts from `train.log`:

- The run loaded `/home/ubuntu/lzxworkspace/codespace/VR_Teleoperation/logs/r2_amp/Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe/model_12000.pt`, matching the intended Jul05 `disturb075/model_12000.pt` source by run name and checkpoint.
- The checkpoint sequence includes `model_12000.pt` through `model_16000.pt`, plus `model_best_task.pt` and top-task checkpoints `model_top_task_12052.pt`, `model_top_task_12060.pt`, and `model_top_task_12095.pt`.
- The tail entry ended at learning iteration `15999/16000` with `Mean task reward: 17.49`, `Best task reward: 63.63`, `staged_disturb_level=1.0000`, `staged_disturb_window_task_return=20.1177`, and `staged_disturb_window_fall_rate=0.1062`.

Evaluation protocol:

- WSL CPU PhysX / CPU policy via `legged_gym/scripts/evaluate.py`.
- `--task=r2amp`, `--num_envs=64`, `--num_episodes=64`, `--episode_seconds=10`.
- Default seven fixed presets; DTW was not enabled.
- `model_best_task.pt` and `model_16000.pt` were first evaluated with no forced disturbance. Because `model_16000.pt` preserved the no-disturb full7 aggregate, forced-disturbance full7 diagnostics were run at `0.75`, `0.925`, and `1.0`.

Evaluation outputs:

```text
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_best_task_baseline_full7
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_baseline_full7
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_full7_disturb075
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_full7_disturb0925
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_full7_disturb100
```

Aggregate result:

| checkpoint / protocol | rows | avg task return | avg fall rate | avg survival s | lin rmse | yaw rmse | action-rate L2 | worst task preset | worst task return | worst fall preset | worst fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `disturb100/model_best_task.pt`, baseline full7 | 7 | 34.59 | 0.056 | 9.63 | 0.277 | 0.338 | 2.45 | `run` | 24.04 | `jump` | 0.125 |
| `disturb100/model_16000.pt`, baseline full7 | 7 | 33.64 | 0.083 | 9.52 | 0.289 | 0.353 | 2.48 | `run` | 25.03 | `jump` | 0.250 |
| `disturb100/model_16000.pt`, forced `0.75` full7 | 7 | 33.52 | 0.049 | 9.74 | 0.309 | 0.389 | 2.58 | `run` | 24.23 | `jump` | 0.156 |
| `disturb100/model_16000.pt`, forced `0.925` full7 | 7 | 29.55 | 0.092 | 9.42 | 0.386 | 0.455 | 2.84 | `run` | 17.17 | `jump` | 0.328 |
| `disturb100/model_16000.pt`, forced `1.0` full7 | 7 | 22.26 | 0.208 | 8.49 | 0.504 | 0.600 | 3.41 | `jump` | 7.06 | `jump` | 0.547 |

Selected per-preset facts:

- `model_16000.pt` preserved the no-disturb baseline relative to its early best checkpoint: avg task return moved from `34.59` to `33.64`, and avg fall rate from `0.056` to `0.083`; the weakest no-disturb task preset remained `run` at `25.03`.
- At forced `0.75`, `model_16000.pt` remained close to its no-disturb result: avg task return `33.52`, avg fall rate `0.049`, and avg survival `9.74s`. The weakest task preset was still `run` at `24.23`.
- At forced `0.925`, performance degraded but remained usable at the aggregate level: avg task return `29.55`, avg fall rate `0.092`, and avg survival `9.42s`. The main weak point was `jump`, which had the worst fall rate at `0.328`.
- At forced `1.0`, the stress boundary became clear: avg task return fell to `22.26`, avg fall rate rose to `0.208`, and `jump` was the dominant failure mode with `task_return=7.06`, `fall_rate=0.547`, and `survival=6.16s`. `stand` also degraded with `fall_rate=0.328`, while `strafe_right` remained comparatively strong at `task_return=33.74` and `fall_rate=0.063`.

Focused forced-`1.0` failure diagnostic:

```text
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/metrics.csv
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/termination_reasons.csv
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/state_trace.csv
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/failure_diagnostics_summary.csv
outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/failure_diagnostics_summary.json
```

- Checkpoint/config: the same Jul08_12 `model_16000.pt` and `selective_walk_profile_teacher_retention_disturb100_probe.json`.
- Protocol: WSL CPU, `jump` and `stand`, `64` episodes per preset, `--eval_disturb_ratio=1.0`, `--record_termination_reasons`, `--record_state_trace`, and `--state_trace_window_steps=50`.
- `failure_diagnostics_summary.*` was generated from the three evaluator CSVs; terminal-state values use only rows with `steps_until_done=0`.

| preset | task return | fall rate | survival s | `contact:base_link` | contact mean survival s | `orientation:roll_pitch` | timeout | terminal mean base z | terminal max contact force |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump` | 10.49 | 0.469 | 6.80 | 30/64 (0.469) | 3.16 | 0/64 (0.000) | 34/64 (0.531) | 0.612 | 1307.21 |
| `stand` | 10.53 | 0.484 | 6.53 | 29/64 (0.453) | 2.71 | 2/64 (0.031) | 33/64 (0.516) | 0.573 | 1184.95 |

Diagnostic interpretation:

- The focused diagnostic is a separate stochastic rollout, so its exact task/fall values should not be substituted for the earlier seven-preset table. Its termination classification is the evidence needed here: both weak profiles are dominated by early `base_link` contact, while orientation-only termination is absent for `jump` and rare for `stand`.
- This evidence does not support changing an orientation threshold or introducing a new reward scale. The existing failure is primarily contact collapse under full disturbance.
- Code inspection found a matching curriculum-control gap: `stage_monitor_profiles` filtered which episodes entered the window, but `_maybe_advance_staged_disturb_release()` still decided from one aggregate return/fall mean. Therefore strong monitored profiles could offset weak `jump/stand` values even though all seven names were listed.

Evidence-based code/config follow-up completed on July10 and review-hardened on July11:

- Added `legged_gym/envs/r2/staged_disturb_gate.py` with simulator-independent aggregate and strict per-profile readiness/pass checks.
- Added the default-off schema option `disturb.stage_require_all_monitor_profiles`. When enabled, `R2InterruptRobot` separately accumulates count/return/fall for each named monitor profile, waits for each profile to reach `stage_min_episodes`, and requires each one to pass the current-stage gates. Strict mode also requires command resampling to occur after the maximum episode length, so a full episode return cannot be attributed to a profile selected halfway through the episode. Existing configs retain aggregate behavior because the option defaults to `false`.
- Added `scripts/run_jul08_disturb100_diagnostics.py` and `scripts/summarize_failure_diagnostics.py` so the exact diagnostic and terminal-state summary are reproducible.
- Added `configs/ablation/selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json`; status: **not trained**.

Planned recovery experiment:

| field | value |
|---|---|
| source | `logs/r2_amp/Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe/model_16000.pt` |
| run name | `selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery` |
| profile weights | `stand=0.25`, `jump=0.25`, remaining five profiles sum to `0.50` |
| profile hold | `commands.resampling_time=30.0s`, longer than the `20s` episode, so each gate episode has one profile |
| staged levels | `0.925 -> 0.95 -> 0.975 -> 1.0` |
| per-profile gates | min return `[18, 20, 22, 24]`; max fall `[0.20, 0.16, 0.12, 0.10]`; `1024` episodes per profile |
| unchanged controls | teacher retention `0.25`; AMP style reward `0.0`; PPO settings, command anchors, and reward scales unchanged |
| budget / expected terminal | additional `4000` iterations; expected `model_20000.pt` |
| status | **not trained** |

Training command after activating the training environment:

```bash
python legged_gym/scripts/train.py \
  --task=r2amp \
  --headless \
  --resume \
  --load_run Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe \
  --checkpoint=16000 \
  --cfg_override_json configs/ablation/selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json \
  --run_name selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery \
  --max_iterations=4000
```

Interpretation:

- This `Jul08_12` run is the valid test that the Jul06 source-mismatch run failed to provide: it starts from the intended Jul05 `disturb075/model_12000.pt` source, reaches `model_16000.pt`, and preserves the no-disturb seven-preset aggregate.
- The useful forced-disturbance operating range is stronger at `0.75` and still broadly usable at `0.925`; full `1.0` disturbance is not solved, with `jump` and then `stand` carrying most of the fall-rate increase.
- The focused diagnostic is now complete and supports an opt-in per-profile curriculum guard plus targeted `jump/stand` resampling. The next empirical step is to train the new recovery config, verify the first `Loading model from` line points to Jul08_12 `model_16000.pt`, and evaluate the resulting `model_20000.pt` first on baseline full7, then at forced `0.925` and `1.0`.

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
