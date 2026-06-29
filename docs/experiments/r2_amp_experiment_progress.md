# R2 AMP Experiment Progress

Last updated: 2026-06-30

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
logs/r2_amp/June15/sw1
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
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | `logs/r2_amp/June20/Jun20_15-18-58_command_hold_conservative_penalty_ramp` | `5818`, `7663`, `7930` | `23.49` | `0.0000` | evaluated |
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | `logs/r2_amp/June20/Jun20_15-19-48_command_hold_controlled_disturb_release` | `1166`, `1706`, `1944` | `-4.26` | `0.9956` | evaluated |
| `command_hold_no_push` | `configs/ablation/command_hold_no_push.json` | `logs/r2_amp/June20/Jun20_15-21-52_command_hold_no_push` | `6059`, `6973`, `7440` | `31.15` | `0.0000` | evaluated |
| `command_hold_style_lowcap` | `configs/ablation/command_hold_style_lowcap.json` | `logs/r2_amp/June20/Jun20_15-22-56_command_hold_style_lowcap` | `7439`, `7600`, `7937` | `28.81` | `0.0000` | evaluated |

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
| `command_hold_conservative_penalty_ramp` | `configs/ablation/command_hold_conservative_penalty_ramp.json` | `logs/r2_amp/June21/Jun21_12-28-33_command_hold_conservative_penalty_ramp` | `7075`, `7654`, `7657` | `32.80` | `0.0000` | `34.69` | evaluated |
| `command_hold_controlled_disturb_release` | `configs/ablation/command_hold_controlled_disturb_release.json` | `logs/r2_amp/June21/Jun21_12-28-55_command_hold_controlled_disturb_release` | `1450`, `1498`, `1608` | `9.36` | `0.9943` | `21.39` | evaluated |

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
| `command_hold_staged_disturb_release` | `configs/ablation/command_hold_staged_disturb_release.json` | `logs/r2_amp/June23/Jun23_03-38-06_command_hold_staged_disturb_release` | `1315`, `1331`, `1705` | `7.36` | `0.9944` | `1.0000 / 4` | `0.1174` | `26.76` | evaluated |
| `command_hold_run_focused_staged_disturb_release` | `configs/ablation/command_hold_run_focused_staged_disturb_release.json` | `logs/r2_amp/June23/Jun23_14-58-32_command_hold_run_focused_staged_disturb_release` | `4221`, `4294`, `7112` | `8.14` | `0.0000` | `0.0000 / 0` | `0.6372` | `12.08` | evaluated |

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
