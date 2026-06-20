# R2 AMP Experiment Progress

Last updated: 2026-06-20

This document is the running record for R2 AMP ablations. Keep it factual: record what was run, what changed, where the artifacts are, what the evaluation showed, and what conclusion is supported by the data.

## Current Question

The current multi-expert AMP policy can show useful early checkpoints but then regress late in training. The working hypothesis is that late collapse is driven more by command/curriculum/disturbance scheduling than by the discriminator architecture alone.

The first July19 batch tests three levers:

- `scratch_command_hold`: keep multi-expert AMP, but disable command curriculum and hold the command range fixed.
- `scratch_no_push`: keep curriculum and AMP, but remove randomized push impulses.
- `scratch_amp_slow_lowcap`: keep multi-expert routing, but make AMP style reward weaker and slower.

`scratch_slow_penalty_ramp` was configured but no July19 training directory exists, so it has no result in this batch.

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
```

Each output directory contains `metrics.csv` and `metrics.json`. Each `metrics.csv` has 7 rows, one row per preset.

### Training Artifacts

| experiment | config | run directory | top task checkpoint iterations | status |
|---|---|---|---|---|
| `scratch_amp_slow_lowcap` | `configs/ablation/scratch_amp_slow_lowcap.json` | `logs/r2_amp/July19/Jun19_16-08-42_scratch_amp_slow_lowcap` | `1214`, `1227`, `1806` | evaluated |
| `scratch_command_hold` | `configs/ablation/scratch_command_hold.json` | `logs/r2_amp/July19/Jun19_16-09-11_scratch_command_hold` | `7120`, `7219`, `7966` | evaluated |
| `scratch_no_push` | `configs/ablation/scratch_no_push.json` | `logs/r2_amp/July19/Jun19_16-12-37_scratch_no_push` | `1676`, `1685`, `1881` | evaluated |
| `scratch_slow_penalty_ramp` | `configs/ablation/scratch_slow_penalty_ramp.json` | missing in July19 | none | not trained |

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

### Supported Conclusion

The strongest continuation target is `scratch_command_hold`.

Reason:

- `scratch_no_push` has the best early checkpoint, but its final checkpoint regresses from `-11.48` to `-64.47`. Removing push improves early learning but does not fix late training drift.
- `scratch_amp_slow_lowcap` regresses from `-22.25` to `-142.87`. Weakening/slowing AMP style reward alone does not solve the problem and is not worth continuing as-is.
- `scratch_command_hold` stays close between best and final: `-21.57` to `-24.24`, with lower final fall rate than its best checkpoint. This supports the hypothesis that command/curriculum scheduling is a main driver of late instability.

In plain terms: push is part of the stress, but the more important issue is the late training schedule. Holding command curriculum stabilizes training better than only weakening AMP or removing push.

## Current Decision

Do not continue:

- `scratch_amp_slow_lowcap`

Keep as diagnostic evidence, not as the final mechanism:

- `scratch_no_push`

Continue from:

- `scratch_command_hold`

Recommended next batch:

1. `command_hold + no_push`: checks whether removing push on top of fixed commands improves early quality without the final collapse.
2. `command_hold + slow push ramp`: keeps robustness training but restores push gradually instead of disabling it.
3. `command_hold + late AMP/style reduction`: tests whether the final phase needs weaker style pressure after task behavior forms.
4. `command_hold + staged command release`: start fixed, then gradually release walk/run before jump.

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
