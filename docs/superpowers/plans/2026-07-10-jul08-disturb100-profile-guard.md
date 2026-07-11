# Jul08 Disturb100 Profile Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the Jul08_12 `model_16000.pt` failures at forced disturbance `1.0`, then prevent aggregate staged-curriculum metrics from hiding weak `jump` and `stand` profiles in the next continuation experiment.

**Architecture:** Add one exact, dry-run-by-default diagnostic launcher for the approved Jul08_12 checkpoint and one reusable CSV/JSON summarizer. After the diagnostic establishes the failure mechanism, add a simulator-independent staged-gate helper and integrate an opt-in all-profile guard into `R2InterruptRobot`; a new ablation JSON enables the guard and rebalances training toward `jump` and `stand` without changing reward scales.

**Tech Stack:** Python 3.8, standard-library `argparse/csv/json/subprocess`, PyTorch in the environment integration, pytest contract tests, WSL CPU Isaac Gym evaluation.

## Global Constraints

- Work only in `E:\codebase\VR_Teleoperation`; preserve unrelated worktree changes.
- Use `model_16000.pt` from `Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe`.
- Diagnostic presets are exactly `jump` and `stand`, with `--eval_disturb_ratio=1.0`, 64 environments, 64 episodes, and 10 seconds per episode.
- New staged-gate behavior is opt-in and defaults off, preserving existing runs.
- Do not change task reward scales unless termination/state-trace evidence specifically requires it.
- Update `CODE_STRUCTURE.md`, `docs/experiments/r2_amp_experiment_progress.md`, and codegraph after code changes.

---

### Task 1: Exact Jul08_12 Diagnostic Workflow

**Files:**
- Create: `scripts/run_jul08_disturb100_diagnostics.py`
- Create: `scripts/summarize_failure_diagnostics.py`
- Modify: `tests/test_amp_training_contracts.py`

**Interfaces:**
- Produces: `build_diagnostic_command() -> List[str]` and `run_diagnostic(execute: bool, command_runner=...)`.
- Produces: `summarize_failure_diagnostics(input_dir: str, output_dir: Optional[str] = None) -> dict`.

- [x] **Step 1: Write failing launcher and summarizer tests**

Assert the command includes the exact run, checkpoint, config, `jump/stand` presets, forced ratio, termination/state flags, and 50-step tail. Build small temporary `metrics.csv`, `termination_reasons.csv`, and `state_trace.csv` fixtures and assert terminal rows (`steps_until_done == 0`) drive the summary.

- [x] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "jul08_disturb100_diagnostic or failure_diagnostics_summary" -q`

Expected: failure because the two scripts do not exist.

- [x] **Step 3: Implement the launcher and summarizer**

The launcher defaults to printing the exact `wsl.exe ... env ... evaluate.py` argv and only executes with `--execute`. The summarizer validates all three evaluator artifacts, groups termination rows by preset/reason/detail, and writes `failure_diagnostics_summary.csv/json` using terminal trace rows only.

- [x] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "jul08_disturb100_diagnostic or failure_diagnostics_summary" -q`

Expected: both tests pass.

### Task 2: Run the Approved Diagnostic and Establish Evidence

**Files:**
- Create at runtime: `outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics/`

**Interfaces:**
- Consumes: Task 1 launcher and Jul08_12 checkpoint.
- Produces: evaluator metrics, termination reasons, state traces, and diagnostic summaries.

- [x] **Step 1: Dry-run and inspect the command**

Run: `python scripts/run_jul08_disturb100_diagnostics.py`

Expected: exact WSL command with no process execution.

- [x] **Step 2: Execute the WSL CPU evaluation**

Run: `python scripts/run_jul08_disturb100_diagnostics.py --execute`

Expected: the evaluator writes six raw artifacts and the summarizer adds two derived artifacts.

- [x] **Step 3: Summarize the evidence**

Run: `python scripts/summarize_failure_diagnostics.py --input_dir outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_16000_jump_stand_disturb100_failure_diagnostics`

Expected: two preset rows and reason/state summaries derived from 64 episodes per preset.

### Task 3: Opt-In Per-Profile Staged Gate

**Files:**
- Create: `legged_gym/envs/r2/staged_disturb_gate.py`
- Modify: `legged_gym/envs/r2/r2interrupt.py`
- Modify: `legged_gym/envs/r2/r2interrupt_config.py`
- Modify: `tests/test_amp_training_contracts.py`

**Interfaces:**
- Produces: `staged_disturb_window_ready(...) -> bool`.
- Produces: `staged_disturb_window_passes(...) -> bool`.
- Consumes in `R2InterruptRobot`: aggregate and named-profile episode count, return sum, and fall sum.

- [x] **Step 1: Write failing pure gate tests**

Cover aggregate backward compatibility, a weak `jump` profile hidden by a passing aggregate, all profiles passing, and one profile below the minimum episode count.

- [x] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "staged_disturb_all_profile_gate" -q`

Expected: failure because `staged_disturb_gate.py` does not exist.

- [x] **Step 3: Implement minimal pure gate logic**

The readiness function requires the aggregate window and, when enabled, every named profile to reach `stage_min_episodes`. The pass function preserves aggregate behavior by default and, when enabled, requires every profile's mean return and fall rate to pass the same current-stage thresholds.

- [x] **Step 4: Integrate the opt-in guard**

Add `stage_require_all_monitor_profiles = False` to the schema. Validate that enabling it requires named monitored profiles with positive mixture weights, collect/reset named-profile windows, use the pure helper for advancement/regression, and expose profile task-return/fall-rate values through `extras['episode']`.

- [x] **Step 5: Run tests to verify GREEN**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "staged_disturb" -q`

Expected: all staged-disturb tests pass.

### Task 4: Evidence-Based Recovery Ablation

**Files:**
- Create: `configs/ablation/selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json`
- Modify: `tests/test_amp_training_contracts.py`

**Interfaces:**
- Consumes: Task 3's `stage_require_all_monitor_profiles` option.
- Produces: a 4,000-iteration continuation config from Jul08_12 `model_16000.pt`, targeting `model_20000.pt`.

- [x] **Step 1: Write the failing config contract test**

Require weights summing to 1.0 with `stand=0.25` and `jump=0.25`, `stage_monitor_profiles=["stand", "jump"]`, strict all-profile gating, levels `[0.925, 0.95, 0.975, 1.0]`, unchanged teacher retention `0.25`, and style reward `0.0`.

- [x] **Step 2: Run test to verify RED**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "profile_guard_recovery_json" -q`

Expected: failure because the config does not exist.

- [x] **Step 3: Add the config**

Keep the Jul08 command anchors and optimizer settings; rebalance profile weights to `stand/jump=0.25` and `walk_slow/walk_fast/run/turn_left/strafe_right=0.10/0.12/0.12/0.08/0.08`. Set `commands.resampling_time=30.0`, strictly beyond the 20-second episode, so each episode belongs to one profile. Use per-stage minimum returns `[18, 20, 22, 24]`, maximum fall rates `[0.20, 0.16, 0.12, 0.10]`, and 1,024 episodes per monitored profile before each gate decision.

- [x] **Step 4: Run test to verify GREEN**

Run: `python -m pytest tests/test_amp_training_contracts.py -k "profile_guard_recovery_json" -q`

Expected: pass.

### Task 5: Documentation, Index, and Final Verification

**Files:**
- Modify: `CODE_STRUCTURE.md`
- Modify: `docs/experiments/r2_amp_experiment_progress.md`

**Interfaces:**
- Records diagnostic facts separately from the recovery hypothesis and marks the new training config `not trained`.

- [x] **Step 1: Update structure and experiment documentation**

Document both scripts, the pure gate functions, the opt-in environment integration, exact output metrics, failure reasons, and the new config's resume command/status.

- [x] **Step 2: Sync codegraph**

Run: `codegraph sync .`

Expected: changed Python files indexed successfully.

- [x] **Step 3: Run focused and full contract verification**

Run: `python -m pytest tests/test_amp_training_contracts.py -q`

Run: `python -m json.tool configs/ablation/selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json`

Run: `git diff --check`

Expected: tests pass, JSON parses, and diff check reports no whitespace errors.
