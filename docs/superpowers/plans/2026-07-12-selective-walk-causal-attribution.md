# Selective-Walk Causal Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four warm-start causal-attribution ablation configs that isolate continuation drift, 30-second profile hold, `stand/jump` oversampling, and the high-start disturbance schedule after the failed Jul11 recovery run.

**Architecture:** Treat `selective_walk_profile_teacher_retention_disturb100_probe.json` as the invariant source contract and create four full, self-describing JSON copies whose behavioral diffs are mechanically checked against one another. Add focused semantic tests that prove each arm changes only its assigned factor, then document the training/evaluation gates and exact commands without changing Python runtime behavior.

**Tech Stack:** JSON recursive config overrides, Python 3.8, pytest contract tests, Markdown experiment documentation, CodeGraph CLI.

## Global Constraints

- Work only in `E:\codebase\VR_Teleoperation`; preserve unrelated worktree changes.
- All four arms resume from `logs/r2_amp/Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe/model_16000.pt`.
- Every arm uses training seed `0`, `teacher_policy_retention_coef=0.25`, `style_reward_weight=0.0`, `save_interval=250`, and exactly `2000` additional iterations, targeting `model_18000.pt`.
- Do not change PPO hyperparameters, reward scales, command anchors/jitter, AMP expert routing, or domain-randomization behavior.
- First-stage configs must not enable strict per-profile gating; `stage_require_all_monitor_profiles` is absent or `false`.
- C0 changes no behavior; H changes only `commands.resampling_time` to `30.0`; W changes only profile weights; S changes only staged levels/window/threshold fields.
- Use TDD: focused test must fail before JSON creation and pass afterward.
- Update `CODE_STRUCTURE.md` and `docs/experiments/r2_amp_experiment_progress.md`; mark all four arms `not trained`.
- Run JSON parsing, focused contracts, the full AMP contract file, `git diff --check`, and CodeGraph sync before completion.

---

### Task 1: Four Causal-Ablation JSON Contracts and Configs

**Files:**
- Create: `configs/ablation/selective_walk_disturb100_causal_control.json`
- Create: `configs/ablation/selective_walk_disturb100_hold30_only.json`
- Create: `configs/ablation/selective_walk_disturb100_stand_jump_weights_only.json`
- Create: `configs/ablation/selective_walk_disturb100_high_start_schedule_only.json`
- Modify: `tests/test_amp_training_contracts.py` after `test_selective_walk_profile_guard_recovery_json_contract`

**Interfaces:**
- Consumes: the full JSON structure of `configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json`.
- Produces: four recursive override payloads accepted by `--cfg_override_json` and a contract test named `test_selective_walk_disturb100_causal_attribution_json_contracts`.

- [ ] **Step 1: Add the focused contract test before creating any JSON**

Append this test after the existing profile-guard recovery contract. It compares complete payloads after replacing only the explicitly permitted paths, so accidental optimizer/reward/command changes fail the test.

```python
def test_selective_walk_disturb100_causal_attribution_json_contracts():
    ablation_dir = ROOT_DIR / "configs/ablation"

    def load(filename):
        return json.loads((ablation_dir / filename).read_text(encoding="utf-8"))

    def without_metadata(payload):
        normalized = json.loads(json.dumps(payload))
        normalized.pop("notes")
        runner = normalized["train"]["runner"]
        runner.pop("run_name")
        runner.pop("max_iterations")
        return normalized

    source = load("selective_walk_profile_teacher_retention_disturb100_probe.json")
    filenames = {
        "C0": "selective_walk_disturb100_causal_control.json",
        "H": "selective_walk_disturb100_hold30_only.json",
        "W": "selective_walk_disturb100_stand_jump_weights_only.json",
        "S": "selective_walk_disturb100_high_start_schedule_only.json",
    }
    payloads = {arm: load(filename) for arm, filename in filenames.items()}

    for arm, payload in payloads.items():
        runner = payload["train"]["runner"]
        assert runner["max_iterations"] == 2000
        assert runner["save_interval"] == 250
        assert runner["save_top_task_checkpoints"] == 3
        assert runner["run_name"].startswith("selective_walk_disturb100_")
        assert payload["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.25
        assert payload["train"]["amp"]["style_reward_weight"] == 0.0
        assert "Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe" in payload["notes"]
        assert "model_16000.pt" in payload["notes"]
        assert "model_18000.pt" in payload["notes"]
        assert "--max_iterations=2000" in payload["notes"]
        assert payload["env"]["disturb"].get(
            "stage_require_all_monitor_profiles", False
        ) is False

    control = payloads["C0"]
    assert without_metadata(control) == without_metadata(source)

    hold = json.loads(json.dumps(payloads["H"]))
    assert hold["env"]["commands"].pop("resampling_time") == 30.0
    assert without_metadata(hold) == without_metadata(control)

    weighted = json.loads(json.dumps(payloads["W"]))
    weights = {
        profile["name"]: float(profile["weight"])
        for profile in weighted["env"]["commands"]["profile_mixture"]
    }
    assert weights == {
        "stand": 0.25,
        "walk_slow": 0.10,
        "walk_fast": 0.12,
        "run": 0.12,
        "jump": 0.25,
        "turn_left": 0.08,
        "strafe_right": 0.08,
    }
    control_weights = {
        profile["name"]: profile["weight"]
        for profile in control["env"]["commands"]["profile_mixture"]
    }
    for profile in weighted["env"]["commands"]["profile_mixture"]:
        profile["weight"] = control_weights[profile["name"]]
    assert without_metadata(weighted) == without_metadata(control)

    scheduled = json.loads(json.dumps(payloads["S"]))
    scheduled_disturb = scheduled["env"]["disturb"]
    assert scheduled_disturb["stage_levels"] == [0.925, 0.95, 0.975, 1.0]
    assert scheduled_disturb["stage_min_episodes"] == 1024
    assert scheduled_disturb["stage_min_task_return"] == [18.0, 20.0, 22.0, 24.0]
    assert scheduled_disturb["stage_max_fall_rate"] == [0.20, 0.16, 0.12, 0.10]
    source_disturb = control["env"]["disturb"]
    for key in (
        "stage_levels",
        "stage_min_episodes",
        "stage_min_task_return",
        "stage_max_fall_rate",
    ):
        scheduled_disturb[key] = source_disturb[key]
    assert without_metadata(scheduled) == without_metadata(control)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m pytest tests/test_amp_training_contracts.py::test_selective_walk_disturb100_causal_attribution_json_contracts -q
```

Expected: FAIL with `FileNotFoundError` for `selective_walk_disturb100_causal_control.json`.

- [ ] **Step 3: Create C0 as the invariant continuation control**

Copy the complete object from `selective_walk_profile_teacher_retention_disturb100_probe.json`, then make exactly these metadata changes:

```text
notes: state hypothesis "pure continuation control"; resume source Jul08_12 model_16000.pt; command uses --checkpoint=16000 --max_iterations=2000; expected model_18000.pt; cite Schulman et al. 2017 and Li & Hoiem 2016.
train.runner.run_name: selective_walk_disturb100_causal_control
train.runner.max_iterations: 2000
```

Do not add `commands.resampling_time`; inherited runtime value remains `10.0`. Do not add or enable `stage_require_all_monitor_profiles`.

- [ ] **Step 4: Create H from C0 with one behavioral change**

Copy the completed C0 object and change exactly:

```json
{
  "env": {
    "commands": {
      "resampling_time": 30.0
    }
  },
  "train": {
    "runner": {
      "run_name": "selective_walk_disturb100_hold30_only"
    }
  }
}
```

Replace notes with the hold-only hypothesis and the same source/CLI/terminal contract. Do not change profile weights, staged fields, monitored profiles, PPO, AMP, commands, jitter, or rewards.

- [ ] **Step 5: Create W from C0 with one behavioral change**

Copy C0, set `train.runner.run_name` to `selective_walk_disturb100_stand_jump_weights_only`, replace notes with the weights-only hypothesis, and replace only the seven `weight` values:

```json
{
  "stand": 0.25,
  "walk_slow": 0.10,
  "walk_fast": 0.12,
  "run": 0.12,
  "jump": 0.25,
  "turn_left": 0.08,
  "strafe_right": 0.08
}
```

The sum must equal `1.0`. Do not add `resampling_time` or change any profile command/jitter vector.

- [ ] **Step 6: Create S from C0 with one behavioral change**

Copy C0, set `train.runner.run_name` to `selective_walk_disturb100_high_start_schedule_only`, replace notes with the schedule-only hypothesis, and replace exactly these fields:

```json
{
  "stage_levels": [0.925, 0.95, 0.975, 1.0],
  "stage_min_episodes": 1024,
  "stage_min_task_return": [18.0, 20.0, 22.0, 24.0],
  "stage_max_fall_rate": [0.20, 0.16, 0.12, 0.10]
}
```

Keep the original seven `stage_monitor_profiles`, aggregate gate semantics, original profile weights, and inherited `10s` resampling. Do not add `stage_require_all_monitor_profiles=true`.

- [ ] **Step 7: Parse all JSONs and verify GREEN**

Run:

```powershell
$files = @(
  'configs/ablation/selective_walk_disturb100_causal_control.json',
  'configs/ablation/selective_walk_disturb100_hold30_only.json',
  'configs/ablation/selective_walk_disturb100_stand_jump_weights_only.json',
  'configs/ablation/selective_walk_disturb100_high_start_schedule_only.json'
)
foreach ($file in $files) { python -m json.tool $file | Out-Null }
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m pytest tests/test_amp_training_contracts.py::test_selective_walk_disturb100_causal_attribution_json_contracts -q
```

Expected: four parse commands exit `0`; focused test reports `1 passed`.

- [ ] **Step 8: Commit Task 1**

```powershell
git add configs/ablation/selective_walk_disturb100_causal_control.json configs/ablation/selective_walk_disturb100_hold30_only.json configs/ablation/selective_walk_disturb100_stand_jump_weights_only.json configs/ablation/selective_walk_disturb100_high_start_schedule_only.json tests/test_amp_training_contracts.py
git commit -m "test: add selective-walk causal ablation batch"
```

### Task 2: Documentation, Training Handoff, and Full Verification

**Files:**
- Modify: `CODE_STRUCTURE.md` near the existing `selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json` description.
- Modify: `docs/experiments/r2_amp_experiment_progress.md` immediately before `## Maintenance Rules`.

**Interfaces:**
- Consumes: Task 1's four exact config paths and run names.
- Produces: repository structure documentation, four `not trained` experiment records, exact Linux training commands, evaluation gates, and final validation evidence.

- [ ] **Step 1: Document the four config responsibilities in `CODE_STRUCTURE.md`**

Add one paragraph naming all four files and recording these exact responsibilities:

```text
C0: unchanged continuation behavior from Jul08_12 model_16000.pt.
H: only commands.resampling_time=30.0.
W: only stand/jump-centered profile weights.
S: only [0.925, 0.95, 0.975, 1.0] schedule/window/threshold fields.
All: additional 2000 iterations, expected model_18000.pt, strict profile gate disabled.
```

Explain that complete-payload equivalence is enforced by `test_selective_walk_disturb100_causal_attribution_json_contracts`, preventing accidental PPO/reward/command-anchor changes.

- [ ] **Step 2: Add the planned batch to the experiment progress document**

Create a `July12 Planned Causal-Attribution Batch` section containing:

- common source checkpoint and the four config/run names;
- status `not trained` for every arm;
- the single-variable matrix;
- preservation gates: average task `>=30`, average fall `<=0.15`, per-preset fall `<=0.35`;
- forced `0.925` harmful threshold: versus C0, task delta `<=-5` and fall delta `>=+0.10`;
- forced `1.0` eligibility: forced `0.925` task `>=27` and fall `<=0.15`;
- failure diagnostic trigger: any preset fall `>=0.50`;
- note that T/G strict-gate phase is deferred until first-stage results exist.

Include these four literal commands in the finished document:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe \
  --checkpoint=16000 \
  --cfg_override_json configs/ablation/selective_walk_disturb100_causal_control.json \
  --run_name selective_walk_disturb100_causal_control \
  --max_iterations=2000

CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe \
  --checkpoint=16000 \
  --cfg_override_json configs/ablation/selective_walk_disturb100_hold30_only.json \
  --run_name selective_walk_disturb100_hold30_only \
  --max_iterations=2000

CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe \
  --checkpoint=16000 \
  --cfg_override_json configs/ablation/selective_walk_disturb100_stand_jump_weights_only.json \
  --run_name selective_walk_disturb100_stand_jump_weights_only \
  --max_iterations=2000

CUDA_VISIBLE_DEVICES=3 conda run -n hugwbc --no-capture-output python legged_gym/scripts/train.py \
  --task=r2amp --headless --seed=0 \
  --resume \
  --load_run Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe \
  --checkpoint=16000 \
  --cfg_override_json configs/ablation/selective_walk_disturb100_high_start_schedule_only.json \
  --run_name selective_walk_disturb100_high_start_schedule_only \
  --max_iterations=2000
```

Explicitly require checking the first `Loading model from` line before allowing each run to continue.

- [ ] **Step 3: Run focused and full verification**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m pytest tests/test_amp_training_contracts.py::test_selective_walk_disturb100_causal_attribution_json_contracts -q
python -m pytest tests/test_amp_training_contracts.py -q
git diff --check
```

Expected: focused test passes; full contract file has zero semantic failures (known Windows temporary-directory ACL failures, if they recur unchanged, must be reported separately and not hidden); diff check produces no errors.

- [ ] **Step 4: Sync and query CodeGraph**

Run:

```powershell
codegraph sync E:\codebase\VR_Teleoperation
codegraph query "selective walk causal attribution C0 hold30 weights schedule"
```

Expected: sync exits `0`; query surfaces the new contract test or updated documentation. If CodeGraph cannot index JSON/doc content, verify the exact names with `rg` and report the limitation.

- [ ] **Step 5: Commit Task 2**

```powershell
git add CODE_STRUCTURE.md docs/experiments/r2_amp_experiment_progress.md
git commit -m "docs: record selective-walk causal experiment batch"
```

- [ ] **Step 6: Verify final branch state**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: only pre-existing unrelated changes remain; the two implementation commits appear after the design/plan commits.
