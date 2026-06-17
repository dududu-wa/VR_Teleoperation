# Motion Prior Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build state/command-dependent, hard-routed AMP motion-prior experts for R2 using one policy and multiple discriminators.

**Architecture:** Keep the current actor-critic and PPO path. Add per-expert `MotionLoader`, `AMPDiscriminator`, `AMPReplayBuffer`, command-based expert ids, routed style rewards, routed discriminator updates, routed evaluation metrics, and checkpoint compatibility.

**Tech Stack:** Python, PyTorch, Isaac Gym environment code, existing `legged_gym` and `rsl_rl` AMP modules.

---

## File Structure

- Modify `legged_gym/envs/r2/r2_amp_config.py`: declare expert motion paths, hard-route thresholds, and selective AMP toggle fields.
- Modify `legged_gym/envs/r2/r2.py`: initialize per-expert motion loaders, route commands to expert ids, emit `infos["amp_expert_id"]`, and sample reference motions by expert.
- Modify `rsl_rl/rsl_rl/runners/on_policy_runner.py`: create per-expert discriminators and replay buffers; save/load expert checkpoint dictionaries with legacy compatibility.
- Modify `rsl_rl/rsl_rl/algorithms/amp_ppo.py`: route style reward and discriminator updates by expert id; support per-expert style enable flags.
- Modify `legged_gym/scripts/evaluate.py`: compute AMP style and discriminator metrics through routed discriminators.
- Modify `tests/test_amp_training_contracts.py`: add no-IsaacGym contract tests.
- Modify `CODE_STRUCTURE.md`: document the multi-expert AMP pipeline down to function-level responsibilities.
- Create `configs/ablation/expert_hard_gate_walk_run_jump.json`.
- Create `configs/ablation/expert_hard_gate_walk_run.json`.
- Create `configs/ablation/expert_hard_gate_selective_walk.json`.
- Create `configs/ablation/expert_hard_gate_no_style_warmup.json`.

### Task 1: Config Contract

**Files:**
- Modify: `legged_gym/envs/r2/r2_amp_config.py`
- Modify: `tests/test_amp_training_contracts.py`

- [ ] **Step 1: Write the failing config contract test**

Add this test:

```python
def test_r2_amp_config_declares_motion_experts():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2_amp_config.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    fields = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    fields.add(target.id)
    assert {
        "motion_experts",
        "default_motion_expert",
        "expert_run_velocity_threshold",
        "expert_run_frequency_threshold",
        "expert_jump_swing_height_threshold",
        "expert_jump_body_height_threshold",
        "expert_style_enabled",
    }.issubset(fields)
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: fails because the expert config fields do not exist yet.

- [ ] **Step 3: Add config fields**

Add to both `R2AmpCfg.amp` and `R2AmpCfgPPO.amp`:

```python
        # Lu et al. 2026 use state-dependent AMP routing to separate motion
        # priors. Keep one policy, but split discriminators by command semantics.
        motion_experts = {
            "walk": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/walk",
            "run": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/run",
            "jump": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/jump",
        }
        default_motion_expert = "walk"
        expert_run_velocity_threshold = 1.0
        expert_run_frequency_threshold = 2.0
        expert_jump_swing_height_threshold = 0.18
        expert_jump_body_height_threshold = 0.02
        expert_style_enabled = {"walk": True, "run": True, "jump": True}
```

- [ ] **Step 4: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all existing tests and the new config test pass.

### Task 2: Environment Expert Routing

**Files:**
- Modify: `legged_gym/envs/r2/r2.py`
- Modify: `tests/test_amp_training_contracts.py`

- [ ] **Step 1: Write the failing environment contract test**

Add:

```python
def test_r2_env_exposes_amp_expert_routing_contract():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2.py").read_text(encoding="utf-8")
    assert "def get_amp_expert_ids" in source
    assert "amp_expert_id" in source
    assert "expert_ids=None" in source
    assert "_motion_loaders" in source
    assert "expert_jump_swing_height_threshold" in source
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: fails on missing environment routing contract.

- [ ] **Step 3: Initialize per-expert motion loaders**

In the AMP init block in `R2Robot._init_buffers()`, replace the single-loader
initialization with:

```python
            self.amp_expert_names = []
            self._motion_loaders = {}
            motion_experts = getattr(self.cfg.amp, "motion_experts", None)
            if motion_experts:
                for expert_name, expert_path in motion_experts.items():
                    resolved_path = expert_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                    self._motion_loaders[expert_name] = MotionLoader(resolved_path, self.device)
                    self.amp_expert_names.append(expert_name)
                default_expert = getattr(self.cfg.amp, "default_motion_expert", self.amp_expert_names[0])
                if default_expert not in self._motion_loaders:
                    raise ValueError(f"AMP default motion expert is not configured: {default_expert}")
                self.default_amp_expert_name = default_expert
                self.default_amp_expert_id = self.amp_expert_names.index(default_expert)
                self._motion_loader = self._motion_loaders[default_expert]
            else:
                motion_file = self.cfg.amp.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                self._motion_loader = MotionLoader(motion_file, self.device)
                self.amp_expert_names = ["default"]
                self._motion_loaders = {"default": self._motion_loader}
                self.default_amp_expert_name = "default"
                self.default_amp_expert_id = 0
```

- [ ] **Step 4: Add `get_amp_expert_ids()`**

Add near existing AMP observation helpers:

```python
    def get_amp_expert_ids(self):
        """Route each env to a motion-prior expert from command semantics."""
        if not hasattr(self, "amp_expert_names"):
            raise RuntimeError("AMP experts are not initialized.")
        expert_ids = torch.full(
            (self.num_envs,),
            int(self.default_amp_expert_id),
            dtype=torch.long,
            device=self.device,
        )
        name_to_id = {name: idx for idx, name in enumerate(self.amp_expert_names)}
        if "jump" in name_to_id:
            jump_mask = (self.commands[:, 4] == 0) & (
                (self.commands[:, 6] >= self.cfg.amp.expert_jump_swing_height_threshold)
                | (self.commands[:, 7] > self.cfg.amp.expert_jump_body_height_threshold)
            )
            expert_ids[jump_mask] = name_to_id["jump"]
        if "run" in name_to_id:
            run_mask = (
                (torch.abs(self.commands[:, 0]) > self.cfg.amp.expert_run_velocity_threshold)
                | (self.commands[:, 3] >= self.cfg.amp.expert_run_frequency_threshold)
            )
            if "jump" in name_to_id:
                run_mask = run_mask & (expert_ids != name_to_id["jump"])
            expert_ids[run_mask] = name_to_id["run"]
        return expert_ids
```

- [ ] **Step 5: Emit `amp_expert_id`**

Where `self.extras["amp_obs"]` is written, add:

```python
            self.extras["amp_expert_id"] = self.get_amp_expert_ids()
```

- [ ] **Step 6: Extend reference sampling**

Change:

```python
    def collect_reference_motions(self, num_samples, current_times=None):
```

to:

```python
    def collect_reference_motions(self, num_samples, current_times=None, expert_ids=None):
```

Keep old behavior if `expert_ids is None`. If expert ids are provided, allocate
the result tensor, group rows by expert id, sample each group from the matching
loader, and scatter the sampled AMP observations back to the original row order.

- [ ] **Step 7: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

### Task 3: Runner Multi-Expert Wiring

**Files:**
- Modify: `rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `tests/test_amp_training_contracts.py`

- [ ] **Step 1: Write the failing runner contract test**

Add:

```python
def test_runner_wires_amp_motion_experts():
    source = (ROOT_DIR / "rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(
        encoding="utf-8"
    )
    assert "discriminators" in source
    assert "amp_replay_buffers" in source
    assert "discriminator_state_dicts" in source
    assert "disc_optimizer_state_dicts" in source
    assert "expert_style_enabled" in source
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: fails on missing runner multi-expert fields.

- [ ] **Step 3: Create per-expert modules**

In `_init_amp()`, build:

```python
        expert_names = list(getattr(self.env, "amp_expert_names", ["default"]))
        self.discriminators = torch.nn.ModuleDict()
        self.amp_replay_buffers = {}
        for expert_name in expert_names:
            self.discriminators[expert_name] = AMPDiscriminator(
                amp_obs_dim=amp_obs_size,
                hidden_dims=amp_cfg.get("disc_hidden_dims", [1024, 512]),
            ).to(self.device)
            self.amp_replay_buffers[expert_name] = AMPReplayBuffer(
                buffer_size=amp_cfg.get("replay_buffer_size", 1000000),
                amp_obs_size=amp_obs_size,
                device=self.device,
            )
        self.discriminator = self.discriminators[expert_names[0]]
        self.amp_replay_buffer = self.amp_replay_buffers[expert_names[0]]
```

- [ ] **Step 4: Pass expert structures into AMPPPO**

Call `AMPPPO` with:

```python
            discriminators=self.discriminators,
            amp_replay_buffers=self.amp_replay_buffers,
            expert_style_enabled=amp_cfg.get("expert_style_enabled", None),
```

Do not pass the old positional `discriminator` and `amp_replay_buffer` arguments
after this change.

- [ ] **Step 5: Save checkpoint dictionaries**

In `save()`, if AMP is enabled:

```python
            save_dict["discriminator_state_dicts"] = {
                name: disc.state_dict() for name, disc in self.discriminators.items()
            }
            save_dict["disc_optimizer_state_dicts"] = {
                name: opt.state_dict() for name, opt in self.alg.disc_optimizers.items()
            }
            save_dict["discriminator_state_dict"] = self.discriminator.state_dict()
            save_dict["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
```

- [ ] **Step 6: Load new and legacy checkpoints**

In `load()`, prefer `discriminator_state_dicts`. If absent, load legacy
`discriminator_state_dict` into `self.discriminator`. Do the same for
optimizer dictionaries.

- [ ] **Step 7: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

### Task 4: AMPPPO Routed Style Reward and Updates

**Files:**
- Modify: `rsl_rl/rsl_rl/algorithms/amp_ppo.py`
- Modify: `tests/test_amp_training_contracts.py`

- [ ] **Step 1: Write the failing AMPPPO contract test**

Add:

```python
def test_amp_ppo_routes_by_expert_id():
    source = (ROOT_DIR / "rsl_rl/rsl_rl/algorithms/amp_ppo.py").read_text(
        encoding="utf-8"
    )
    assert "amp_expert_id" in source
    assert "disc_optimizers" in source
    assert "expert_style_enabled" in source
    assert "style_reward_contrib/" in source
    assert "disc_update_skipped/" in source
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: fails on missing routed expert logic.

- [ ] **Step 3: Update constructor contract**

Change `AMPPPO.__init__` to accept:

```python
        discriminators,
        amp_replay_buffers,
        env,
        expert_style_enabled=None,
```

Store:

```python
        self.discriminators = discriminators
        self.expert_names = list(self.discriminators.keys())
        self.discriminator = self.discriminators[self.expert_names[0]]
        self.amp_replay_buffers = amp_replay_buffers
        self.amp_replay_buffer = self.amp_replay_buffers[self.expert_names[0]]
        self.expert_style_enabled = {
            name: True for name in self.expert_names
        }
        if expert_style_enabled is not None:
            self.expert_style_enabled.update(expert_style_enabled)
        self.disc_optimizers = {
            name: torch.optim.AdamW(
                discriminator.parameters(),
                lr=disc_learning_rate,
                weight_decay=disc_weight_decay,
            )
            for name, discriminator in self.discriminators.items()
        }
        self.disc_optimizer = self.disc_optimizers[self.expert_names[0]]
        self.amp_expert_id_collector = []
```

- [ ] **Step 4: Route discriminator scores**

Add helper:

```python
    def _routed_discriminator(self, amp_obs, expert_ids):
        scores = torch.empty(amp_obs.shape[0], 1, device=amp_obs.device)
        for expert_idx, expert_name in enumerate(self.expert_names):
            mask = expert_ids == expert_idx
            if torch.any(mask):
                scores[mask] = self.discriminators[expert_name](amp_obs[mask])
        return scores
```

- [ ] **Step 5: Route style reward in `process_env_step()`**

Read:

```python
        expert_ids = infos.get("amp_expert_id")
        if expert_ids is None:
            if len(self.expert_names) > 1:
                raise KeyError("AMP experts require infos['amp_expert_id']")
            expert_ids = torch.zeros(amp_obs.shape[0], dtype=torch.long, device=amp_obs.device)
```

Use `_routed_discriminator(amp_obs, expert_ids)` instead of
`self.discriminator(amp_obs)`.

If `self.expert_style_enabled[expert_name]` is false, zero the final style
reward contribution for rows routed to that expert, but still collect AMP obs so
the logs can show routing fractions.

- [ ] **Step 6: Insert replay samples by expert**

In `update()`, concatenate collected `amp_obs` and `amp_expert_id`, group by
expert id, and insert each group into the matching replay buffer.

- [ ] **Step 7: Update discriminators by expert**

Replace single `_update_discriminator()` with a loop over experts. For each
expert:

```python
if replay_buffer.count < half_batch:
    metrics[f"disc_update_skipped/{expert_name}"] = 1.0
    continue
expert_ids = torch.full((half_batch,), expert_idx, dtype=torch.long, device=self.device)
ref_amp_obs_3d = self.env.collect_reference_motions(half_batch, expert_ids=expert_ids)
```

Train `self.discriminators[expert_name]` with
`self.disc_optimizers[expert_name]`.

- [ ] **Step 8: Add per-expert metrics**

Record:

```python
metrics[f"disc_loss/{expert_name}"] = disc_loss.item()
metrics[f"disc_agent_logit/{expert_name}"] = agent_logit.mean().item()
metrics[f"disc_ref_logit/{expert_name}"] = ref_logit.mean().item()
metrics[f"amp_expert_fraction/{expert_name}"] = expert_fraction
metrics[f"style_reward_contrib/{expert_name}"] = expert_style_contrib
```

- [ ] **Step 9: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

### Task 5: Evaluation Routing

**Files:**
- Modify: `legged_gym/scripts/evaluate.py`
- Modify: `tests/test_amp_training_contracts.py`

- [ ] **Step 1: Write the failing evaluate contract test**

Add:

```python
def test_evaluate_uses_routed_amp_discriminator():
    source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "amp_expert_id" in source
    assert "discriminators" in source
    assert "_routed_discriminator_score" in source
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: fails on missing evaluate routing.

- [ ] **Step 3: Add routed score helper**

Add:

```python
def _routed_discriminator_score(runner, amp_obs, expert_ids=None):
    discriminators = getattr(runner, "discriminators", None)
    if discriminators is None:
        return runner.discriminator(amp_obs)
    expert_names = list(discriminators.keys())
    if expert_ids is None:
        expert_ids = torch.zeros(amp_obs.shape[0], dtype=torch.long, device=amp_obs.device)
    scores = torch.empty(amp_obs.shape[0], 1, device=amp_obs.device)
    for expert_idx, expert_name in enumerate(expert_names):
        mask = expert_ids == expert_idx
        if torch.any(mask):
            scores[mask] = discriminators[expert_name](amp_obs[mask])
    return scores
```

- [ ] **Step 4: Replace direct discriminator calls**

Replace `runner.discriminator(amp_obs)` and `runner.discriminator(ref_obs...)`
with `_routed_discriminator_score(...)`. For reference samples, pass the same
expert ids used for the policy batch.

- [ ] **Step 5: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

### Task 6: Ablation Configs and Documentation

**Files:**
- Create: `configs/ablation/expert_hard_gate_walk_run_jump.json`
- Create: `configs/ablation/expert_hard_gate_walk_run.json`
- Create: `configs/ablation/expert_hard_gate_selective_walk.json`
- Create: `configs/ablation/expert_hard_gate_no_style_warmup.json`
- Modify: `CODE_STRUCTURE.md`

- [ ] **Step 1: Add `expert_hard_gate_walk_run_jump.json`**

```json
{
  "env": {
    "amp": {
      "motion_experts": {
        "walk": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/walk",
        "run": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/run",
        "jump": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/jump"
      },
      "default_motion_expert": "walk",
      "expert_style_enabled": {"walk": true, "run": true, "jump": true}
    }
  },
  "train": {
    "amp": {
      "expert_style_enabled": {"walk": true, "run": true, "jump": true}
    }
  },
  "notes": "State/command-dependent hard-routed AMP prior for walk/run/jump."
}
```

- [ ] **Step 2: Add `expert_hard_gate_walk_run.json`**

Use only `walk` and `run` in `motion_experts`. Keep `default_motion_expert` as
`walk`. Keep both expert styles enabled.

- [ ] **Step 3: Add `expert_hard_gate_selective_walk.json`**

Use walk/run/jump experts, but set:

```json
"expert_style_enabled": {"walk": true, "run": false, "jump": false}
```

This tests the selective AMP hypothesis that dynamic gaits can be over-
constrained by style priors.

- [ ] **Step 4: Add `expert_hard_gate_no_style_warmup.json`**

Use walk/run/jump experts and set:

```json
"style_reward_start_after": 0,
"style_reward_warmup_iterations": 0
```

- [ ] **Step 5: Update `CODE_STRUCTURE.md`**

Document:

```text
commands -> R2Robot.get_amp_expert_ids() -> infos["amp_expert_id"]
motion_experts -> per-expert MotionLoader
OnPolicyRunner._init_amp() -> per-expert AMPDiscriminator and AMPReplayBuffer
AMPPPO.process_env_step() -> routed style reward
AMPPPO._update_discriminator() -> routed discriminator updates
evaluate.py -> routed discriminator score
```

- [ ] **Step 6: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

### Task 7: Module-Level Verification

**Files:**
- No new files.

- [ ] **Step 1: Run compile check**

Run:

```powershell
python -m py_compile legged_gym\envs\r2\r2_amp_config.py legged_gym\envs\r2\r2.py rsl_rl\rsl_rl\algorithms\amp_ppo.py rsl_rl\rsl_rl\runners\on_policy_runner.py legged_gym\scripts\evaluate.py tests\test_amp_training_contracts.py
```

Expected: exits with code 0.

- [ ] **Step 2: Run contract test**

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python tests\test_amp_training_contracts.py
```

Expected: all tests pass.

- [ ] **Step 3: Run JSON parse check**

Run:

```powershell
python -c "import json, pathlib; [json.loads(p.read_text()) for p in pathlib.Path('configs/ablation').glob('expert_hard_gate*.json')]; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Attempt codegraph refresh**

Run:

```powershell
codegraph index -p .
```

Expected: success. If it fails with `unable to open database file`, report it as
tooling failure and keep the `rg`/compile/test evidence.
