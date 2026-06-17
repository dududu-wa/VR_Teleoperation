# Motion Prior Experts Design

## Goal

Replace the current single mixed AMP discriminator with a single policy plus
multiple hard-routed motion-prior discriminators. The first implementation keeps
the actor-critic policy, PPO storage, task rewards, and existing style reward
schedule/cap unchanged. Only the AMP motion prior is split by command semantics.

## Primary Research Basis

- Lu et al., 2026, "Unified Walking, Running, and Recovery for Humanoids via
  State-Dependent Adversarial Motion Priors", https://arxiv.org/abs/2605.18611.
  This is the closest reference for this design. It extends AMP by routing
  training transitions through a state-dependent gate to separate discriminators.
- Peng et al., 2021, "AMP: Adversarial Motion Priors for Stylized Physics-Based
  Character Control", https://arxiv.org/abs/2104.02180. This is the base method
  used by the current repository: a discriminator produces an auxiliary style
  reward from motion examples.
- Wu et al., 2026, "Multi-Gait Learning for Humanoid Robots Using Reinforcement
  Learning with Selective Adversarial Motion Prior",
  https://arxiv.org/abs/2604.19102. This motivates the ablation that some gaits
  may benefit from AMP while highly dynamic gaits may be over-constrained.
- Peng et al., 2018, "DeepMimic: Example-Guided Deep Reinforcement Learning of
  Physics-Based Character Skills", https://arxiv.org/abs/1804.02717. This
  motivates keeping motion references aligned with the skill being trained.

Secondary background only:

- Ho and Ermon, 2016, "Generative Adversarial Imitation Learning",
  https://arxiv.org/abs/1606.03476.
- Jacobs et al., 1991, "Adaptive Mixtures of Local Experts", Neural
  Computation.
- Choi and Han, 2021, "MCL-GAN: Generative Adversarial Networks with Multiple
  Specialized Discriminators", https://arxiv.org/abs/2107.07260.

## Public Code To Inspect

These repos are references, not direct copy targets:

- `xbpeng/MimicKit`: current lightweight motion imitation framework with AMP,
  DeepMimic, ASE, SMP, and related methods. It is the best modern code reference.
- `NVlabs/ProtoMotions`: modern large-scale humanoid motion learning framework.
  Useful for motion dataset and humanoid training organization.
- `nv-tlabs/ASE`: older AMP/ASE implementation. Its README points users to
  MimicKit for newer implementations.
- `xbpeng/DeepMimic`: classic motion imitation code. Useful as conceptual
  background for matching policies and reference motions.
- `isaac-sim/IsaacGymEnvs`: official Isaac Gym environments. Useful for AMP in
  Isaac Gym style stacks, but archived.
- `leggedrobotics/legged_gym` and `roboterax/humanoid-gym`: useful for local
  environment/config style, not for AMP expert routing.

## Current Repository Facts

- `legged_gym/utils/motion_loader.py` recursively scans `.npz` motion files.
  Pointing it at the root mixes all clips; pointing it at a subdirectory isolates
  one category.
- `legged_gym/envs/r2/r2.py` currently initializes one `self._motion_loader` and
  `collect_reference_motions()` samples only from that loader.
- `rsl_rl/rsl_rl/runners/on_policy_runner.py` currently creates one
  `AMPDiscriminator` and one `AMPReplayBuffer`.
- `rsl_rl/rsl_rl/algorithms/amp_ppo.py` currently computes all style rewards
  with one discriminator and trains that discriminator from one replay buffer.
- `legged_gym/scripts/evaluate.py` directly calls `runner.discriminator(amp_obs)`;
  multi-expert training requires routed evaluation too.

## Architecture

The implementation adds per-expert AMP components:

```text
commands
  -> get_amp_expert_ids()
  -> infos["amp_expert_id"]
  -> AMPPPO.process_env_step()
  -> discriminator[expert_id](amp_obs)
  -> style reward schedule/cap already in AMPPPO
  -> PPO storage receives mixed task/style reward
```

Discriminator updates also become per-expert:

```text
for each expert:
    agent samples = replay_buffer[expert]
    reference samples = motion_loader[expert]
    update discriminator[expert]
```

The checkpoint format adds dictionaries:

```text
discriminator_state_dicts = {"walk": ..., "run": ..., "jump": ...}
disc_optimizer_state_dicts = {"walk": ..., "run": ..., "jump": ...}
```

Legacy keys remain supported:

```text
discriminator_state_dict
disc_optimizer_state_dict
```

## Config Contract

Default `r2amp` enables experts:

```python
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
```

If `motion_experts` is absent or empty, the current single `motion_file` and
single discriminator behavior remains available for legacy runs and ablations.

## Hard Routing Rule

First version:

```text
jump: commands[:, 4] == 0 and
      (commands[:, 6] >= expert_jump_swing_height_threshold or
       commands[:, 7] > expert_jump_body_height_threshold)

run:  not jump and
      (abs(commands[:, 0]) > expert_run_velocity_threshold or
       commands[:, 3] >= expert_run_frequency_threshold)

walk: otherwise
```

Standing commands route to `walk` until a real `stand` motion expert exists.

## Logging

Required metrics:

```text
amp_expert_fraction/walk
amp_expert_fraction/run
amp_expert_fraction/jump
style_reward_contrib/walk
style_reward_contrib/run
style_reward_contrib/jump
disc_loss/walk
disc_loss/run
disc_loss/jump
disc_agent_logit/walk
disc_agent_logit/run
disc_agent_logit/jump
disc_ref_logit/walk
disc_ref_logit/run
disc_ref_logit/jump
disc_update_skipped/walk
disc_update_skipped/run
disc_update_skipped/jump
```

These metrics make the expert routing auditable: routing balance, expert data
coverage, discriminator strength, and style reward contribution are all visible.

## Failure Handling

- If `motion_experts` is configured, every listed path must exist and contain at
  least one `.npz`; otherwise initialization fails early.
- If `infos["amp_expert_id"]` is missing while multiple experts are active,
  `AMPPPO.process_env_step()` raises `KeyError`.
- If a replay buffer does not have enough samples for an expert discriminator
  update, that expert update is skipped and `disc_update_skipped/<expert>` is
  logged.

## Ablation Matrix

- `mixed_single_disc`: current mixed prior baseline.
- `walk_single_disc`: walk-only prior baseline.
- `expert_hard_gate_walk_run_jump`: main experiment.
- `expert_hard_gate_walk_run`: no jump expert; jump/hop transitions fall back to
  walk or run by rule.
- `expert_hard_gate_selective_walk`: selective AMP; only walk expert receives
  style reward while run/jump style contribution is zero. This directly tests
  the selective AMP claim that dynamic gaits can be over-constrained.
- `expert_hard_gate_no_style_warmup`: main experiment with warmup disabled.

## Validation

Repository-level contract tests must run without Isaac Gym import. They should
verify config fields, environment routing source, runner checkpoint keys, AMPPPO
expert routing, evaluate routing, JSON configs, and `CODE_STRUCTURE.md` updates.

Full simulator validation remains a training-machine task:

```powershell
python legged_gym/scripts/train.py --task r2amp --cfg_override_json configs/ablation/expert_hard_gate_walk_run_jump.json --run_name expert_hard_gate_walk_run_jump
python legged_gym/scripts/evaluate.py --task r2amp --load_run <run> --checkpoint model_best_task.pt --preset walk_slow --preset run --preset jump
```
