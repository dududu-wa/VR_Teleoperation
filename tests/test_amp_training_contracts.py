import sys
import types
import ast
import contextlib
import csv
import importlib.util
import json
import uuid
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "rsl_rl"))
isaacgym_stub = types.ModuleType("isaacgym")
isaacgym_torch_utils_stub = types.ModuleType("isaacgym.torch_utils")
sys.modules.setdefault("isaacgym", isaacgym_stub)
sys.modules.setdefault("isaacgym.torch_utils", isaacgym_torch_utils_stub)

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module


def _repo_test_dir(label):
    """Create a writable test directory without Windows tempfile ACL changes."""
    path = ROOT_DIR / ".test_tmp" / f"{label}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _bare_amp_algo():
    alg = object.__new__(AMPPPO)
    alg.style_reward_weight = 1.0
    alg.style_reward_time_scale = 0.02
    alg.style_reward_start_after = 10
    alg.style_reward_warmup_iterations = 10
    alg.style_reward_min_task_reward = None
    alg.style_reward_max_task_ratio = 0.25
    return alg


def test_amp_style_schedule_and_task_ratio_gate():
    alg = _bare_amp_algo()
    style_reward = torch.ones(2)
    task_reward = torch.tensor([-0.004, -1.0])
    task_reward_weighted = task_reward.clone()

    alg.set_learning_iteration(9)
    weighted, task_gate = alg._weight_style_reward(
        style_reward,
        task_reward,
        task_reward_weighted,
    )
    assert torch.allclose(weighted, torch.zeros_like(weighted))
    assert torch.allclose(task_gate, torch.ones_like(task_gate))

    alg.set_learning_iteration(10)
    weighted, _ = alg._weight_style_reward(
        style_reward,
        task_reward,
        task_reward_weighted,
    )
    assert torch.allclose(weighted, torch.tensor([0.001, 0.002]), atol=1e-7)

    alg.set_learning_iteration(19)
    weighted, _ = alg._weight_style_reward(
        style_reward,
        task_reward,
        task_reward_weighted,
    )
    assert torch.allclose(weighted, torch.tensor([0.001, 0.02]), atol=1e-7)

    alg.style_reward_min_task_reward = 0.0
    weighted, task_gate = alg._weight_style_reward(
        style_reward,
        torch.tensor([-0.1, 0.1]),
        torch.tensor([-0.1, 0.1]),
    )
    assert torch.allclose(task_gate, torch.tensor([0.0, 1.0]))
    assert torch.allclose(weighted, torch.tensor([0.0, 0.02]), atol=1e-7)


def test_runner_keeps_top_task_checkpoints():
    runner = object.__new__(OnPolicyRunner)
    runner.log_dir = "in_memory_log"
    runner.save_best_after = 0
    runner.save_best_task_checkpoint = True
    runner.save_top_task_checkpoints = 2
    runner.best_task_reward = float("-inf")
    runner.top_task_checkpoints = []
    runner._emit_log = lambda message: None
    saved = {}
    removed = set()

    def save_stub(path, infos=None):
        saved[path] = infos or {}

    runner.save = save_stub
    original_exists = runner_module.os.path.exists
    original_remove = runner_module.os.remove
    runner_module.os.path.exists = lambda path: path in saved and path not in removed
    runner_module.os.remove = lambda path: removed.add(path)
    try:
        runner._maybe_save_best_checkpoints(1, [1.0])
        runner._maybe_save_best_checkpoints(2, [2.0])
        runner._maybe_save_best_checkpoints(3, [0.5])
        runner._maybe_save_best_checkpoints(4, [3.0])
    finally:
        runner_module.os.path.exists = original_exists
        runner_module.os.remove = original_remove

    top_names = {
        Path(path).name
        for path in saved
        if Path(path).name.startswith("model_top_task_") and path not in removed
    }
    assert top_names == {"model_top_task_2.pt", "model_top_task_4.pt"}
    assert any(Path(path).name == "model_top_task_1.pt" for path in removed)
    best_path = next(path for path in saved if Path(path).name == "model_best_task.pt")
    assert saved[best_path]["best_metric_value"] == 3.0


def test_runner_wires_amp_schedule_config_to_algorithm():
    source = (ROOT_DIR / "rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    amp_call = None
    has_iteration_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "AMPPPO":
                amp_call = node
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_learning_iteration":
                has_iteration_call = True
    assert amp_call is not None
    keyword_names = {keyword.arg for keyword in amp_call.keywords}
    assert {
        "style_reward_start_after",
        "style_reward_warmup_iterations",
        "style_reward_min_task_reward",
        "style_reward_max_task_ratio",
    }.issubset(keyword_names)
    assert has_iteration_call


def test_r2_amp_config_declares_schedule_and_topk_fields():
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
        "save_top_task_checkpoints",
        "style_reward_start_after",
        "style_reward_warmup_iterations",
        "style_reward_min_task_reward",
        "style_reward_max_task_ratio",
    }.issubset(fields)


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


def test_r2_env_exposes_amp_expert_routing_contract():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2.py").read_text(encoding="utf-8")
    assert "def get_amp_expert_ids" in source
    assert "amp_expert_id" in source
    assert "expert_ids=None" in source
    assert "_motion_loaders" in source
    assert "expert_jump_swing_height_threshold" in source
    assert "current_times.detach().cpu().numpy()" in source
    assert "AMP expert loader schema mismatch" in source
    assert "torch.randint(len(self.amp_expert_names)" in source


def test_runner_wires_amp_motion_experts():
    source = (ROOT_DIR / "rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(
        encoding="utf-8"
    )
    assert "discriminators" in source
    assert "amp_replay_buffers" in source
    assert "discriminator_state_dicts" in source
    assert "disc_optimizer_state_dicts" in source
    assert "expert_style_enabled" in source
    amp_call_index = source.index("self.alg = AMPPPO")
    storage_index = source.index("self.alg.init_storage", amp_call_index)
    assert amp_call_index < storage_index


def test_amp_ppo_routes_by_expert_id():
    source = (ROOT_DIR / "rsl_rl/rsl_rl/algorithms/amp_ppo.py").read_text(
        encoding="utf-8"
    )
    assert "amp_expert_id" in source
    assert "disc_optimizers" in source
    assert "expert_style_enabled" in source
    assert "style_reward_contrib/" in source
    assert "disc_update_skipped/" in source


def test_ppo_supports_teacher_policy_retention_loss():
    ppo_source = (ROOT_DIR / "rsl_rl/rsl_rl/algorithms/ppo.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT_DIR / "rsl_rl/rsl_rl/runners/on_policy_runner.py"
    ).read_text(encoding="utf-8")

    assert "teacher_policy_retention_coef" in ppo_source
    assert "capture_teacher_policy" in ppo_source
    assert "_teacher_retention_loss" in ppo_source
    assert "teacher_policy_retention_loss" in ppo_source
    assert "teacher_policy_retention_skipped" in ppo_source
    assert "copy.deepcopy" in ppo_source
    assert "Learning without Forgetting" in ppo_source
    assert "capture_teacher_policy()" in runner_source


class _FixedTeacherPolicy:
    def __init__(self, action_mean):
        self.action_mean = action_mean

    def act_inference(self, observations, masks=None, privileged_obs=None):
        assert privileged_obs is not None
        return self.action_mean, None


def test_teacher_retention_loss_matches_action_mean_mse():
    alg = object.__new__(PPO)
    alg.teacher_policy_retention_coef = 0.25
    current_mean = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    teacher_mean = torch.tensor([[1.0, 1.0], [0.5, -1.0]])
    alg.teacher_actor_critic = _FixedTeacherPolicy(teacher_mean)

    loss, skipped = alg._teacher_retention_loss(
        torch.zeros(2, 3),
        torch.zeros(2, 4),
        None,
        current_mean,
    )

    assert skipped is False
    assert torch.allclose(loss, 0.25 * (current_mean - teacher_mean).pow(2).mean())

    alg.teacher_actor_critic = None
    loss, skipped = alg._teacher_retention_loss(
        torch.zeros(2, 3),
        torch.zeros(2, 4),
        None,
        current_mean,
    )
    assert skipped is True
    assert torch.allclose(loss, torch.tensor(0.0))


def test_amp_ppo_resolves_expert_ids_before_collector_mutation():
    alg = object.__new__(AMPPPO)
    alg.expert_names = ["walk", "run"]
    try:
        alg._resolve_expert_ids(
            {"amp_expert_id": torch.tensor([0, 2])},
            2,
            torch.device("cpu"),
        )
    except ValueError as exc:
        assert "Invalid AMP expert ids" in str(exc)
        assert "2" in str(exc)
    else:
        raise AssertionError("invalid AMP expert id should raise ValueError")

    try:
        alg._resolve_expert_ids({}, 2, torch.device("cpu"))
    except KeyError as exc:
        assert "amp_expert_id" in str(exc)
    else:
        raise AssertionError("multi-expert AMP should require amp_expert_id")

    alg.expert_names = ["walk"]
    expert_ids = alg._resolve_expert_ids({}, 3, torch.device("cpu"))
    assert torch.equal(expert_ids, torch.zeros(3, dtype=torch.long))


def test_evaluate_uses_routed_amp_discriminator():
    source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "amp_expert_id" in source
    assert "discriminators" in source
    assert "_routed_discriminator_score" in source
    assert "_apply_expert_style_enabled" in source
    assert "expert_style_enabled" in source


def test_evaluate_dtw_is_opt_in():
    source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "compute_dtw" in source
    assert "args.compute_dtw," in source
    assert "reward_acc," in source
    assert "if compute_dtw:" in source


def test_evaluate_can_record_termination_reason_diagnostics():
    evaluate_source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    helpers_source = (ROOT_DIR / "legged_gym/utils/helpers.py").read_text(
        encoding="utf-8"
    )

    assert "--record_termination_reasons" in helpers_source
    assert "TERMINATION_REASON_FIELDS" in evaluate_source
    assert "_detect_termination_reason" in evaluate_source
    assert "_summarize_termination_reasons" in evaluate_source
    assert "termination_reason" in evaluate_source
    assert "termination_detail" in evaluate_source
    assert 'getattr(env, "body_names", [])' in evaluate_source
    assert "termination_reasons.csv" in evaluate_source
    assert "termination_reasons.json" in evaluate_source


def test_evaluate_can_record_state_trace_diagnostics():
    evaluate_source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    helpers_source = (ROOT_DIR / "legged_gym/utils/helpers.py").read_text(
        encoding="utf-8"
    )
    r2_source = (ROOT_DIR / "legged_gym/envs/r2/r2.py").read_text(
        encoding="utf-8"
    )

    assert "--record_state_trace" in helpers_source
    assert "--state_trace_window_steps" in helpers_source
    assert "STATE_TRACE_FIELDS" in evaluate_source
    assert "_init_state_trace_buffers" in evaluate_source
    assert "_append_state_trace" in evaluate_source
    assert "_flush_state_trace_episode" in evaluate_source
    assert "state_trace.csv" in evaluate_source
    assert "state_trace.json" in evaluate_source
    assert "steps_until_done" in evaluate_source
    assert "contact_force_max" in evaluate_source
    assert "max_episodes" in evaluate_source
    assert "len(episode_rows) < max_episodes" in evaluate_source
    assert "record_eval_pre_reset_state" in evaluate_source
    assert "record_eval_pre_reset_state" in r2_source
    assert "_cache_eval_pre_reset_state" in r2_source


def test_play_supports_finite_recorded_diagnostic_runs():
    helpers_source = (ROOT_DIR / "legged_gym/utils/helpers.py").read_text(
        encoding="utf-8"
    )
    play_source = (ROOT_DIR / "legged_gym/scripts/play.py").read_text(
        encoding="utf-8"
    )

    assert "--play_seconds" in helpers_source
    assert "--record_seconds" in helpers_source
    assert "args.play_seconds" in play_source
    assert "args.record_seconds" in play_source
    assert "record_duration_s" in play_source
    assert "env_cfg.env.episode_length_s = play_seconds" in play_source


def test_evaluate_supports_forced_disturbance_sweep_metrics():
    helper_source = (ROOT_DIR / "legged_gym/utils/helpers.py").read_text(
        encoding="utf-8"
    )
    eval_source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "--eval_disturb_ratio" in helper_source
    assert "survival_time_mean_s" in eval_source
    assert "_apply_eval_disturbance(env, args, done_ids)" in eval_source
    assert "_disable_eval_disturbance(env, env_ids)" in eval_source
    tree = ast.parse(eval_source)
    disable_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_disable_eval_disturbance"
    )
    assert any(
        isinstance(node, ast.With)
        and "torch.inference_mode" in ast.get_source_segment(eval_source, node)
        for node in ast.walk(disable_fn)
    )
    assert "env.disturb_masks[env_ids] = False" in eval_source
    assert "env.interrupt_mask[env_ids] = False" in eval_source
    assert "env.use_disturb = bool(" in eval_source
    assert 'getattr(cfg_disturb, "use_disturb"' in eval_source
    assert "env.disturb_rad_curriculum[env_ids] = float(args.eval_disturb_ratio)" in eval_source


def test_evaluate_can_record_reward_term_diagnostics():
    helper_source = (ROOT_DIR / "legged_gym/utils/helpers.py").read_text(
        encoding="utf-8"
    )
    eval_source = (ROOT_DIR / "legged_gym/scripts/evaluate.py").read_text(
        encoding="utf-8"
    )
    env_source = (ROOT_DIR / "legged_gym/envs/r2/r2.py").read_text(
        encoding="utf-8"
    )

    assert "--record_reward_terms" in helper_source
    assert "reward_terms.csv" in eval_source
    assert "reward_terms.json" in eval_source
    assert "_collect_reward_terms" in eval_source
    assert "_summarize_reward_terms" in eval_source
    assert "env.record_reward_terms = bool(args.record_reward_terms)" in eval_source
    assert "self.record_reward_terms" in env_source
    assert "self.last_reward_terms" in env_source


def test_expert_hard_gate_ablation_json_and_docs_contract():
    ablation_dir = ROOT_DIR / "configs/ablation"
    paths = sorted(ablation_dir.glob("expert_hard_gate*.json"))
    assert {path.name for path in paths} == {
        "expert_hard_gate_no_style_warmup.json",
        "expert_hard_gate_selective_walk.json",
        "expert_hard_gate_walk_run.json",
        "expert_hard_gate_walk_run_jump.json",
    }
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload.get("notes"), str) and payload["notes"]
        assert "motion_experts" in payload["env"]["amp"]
        assert "expert_style_enabled" in payload["env"]["amp"]
        assert "expert_style_enabled" in payload["train"]["amp"]

    docs = (ROOT_DIR / "CODE_STRUCTURE.md").read_text(encoding="utf-8")
    for token in ("get_amp_expert_ids", "amp_expert_id", "_routed_discriminator_score"):
        assert token in docs


def test_interrupt_disturb_release_can_bypass_terrain_curriculum_gate():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2interrupt.py").read_text(
        encoding="utf-8"
    )
    config_source = (
        ROOT_DIR / "legged_gym/envs/r2/r2interrupt_config.py"
    ).read_text(encoding="utf-8")
    assert "start_by_curriculum = True" in config_source
    assert "self.start_disturb_by_curriculum" in source
    assert "disturb_ready_mask = ~heading_mask" in source
    assert "if not self.start_disturb_by_curriculum:" in source
    assert "disturb_ready_mask = torch.ones_like(heading_mask)" in source
    assert "disturb_allowed_mask = ~self.terrain_curriculum_mode" in source
    assert "torch.ones_like(self.terrain_curriculum_mode)" in source


def test_staged_disturb_release_config_and_code_contract():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2interrupt.py").read_text(
        encoding="utf-8"
    )
    config_source = (
        ROOT_DIR / "legged_gym/envs/r2/r2interrupt_config.py"
    ).read_text(encoding="utf-8")
    payload = json.loads(
        (ROOT_DIR / "configs/ablation/command_hold_staged_disturb_release.json").read_text(
            encoding="utf-8"
        )
    )
    assert "staged_release = False" in config_source
    assert "stage_monitor_expert = None" in config_source
    assert "self.staged_disturb_release" in source
    assert "self.staged_disturb_monitor_expert" in source
    assert "_staged_disturb_expert_mask" in source
    assert "_record_staged_disturb_episode_stats" in source
    assert "_maybe_advance_staged_disturb_release" in source
    assert "torch.clamp(self.disturb_rad_curriculum, max=stage_level)" in source
    assert payload["env"]["commands"]["curriculum"] is False
    assert payload["env"]["disturb"]["start_by_curriculum"] is False
    assert payload["env"]["disturb"]["staged_release"] is True
    assert payload["env"]["disturb"]["stage_levels"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert payload["env"]["disturb"]["stage_monitor_expert"] == "run"
    assert payload["train"]["runner"]["run_name"] == "command_hold_staged_disturb_release"


def test_second_staged_disturb_release_experiment_is_run_focused():
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_run_focused_staged_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    ranges = payload["env"]["commands"]["ranges"]
    disturb = payload["env"]["disturb"]

    assert payload["env"]["commands"]["curriculum"] is False
    assert payload["train"]["runner"]["run_name"] == "command_hold_run_focused_staged_disturb_release"
    assert payload["train"]["runner"]["max_iterations"] == 8000
    assert disturb["start_by_curriculum"] is False
    assert disturb["staged_release"] is True
    assert disturb["stage_levels"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert disturb["stage_monitor_expert"] == "run"
    assert ranges["lin_vel_x"][0] > 1.0
    assert ranges["gait_frequency"][0] >= 2.0
    assert abs(ranges["lin_vel_y"][0]) <= 0.2 and abs(ranges["lin_vel_y"][1]) <= 0.2
    assert ranges["foot_swing_height"][1] < 0.18
    assert ranges["body_height"][1] <= 0.02


def test_run_recovery_staged_disturb_release_uses_per_stage_gates():
    source = (ROOT_DIR / "legged_gym/envs/r2/r2interrupt.py").read_text(
        encoding="utf-8"
    )
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_run_recovery_staged_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    ranges = payload["env"]["commands"]["ranges"]
    disturb = payload["env"]["disturb"]
    stage_levels = disturb["stage_levels"]
    min_returns = disturb["stage_min_task_return"]
    max_fall_rates = disturb["stage_max_fall_rate"]

    assert "_expand_staged_disturb_gate_values" in source
    assert "_current_staged_disturb_gate" in source
    assert payload["train"]["runner"]["run_name"] == "command_hold_run_recovery_staged_disturb_release"
    assert payload["train"]["runner"]["max_iterations"] == 8000
    assert disturb["start_by_curriculum"] is False
    assert disturb["staged_release"] is True
    assert disturb["stage_monitor_expert"] == "run"
    assert len(stage_levels) == len(min_returns) == len(max_fall_rates)
    assert stage_levels[1] - stage_levels[0] < 0.25
    assert min_returns[0] < min_returns[-1]
    assert max_fall_rates[0] > max_fall_rates[-1]
    assert ranges["lin_vel_x"][0] < 1.0 < ranges["lin_vel_x"][1]
    assert ranges["gait_frequency"][0] < 2.0 < ranges["gait_frequency"][1]
    assert abs(ranges["lin_vel_y"][0]) <= 0.2 and abs(ranges["lin_vel_y"][1]) <= 0.2
    assert ranges["foot_swing_height"][1] < 0.18
    assert ranges["body_height"][1] < 0.02


def test_eval_manifold_staged_disturb_release_uses_command_profile_mixture():
    config_source = (ROOT_DIR / "legged_gym/envs/r2/r2_config.py").read_text(
        encoding="utf-8"
    )
    interrupt_config_source = (
        ROOT_DIR / "legged_gym/envs/r2/r2interrupt_config.py"
    ).read_text(encoding="utf-8")
    interrupt_source = (ROOT_DIR / "legged_gym/envs/r2/r2interrupt.py").read_text(
        encoding="utf-8"
    )
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_eval_manifold_staged_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    ranges = payload["env"]["commands"]["ranges"]
    profiles = payload["env"]["commands"]["profile_mixture"]
    disturb = payload["env"]["disturb"]
    names = {profile["name"] for profile in profiles}

    assert "profile_mixture = None" in config_source
    assert "_apply_command_profile_mixture" in interrupt_source
    assert "torch.multinomial" in interrupt_source
    assert 'profile.get("standing", False)' in interrupt_source
    assert 'profile.get("name") == "stand"' not in interrupt_source
    assert "command_profile_ids" in interrupt_source
    assert "stage_monitor_profiles" in interrupt_source
    assert "staged_disturb_failure_windows" in interrupt_source
    assert "stage_init_curriculum_to_level = False" in interrupt_config_source
    assert "staged_disturb_init_curriculum_to_level" in interrupt_source
    assert "torch.full(" in interrupt_source
    assert payload["train"]["runner"]["run_name"] == "command_hold_eval_manifold_staged_disturb_release"
    assert payload["train"]["runner"]["max_iterations"] == 8000
    assert names == {
        "stand",
        "walk_slow",
        "walk_fast",
        "run",
        "jump",
        "turn_left",
        "strafe_right",
    }
    assert abs(sum(float(profile["weight"]) for profile in profiles) - 1.0) < 1e-6
    assert all(len(profile["command"]) == 10 for profile in profiles)
    assert all(len(profile["jitter"]) == 10 for profile in profiles)
    assert ranges["lin_vel_x"][0] <= 0.0 and ranges["lin_vel_x"][1] >= 1.6
    assert ranges["lin_vel_y"][0] <= -0.3 and ranges["lin_vel_y"][1] >= 0.3
    assert ranges["ang_vel_yaw"][0] <= -0.6 and ranges["ang_vel_yaw"][1] >= 0.6
    assert ranges["gait_frequency"][0] <= 1.6 and ranges["gait_frequency"][1] >= 3.0
    assert ranges["foot_swing_height"][0] <= 0.08 and ranges["foot_swing_height"][1] >= 0.2
    assert ranges["body_height"][1] >= 0.03
    assert ranges["body_pitch"][1] >= 0.03
    assert disturb["start_by_curriculum"] is False
    assert disturb["staged_release"] is True
    assert disturb["stage_monitor_expert"] in ("", None)
    assert set(disturb["stage_monitor_profiles"]) == names
    assert disturb["stage_regress_on_failure"] is True
    assert disturb["stage_regress_patience"] >= 2
    assert len(disturb["stage_levels"]) == len(disturb["stage_min_task_return"])
    assert len(disturb["stage_levels"]) == len(disturb["stage_max_fall_rate"])


def test_staged_disturb_all_profile_gate_requires_each_profile_to_pass():
    script_path = ROOT_DIR / "legged_gym/envs/r2/staged_disturb_gate.py"
    spec = importlib.util.spec_from_file_location(
        "staged_disturb_gate",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    aggregate = {
        "episode_count": 4096,
        "return_sum": 28.0 * 4096,
        "fall_sum": 0.05 * 4096,
    }
    profile_stats = {
        "jump": {
            "episode_count": 1024,
            "return_sum": 10.0 * 1024,
            "fall_sum": 0.40 * 1024,
        },
        "stand": {
            "episode_count": 1024,
            "return_sum": 26.0 * 1024,
            "fall_sum": 0.06 * 1024,
        },
    }

    assert module.staged_disturb_window_ready(
        episode_count=aggregate["episode_count"],
        min_episodes=1024,
        profile_stats=profile_stats,
        require_all_profiles=True,
    )
    assert module.staged_disturb_window_passes(
        **aggregate,
        min_episodes=1024,
        min_task_return=20.0,
        max_fall_rate=0.10,
        profile_stats=profile_stats,
        require_all_profiles=False,
    )
    assert not module.staged_disturb_window_passes(
        **aggregate,
        min_episodes=1024,
        min_task_return=20.0,
        max_fall_rate=0.10,
        profile_stats=profile_stats,
        require_all_profiles=True,
    )

    passing_profiles = {
        name: {
            "episode_count": stats["episode_count"],
            "return_sum": 24.0 * stats["episode_count"],
            "fall_sum": 0.08 * stats["episode_count"],
        }
        for name, stats in profile_stats.items()
    }
    assert module.staged_disturb_window_passes(
        **aggregate,
        min_episodes=1024,
        min_task_return=20.0,
        max_fall_rate=0.10,
        profile_stats=passing_profiles,
        require_all_profiles=True,
    )

    under_sampled = dict(passing_profiles)
    under_sampled["jump"] = dict(under_sampled["jump"], episode_count=1023)
    assert not module.staged_disturb_window_ready(
        episode_count=aggregate["episode_count"],
        min_episodes=1024,
        profile_stats=under_sampled,
        require_all_profiles=True,
    )
    try:
        module.validate_profile_gate_resampling(
            require_all_profiles=True,
            resampling_time_s=10.0,
            dt=0.02,
            max_episode_length=1000,
        )
    except ValueError as exc:
        assert "longer than one episode" in str(exc)
    else:
        raise AssertionError("strict profile gates must reject mid-episode profile resampling")
    module.validate_profile_gate_resampling(
        require_all_profiles=True,
        resampling_time_s=30.0,
        dt=0.02,
        max_episode_length=1000,
    )
    module.validate_profile_gate_resampling(
        require_all_profiles=False,
        resampling_time_s=10.0,
        dt=0.02,
        max_episode_length=1000,
    )

    interrupt_config_source = (
        ROOT_DIR / "legged_gym/envs/r2/r2interrupt_config.py"
    ).read_text(encoding="utf-8")
    interrupt_source = (
        ROOT_DIR / "legged_gym/envs/r2/r2interrupt.py"
    ).read_text(encoding="utf-8")
    assert "stage_require_all_monitor_profiles = False" in interrupt_config_source
    assert "staged_disturb_require_all_monitor_profiles" in interrupt_source
    assert "staged_disturb_profile_stats" in interrupt_source


def test_eval_manifold_conservative_disturb_release_json_contract():
    # This adjacent config isolates disturbance pressure from eval-profile coverage.
    base_payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_eval_manifold_staged_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    profiles = payload["env"]["commands"]["profile_mixture"]
    disturb = payload["env"]["disturb"]
    names = {profile["name"] for profile in profiles}
    weights = {profile["name"]: float(profile["weight"]) for profile in profiles}

    assert payload["train"]["runner"]["run_name"] == "command_hold_eval_manifold_conservative_disturb_release"
    assert payload["train"]["runner"]["max_iterations"] == 8000
    assert names == {
        profile["name"]
        for profile in base_payload["env"]["commands"]["profile_mixture"]
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["stand"] >= 0.18
    assert weights["jump"] >= 0.14
    assert disturb["start_by_curriculum"] is False
    assert disturb["staged_release"] is True
    assert disturb["stage_monitor_expert"] in ("", None)
    assert set(disturb["stage_monitor_profiles"]) == names
    assert disturb["stage_regress_on_failure"] is True
    assert disturb["stage_min_episodes"] > base_payload["env"]["disturb"]["stage_min_episodes"]
    assert max(disturb["stage_levels"]) <= 0.75
    assert disturb["stage_levels"][1] - disturb["stage_levels"][0] <= 0.05
    assert len(disturb["stage_levels"]) == len(disturb["stage_min_task_return"])
    assert len(disturb["stage_levels"]) == len(disturb["stage_max_fall_rate"])
    assert disturb["stage_min_task_return"][0] < disturb["stage_min_task_return"][-1]
    assert disturb["stage_max_fall_rate"][0] > disturb["stage_max_fall_rate"][-1]


def test_selective_walk_eval_manifold_conservative_disturb_release_json_contract():
    # Keep the post-audit follow-up as a one-variable change from the conservative
    # eval-manifold config: selective-walk style reward replaces full-style reward.
    base_payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/command_hold_eval_manifold_conservative_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json"
        ).read_text(encoding="utf-8")
    )
    profiles = payload["env"]["commands"]["profile_mixture"]
    profile_names = {profile["name"] for profile in profiles}
    base_profile_names = {
        profile["name"] for profile in base_payload["env"]["commands"]["profile_mixture"]
    }
    disturb = payload["env"]["disturb"]
    env_style = payload["env"]["amp"]["expert_style_enabled"]
    train_style = payload["train"]["amp"]["expert_style_enabled"]

    assert payload["train"]["runner"]["run_name"] == "selective_walk_eval_manifold_conservative_disturb_release"
    assert payload["train"]["runner"]["max_iterations"] == 8000
    assert payload["train"]["runner"]["save_top_task_checkpoints"] == 3
    assert payload["env"]["commands"]["curriculum"] is False
    assert profile_names == base_profile_names
    assert abs(sum(float(profile["weight"]) for profile in profiles) - 1.0) < 1e-6
    assert disturb == base_payload["env"]["disturb"]
    assert env_style == {"walk": True, "run": False, "jump": False}
    assert train_style == {"walk": True, "run": False, "jump": False}
    assert payload["env"]["amp"]["default_motion_expert"] == "walk"


def test_selective_walk_retention_probe_json_contracts():
    ablation_dir = ROOT_DIR / "configs/ablation"
    expected_max_iterations = {
        "selective_walk_resume_null_control.json": 8000,
        "selective_walk_profile_task_only_probe.json": 8000,
        "selective_walk_profile_teacher_retention_probe.json": 8000,
        "selective_walk_profile_teacher_retention_coef010_probe.json": 8000,
        "selective_walk_profile_teacher_retention_disturb075_probe.json": 8000,
        "selective_walk_profile_teacher_retention_disturb100_probe.json": 4000,
    }
    payloads = {}
    for filename, expected_iterations in expected_max_iterations.items():
        path = ablation_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[filename] = payload
        assert isinstance(payload.get("notes"), str) and payload["notes"]
        assert payload["train"]["runner"]["max_iterations"] == expected_iterations
        assert payload["train"]["runner"]["save_top_task_checkpoints"] == 3
        assert "selective_walk" in payload["train"]["runner"]["run_name"]
        assert payload["env"]["amp"]["expert_style_enabled"] == {
            "walk": True,
            "run": False,
            "jump": False,
        }
        assert payload["train"]["amp"]["expert_style_enabled"] == {
            "walk": True,
            "run": False,
            "jump": False,
        }

    null_control = payloads["selective_walk_resume_null_control.json"]
    assert "profile_mixture" not in null_control["env"]["commands"]
    assert null_control["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.0

    task_only = payloads["selective_walk_profile_task_only_probe.json"]
    assert task_only["env"]["commands"]["profile_mixture"]
    assert task_only["train"]["amp"]["style_reward_weight"] == 0.0
    assert task_only["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.0

    retention = payloads["selective_walk_profile_teacher_retention_probe.json"]
    assert retention["env"]["commands"]["profile_mixture"]
    assert retention["train"]["amp"]["style_reward_weight"] == 0.0
    assert retention["train"]["algorithm"]["teacher_policy_retention_coef"] > 0.0
    assert retention["train"]["algorithm"]["teacher_policy_retention_coef"] <= 1.0

    # Keep the July05 follow-up configs as adjacent controls: one changes only
    # retention strength, the other keeps retention strength and releases disturb.
    weak_retention = payloads["selective_walk_profile_teacher_retention_coef010_probe.json"]
    assert weak_retention["train"]["runner"]["run_name"] == (
        "selective_walk_profile_teacher_retention_coef010_probe"
    )
    assert weak_retention["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.10
    assert weak_retention["env"]["disturb"]["stage_levels"] == [0.0]
    assert weak_retention["train"]["amp"]["style_reward_weight"] == 0.0

    disturb_probe = payloads["selective_walk_profile_teacher_retention_disturb075_probe.json"]
    assert disturb_probe["train"]["runner"]["run_name"] == (
        "selective_walk_profile_teacher_retention_disturb075_probe"
    )
    assert disturb_probe["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.25
    assert disturb_probe["env"]["disturb"]["stage_levels"] == [0.0, 0.15, 0.3, 0.5, 0.75]
    assert disturb_probe["env"]["disturb"]["stage_monitor_profiles"] == [
        "stand",
        "walk_slow",
        "walk_fast",
        "run",
        "jump",
        "turn_left",
        "strafe_right",
    ]
    assert disturb_probe["train"]["amp"]["style_reward_weight"] == 0.0

    # Continue the successful July05 disturbance-trained checkpoint instead of
    # restarting the curriculum from the original Jun17 warm-start.
    disturb100_probe = payloads["selective_walk_profile_teacher_retention_disturb100_probe.json"]
    assert disturb100_probe["train"]["runner"]["run_name"] == (
        "selective_walk_profile_teacher_retention_disturb100_probe"
    )
    assert disturb100_probe["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.25
    assert disturb100_probe["env"]["disturb"]["stage_init_curriculum_to_level"] is True
    assert disturb100_probe["env"]["disturb"]["stage_levels"] == [0.75, 0.85, 0.925, 1.0]
    assert disturb100_probe["env"]["disturb"]["stage_min_task_return"] == [20.0, 22.0, 24.0, 25.0]
    assert disturb100_probe["env"]["disturb"]["stage_max_fall_rate"] == [0.18, 0.14, 0.10, 0.08]
    assert disturb100_probe["env"]["disturb"]["stage_monitor_profiles"] == (
        disturb_probe["env"]["disturb"]["stage_monitor_profiles"]
    )
    assert disturb100_probe["train"]["amp"]["style_reward_weight"] == 0.0
    assert "Jul05_16-01-08_selective_walk_profile_teacher_retention_disturb075_probe" in (
        disturb100_probe["notes"]
    )
    assert "model_12000.pt" in disturb100_probe["notes"]


def test_selective_walk_profile_guard_recovery_json_contract():
    payload = json.loads(
        (
            ROOT_DIR
            / "configs/ablation/"
            "selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery.json"
        ).read_text(encoding="utf-8")
    )
    profiles = payload["env"]["commands"]["profile_mixture"]
    weights = {profile["name"]: float(profile["weight"]) for profile in profiles}
    disturb = payload["env"]["disturb"]

    assert payload["train"]["runner"]["run_name"] == (
        "selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery"
    )
    assert payload["train"]["runner"]["max_iterations"] == 4000
    assert payload["train"]["algorithm"]["teacher_policy_retention_coef"] == 0.25
    assert payload["train"]["amp"]["style_reward_weight"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights == {
        "stand": 0.25,
        "walk_slow": 0.10,
        "walk_fast": 0.12,
        "run": 0.12,
        "jump": 0.25,
        "turn_left": 0.08,
        "strafe_right": 0.08,
    }
    assert disturb["stage_init_curriculum_to_level"] is True
    assert disturb["stage_levels"] == [0.925, 0.95, 0.975, 1.0]
    assert disturb["stage_min_episodes"] == 1024
    assert disturb["stage_min_task_return"] == [18.0, 20.0, 22.0, 24.0]
    assert disturb["stage_max_fall_rate"] == [0.20, 0.16, 0.12, 0.10]
    assert disturb["stage_monitor_profiles"] == ["stand", "jump"]
    assert disturb["stage_require_all_monitor_profiles"] is True
    assert payload["env"]["commands"]["resampling_time"] == 30.0
    assert disturb["stage_regress_on_failure"] is True
    assert "rewards" not in payload["env"]
    assert "Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe" in (
        payload["notes"]
    )
    assert "model_16000.pt" in payload["notes"]
    assert "model_20000.pt" in payload["notes"]


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


def test_selective_walk_retention_probe_algorithm_keys_exist_in_cfg_schema():
    source = ast.parse(
        (ROOT_DIR / "legged_gym/envs/base/legged_robot_config.py").read_text(
            encoding="utf-8"
        )
    )
    ppo_class = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "LeggedRobotCfgPPO"
    )
    algorithm_class = next(
        node
        for node in ppo_class.body
        if isinstance(node, ast.ClassDef) and node.name == "algorithm"
    )
    schema_keys = {
        target.id
        for node in algorithm_class.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    ablation_dir = ROOT_DIR / "configs/ablation"
    for filename in (
        "selective_walk_resume_null_control.json",
        "selective_walk_profile_task_only_probe.json",
        "selective_walk_profile_teacher_retention_probe.json",
        "selective_walk_profile_teacher_retention_coef010_probe.json",
        "selective_walk_profile_teacher_retention_disturb075_probe.json",
        "selective_walk_profile_teacher_retention_disturb100_probe.json",
    ):
        payload = json.loads((ablation_dir / filename).read_text(encoding="utf-8"))
        for key in payload["train"]["algorithm"]:
            assert key in schema_keys, f"{filename}: train.algorithm.{key}"


def test_run_disturb_sweep_helper_contract():
    script = (ROOT_DIR / "scripts/run_run_disturb_sweep.ps1").read_text(
        encoding="utf-8"
    )
    assert 'Jun19/Jun19_16-09-11_scratch_command_hold' in script
    assert 'July19/Jun19_16-09-11_scratch_command_hold' not in script
    assert "--preset run" in script
    assert "--eval_disturb_ratio" in script
    for token in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
        assert token in script


def test_selective_walk_followup_eval_plan_script_contract():
    script_path = ROOT_DIR / "scripts/plan_selective_walk_followup_eval.py"
    spec = importlib.util.spec_from_file_location(
        "plan_selective_walk_followup_eval",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    commands = module.build_eval_plan(
        load_run="JunXX/JunXX_selective_walk_followup",
        checkpoint="8000",
        output_prefix="outputs/eval/JuneXX_selective_walk_followup_8000",
    )
    labels = [command.label for command in commands]
    command_text = "\n".join(command.command for command in commands)

    assert labels == [
        "baseline_full7",
        "full7_disturb075",
        "full7_disturb090",
        "full7_disturb0925",
        "full7_disturb095",
        "full7_disturb100",
        "failure_diagnostics_disturb0925",
        "failure_diagnostics_disturb095",
        "failure_diagnostics_disturb100",
    ]
    assert all("--task=r2amp" in command.command for command in commands)
    assert all("--num_envs=64" in command.command for command in commands)
    assert all("--num_episodes=64" in command.command for command in commands)
    assert all("--episode_seconds=10" in command.command for command in commands)
    assert "configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json" in command_text
    assert "--eval_disturb_ratio=0.75" in command_text
    assert "--eval_disturb_ratio=0.925" in command_text
    assert "--eval_disturb_ratio=1.0" in command_text
    assert "--record_termination_reasons" in command_text
    assert "--record_state_trace" in command_text
    assert "--preset stand --preset run --preset jump --preset strafe_right" in command_text


def test_jul08_disturb100_diagnostic_command_contract():
    script_path = ROOT_DIR / "scripts/run_jul08_disturb100_diagnostics.py"
    spec = importlib.util.spec_from_file_location(
        "run_jul08_disturb100_diagnostics",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    command = module.build_diagnostic_command()
    command_text = " ".join(command)

    assert command[:4] == ["wsl.exe", "-d", "Ubuntu-22.04", "--cd"]
    assert "Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe" in command_text
    assert "--checkpoint=16000" in command
    assert (
        "configs/ablation/selective_walk_profile_teacher_retention_disturb100_probe.json"
        in command
    )
    assert "--preset" in command
    assert command.count("--preset") == 2
    assert "jump" in command and "stand" in command
    assert "--eval_disturb_ratio=1.0" in command
    assert "--record_termination_reasons" in command
    assert "--record_state_trace" in command
    assert "--state_trace_window_steps=50" in command
    assert "--sim_device=cpu" in command
    assert "--rl_device=cpu" in command
    assert "--num_envs=64" in command
    assert "--num_episodes=64" in command
    assert "--episode_seconds=10" in command
    assert (
        "outputs/eval/July08_12_selective_walk_profile_teacher_retention_disturb100_probe_"
        "16000_jump_stand_disturb100_failure_diagnostics"
        in command
    )

    dry_run = module.run_diagnostic(execute=False)
    assert dry_run["executed"] is False
    executed = []
    result = module.run_diagnostic(
        execute=True,
        command_runner=lambda argv, check: executed.append((argv, check)),
    )
    assert result["executed"] is True
    assert executed == [(command, True)]


def test_failure_diagnostics_summary_uses_terminal_trace_rows():
    script_path = ROOT_DIR / "scripts/summarize_failure_diagnostics.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_failure_diagnostics",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # tempfile.TemporaryDirectory creates an unreadable ACL in this Windows
    # environment, so retain the same context shape with a repo-local path.
    with contextlib.nullcontext(_repo_test_dir("failure_diagnostics_summary")) as output_dir:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "preset_name",
                    "num_episodes",
                    "task_return_mean",
                    "fall_rate",
                    "survival_time_mean_s",
                    "base_height_violation_rate",
                    "roll_pitch_violation_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "preset_name": "jump",
                        "num_episodes": "4",
                        "task_return_mean": "7.0",
                        "fall_rate": "0.75",
                        "survival_time_mean_s": "4.0",
                        "base_height_violation_rate": "0.2",
                        "roll_pitch_violation_rate": "0.1",
                    },
                    {
                        "preset_name": "stand",
                        "num_episodes": "4",
                        "task_return_mean": "12.0",
                        "fall_rate": "0.5",
                        "survival_time_mean_s": "6.0",
                        "base_height_violation_rate": "0.1",
                        "roll_pitch_violation_rate": "0.2",
                    },
                ]
            )

        with (output_dir / "termination_reasons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "preset_name",
                    "termination_reason",
                    "termination_detail",
                    "count",
                    "rate",
                    "mean_survival_time_s",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "preset_name": "jump",
                        "termination_reason": "contact",
                        "termination_detail": "base_link",
                        "count": "3",
                        "rate": "0.75",
                        "mean_survival_time_s": "2.0",
                    },
                    {
                        "preset_name": "jump",
                        "termination_reason": "timeout",
                        "termination_detail": "",
                        "count": "1",
                        "rate": "0.25",
                        "mean_survival_time_s": "10.0",
                    },
                    {
                        "preset_name": "stand",
                        "termination_reason": "orientation",
                        "termination_detail": "roll_pitch",
                        "count": "2",
                        "rate": "0.5",
                        "mean_survival_time_s": "3.0",
                    },
                    {
                        "preset_name": "stand",
                        "termination_reason": "timeout",
                        "termination_detail": "",
                        "count": "2",
                        "rate": "0.5",
                        "mean_survival_time_s": "10.0",
                    },
                ]
            )

        with (output_dir / "state_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "preset_name",
                    "episode_index",
                    "steps_until_done",
                    "base_z",
                    "roll",
                    "pitch",
                    "lin_vel_error",
                    "yaw_vel_error",
                    "contact_force_max",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "preset_name": "jump",
                        "episode_index": "0",
                        "steps_until_done": "1",
                        "base_z": "0.8",
                        "roll": "0.1",
                        "pitch": "0.1",
                        "lin_vel_error": "1.0",
                        "yaw_vel_error": "1.0",
                        "contact_force_max": "0.0",
                    },
                    {
                        "preset_name": "jump",
                        "episode_index": "0",
                        "steps_until_done": "0",
                        "base_z": "0.5",
                        "roll": "-0.4",
                        "pitch": "0.6",
                        "lin_vel_error": "2.0",
                        "yaw_vel_error": "3.0",
                        "contact_force_max": "100.0",
                    },
                    {
                        "preset_name": "stand",
                        "episode_index": "0",
                        "steps_until_done": "0",
                        "base_z": "0.6",
                        "roll": "0.8",
                        "pitch": "-0.2",
                        "lin_vel_error": "0.5",
                        "yaw_vel_error": "1.5",
                        "contact_force_max": "20.0",
                    },
                    *[
                        {
                            "preset_name": "jump",
                            "episode_index": str(episode_index),
                            "steps_until_done": "0",
                            "base_z": "0.5",
                            "roll": "-0.4",
                            "pitch": "0.6",
                            "lin_vel_error": "2.0",
                            "yaw_vel_error": "3.0",
                            "contact_force_max": "100.0",
                        }
                        for episode_index in range(1, 4)
                    ],
                    *[
                        {
                            "preset_name": "stand",
                            "episode_index": str(episode_index),
                            "steps_until_done": "0",
                            "base_z": "0.6",
                            "roll": "0.8",
                            "pitch": "-0.2",
                            "lin_vel_error": "0.5",
                            "yaw_vel_error": "1.5",
                            "contact_force_max": "20.0",
                        }
                        for episode_index in range(1, 4)
                    ],
                ]
            )

        summary = module.summarize_failure_diagnostics(str(output_dir))
        rows = {row["preset"]: row for row in summary["rows"]}

        assert summary["presets"] == ["jump", "stand"]
        assert rows["jump"]["contact_base_link_rate"] == 0.75
        assert rows["stand"]["orientation_roll_pitch_rate"] == 0.5
        assert rows["jump"]["mean_base_z_terminal"] == 0.5
        assert rows["jump"]["mean_abs_roll_terminal"] == 0.4
        assert rows["jump"]["max_contact_force_terminal"] == 100.0
        assert (output_dir / "failure_diagnostics_summary.csv").exists()
        assert (output_dir / "failure_diagnostics_summary.json").exists()

        termination_path = output_dir / "termination_reasons.csv"
        with termination_path.open(newline="", encoding="utf-8") as f:
            incomplete_rows = list(csv.DictReader(f))[:-1]
        with termination_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "preset_name",
                    "termination_reason",
                    "termination_detail",
                    "count",
                    "rate",
                    "mean_survival_time_s",
                ],
            )
            writer.writeheader()
            writer.writerows(incomplete_rows)
        try:
            module.summarize_failure_diagnostics(str(output_dir))
        except ValueError as exc:
            assert "termination count" in str(exc)
        else:
            raise AssertionError("incomplete termination accounting must fail")


def test_selective_walk_followup_train_plan_script_contract():
    script_path = ROOT_DIR / "scripts/plan_selective_walk_followup_train.py"
    spec = importlib.util.spec_from_file_location(
        "plan_selective_walk_followup_train",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    command = module.build_train_command()

    assert "wsl.exe -d Ubuntu-22.04" in command
    assert "legged_gym/scripts/train.py" in command
    assert "--task=r2amp" in command
    assert "--headless" in command
    assert "--sim_device=cpu" in command
    assert "--rl_device=cpu" in command
    assert "--resume" in command
    assert "--load_run Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk" in command
    assert "--checkpoint=-2" in command
    assert (
        "--cfg_override_json configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json"
        in command
    )
    assert "--run_name selective_walk_eval_manifold_conservative_disturb_release" in command
    assert "--max_iterations=4000" in command
    assert "smoke_sw_eval_manifold_conservative_disturb_release" not in command
    assert module.additional_iterations(resume_iteration=4000, target_iteration=8000) == 4000


def test_selective_walk_followup_eval_runner_requires_recommended_plan():
    script_path = ROOT_DIR / "scripts/run_selective_walk_followup_eval_plan.py"
    spec = importlib.util.spec_from_file_location(
        "run_selective_walk_followup_eval_plan",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    empty_audit = {
        "ready_for_evaluation": False,
        "recommended_eval_plan": [],
    }
    empty_plan = module.prepare_execution(empty_audit)
    assert empty_plan["can_execute"] is False
    assert "No recommended eval commands" in empty_plan["reason"]
    try:
        module.run_eval_plan(empty_audit, execute=True)
    except ValueError as exc:
        assert "No recommended eval commands" in str(exc)
    else:
        raise AssertionError("execute=True must refuse an audit without recommended commands")

    audit = {
        "ready_for_evaluation": True,
        "recommended_eval_plan": [
            {
                "label": "baseline_full7",
                "output_dir": "outputs/eval/example_baseline_full7",
                "command": "echo baseline",
            },
            {
                "label": "full7_disturb075",
                "output_dir": "outputs/eval/example_full7_disturb075",
                "command": "echo disturb",
            },
        ],
    }
    dry_run = module.run_eval_plan(audit, execute=False)
    assert dry_run["executed"] == 0
    assert dry_run["planned"] == 2
    assert "# baseline_full7" in "\n".join(dry_run["lines"])
    assert "echo disturb" in "\n".join(dry_run["lines"])

    executed = []
    result = module.run_eval_plan(
        audit,
        execute=True,
        command_runner=lambda command: executed.append(command),
    )
    assert result["planned"] == 2
    assert result["executed"] == 2
    assert executed == ["echo baseline", "echo disturb"]


def test_selective_walk_followup_eval_summary_script_contract():
    script_path = ROOT_DIR / "scripts/summarize_selective_walk_followup_eval.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_selective_walk_followup_eval",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with contextlib.nullcontext(_repo_test_dir("followup_eval_summary")) as tmp:
        tmp_path = Path(tmp)
        output_prefix = tmp_path / "selective_walk_followup_8000"
        summary_dir = tmp_path / "summary"

        baseline_dir = Path(f"{output_prefix}_baseline_full7")
        baseline_dir.mkdir()
        with (baseline_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["preset_name", "task_return_mean", "fall_rate", "survival_time_mean_s"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "preset_name": "stand",
                    "task_return_mean": "10",
                    "fall_rate": "0.0",
                    "survival_time_mean_s": "10",
                }
            )
            writer.writerow(
                {
                    "preset_name": "run",
                    "task_return_mean": "20",
                    "fall_rate": "0.25",
                    "survival_time_mean_s": "8",
                }
            )

        disturb_dir = Path(f"{output_prefix}_full7_disturb075")
        disturb_dir.mkdir()
        with (disturb_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["preset_name", "task_return_mean", "fall_rate", "survival_time_mean_s"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "preset_name": "jump",
                    "task_return_mean": "30",
                    "fall_rate": "0.125",
                    "survival_time_mean_s": "9",
                }
            )

        summary = module.summarize_eval_outputs(
            output_prefix=str(output_prefix),
            output_dir=str(summary_dir),
            require_all=False,
        )

        assert summary["expected_outputs"] == 9
        assert summary["present_outputs"] == 2
        assert summary["missing_outputs"] == 7
        rows = {row["label"]: row for row in summary["rows"]}
        assert rows["baseline_full7"]["status"] == "present"
        assert rows["baseline_full7"]["rows"] == 2
        assert rows["baseline_full7"]["task_return_mean"] == 15.0
        assert rows["baseline_full7"]["fall_rate"] == 0.125
        assert rows["baseline_full7"]["worst_fall_preset"] == "run"
        assert rows["failure_diagnostics_disturb100"]["status"] == "missing"
        assert (summary_dir / "selective_walk_followup_eval_summary.csv").exists()
        assert (summary_dir / "selective_walk_followup_eval_summary.json").exists()


def test_selective_walk_followup_readiness_audit_contract():
    script_path = ROOT_DIR / "scripts/audit_selective_walk_followup_readiness.py"
    spec = importlib.util.spec_from_file_location(
        "audit_selective_walk_followup_readiness",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with contextlib.nullcontext(_repo_test_dir("followup_readiness")) as tmp:
        tmp_path = Path(tmp)
        logs_root = tmp_path / "logs" / "r2_amp"
        eval_root = tmp_path / "outputs" / "eval"
        transient_run_dir = (
            logs_root
            / "JuneXX"
            / "JunXX_loading_only_selective_walk_eval_manifold_conservative_disturb_release"
        )
        transient_run_dir.mkdir(parents=True)
        (transient_run_dir / "train.log").write_text(
            "Loading model from: source/model_best_task.pt\n"
            "load_path: source/model_best_task.pt\n",
            encoding="utf-8",
        )

        run_dir = logs_root / "JuneXX" / "JunXX_selective_walk_eval_manifold_conservative_disturb_release"
        run_dir.mkdir(parents=True)
        for checkpoint in ("model_best_task.pt", "model_8000.pt", "model_top_task_1234.pt"):
            (run_dir / checkpoint).write_text("checkpoint placeholder", encoding="utf-8")

        output_prefix = eval_root / "JuneXX_selective_walk_followup_8000"
        eval_dir = Path(f"{output_prefix}_baseline_full7")
        eval_dir.mkdir(parents=True)
        (eval_dir / "metrics.csv").write_text(
            "preset_name,task_return_mean,fall_rate,survival_time_mean_s\n"
            "stand,1,0,10\n",
            encoding="utf-8",
        )

        audit_path = tmp_path / "audit.json"
        audit = module.audit_readiness(
            logs_root=str(logs_root),
            eval_root=str(eval_root),
            run_name_contains="selective_walk_eval_manifold_conservative_disturb_release",
            output_prefix=str(output_prefix),
            output_json=str(audit_path),
        )

        assert audit["runs_found"] == 2
        assert audit["checkpoint_count"] == 3
        statuses = {Path(run["run_dir"]).name: run["progress_status"] for run in audit["runs"]}
        artifact_sources = {
            Path(run["run_dir"]).name: run["artifact_source"] for run in audit["runs"]
        }
        assert statuses[transient_run_dir.name] == "load_only_no_training_progress"
        assert statuses[run_dir.name] == "checkpoint_present"
        assert artifact_sources[transient_run_dir.name] == "evaluate_checkpoint_load_log_dir"
        assert artifact_sources[run_dir.name] == "trained_run"
        assert audit["evaluation_complete"] is False
        assert audit["planned_eval_outputs"] == 9
        assert audit["present_eval_outputs"] == 1
        assert audit["missing_eval_outputs"] == 8
        assert audit["ready_for_evaluation"] is True
        assert audit["ready_for_completion"] is False
        assert audit["recommended_checkpoint"] == "8000"
        assert audit["recommended_load_run"] == "JuneXX/JunXX_selective_walk_eval_manifold_conservative_disturb_release"
        assert len(audit["recommended_eval_plan"]) == 9
        first_eval = audit["recommended_eval_plan"][0]
        assert first_eval["label"] == "baseline_full7"
        assert "--load_run JuneXX/JunXX_selective_walk_eval_manifold_conservative_disturb_release" in first_eval["command"]
        assert "--checkpoint=8000" in first_eval["command"]
        assert str(output_prefix) in first_eval["output_dir"]
        checkpoint_names = {checkpoint["checkpoint"] for checkpoint in audit["checkpoints"]}
        assert checkpoint_names == {"best_task", "8000", "top_task_1234"}
        assert audit_path.exists()


def test_evaluate_checkpoint_load_disables_training_log_dir():
    source_path = ROOT_DIR / "legged_gym/scripts/evaluate.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "make_alg_runner"
    ]
    assert calls, "evaluate.py should create the runner through task_registry.make_alg_runner"

    # Evaluation writes artifacts under --output_dir; checkpoint loading should not
    # create train-style log directories under logs/r2_amp.
    log_root_keywords = [
        keyword
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "log_root"
    ]
    assert len(log_root_keywords) == 1
    assert isinstance(log_root_keywords[0].value, ast.Constant)
    assert log_root_keywords[0].value.value is None


def test_task_registry_none_log_root_keeps_default_resume_lookup():
    source_path = ROOT_DIR / "legged_gym/utils/task_registry.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    task_registry_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskRegistry"
    )
    make_alg_runner = next(
        node
        for node in task_registry_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_alg_runner"
    )

    # log_root=None disables new runner logging, but resume checkpoint lookup
    # still needs the default experiment log root used by get_load_path().
    load_root_assignments = [
        node
        for node in ast.walk(make_alg_runner)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "load_root"
    ]
    assert load_root_assignments

    get_load_path_calls = [
        node
        for node in ast.walk(make_alg_runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_load_path"
    ]
    assert len(get_load_path_calls) == 1
    assert isinstance(get_load_path_calls[0].args[0], ast.Name)
    assert get_load_path_calls[0].args[0].id == "load_root"


def test_next_batch_command_hold_ablation_json_contract():
    ablation_dir = ROOT_DIR / "configs/ablation"
    expected = {
        "command_hold_controlled_disturb_release.json",
        "command_hold_no_push.json",
        "command_hold_conservative_penalty_ramp.json",
        "command_hold_style_lowcap.json",
    }
    for filename in expected:
        payload = json.loads((ablation_dir / filename).read_text(encoding="utf-8"))
        assert isinstance(payload.get("notes"), str) and payload["notes"]
        assert payload["env"]["commands"]["curriculum"] is False
        assert payload["train"]["runner"]["run_name"] == Path(filename).stem
        assert payload["train"]["runner"]["max_iterations"] == 8000
        assert payload["train"]["runner"]["save_top_task_checkpoints"] == 3
        assert "motion_experts" in payload["env"]["amp"]
        assert "expert_style_enabled" in payload["env"]["amp"]
        assert "expert_style_enabled" in payload["train"]["amp"]

    controlled = json.loads(
        (ablation_dir / "command_hold_controlled_disturb_release.json").read_text(
            encoding="utf-8"
        )
    )
    assert controlled["env"]["disturb"]["start_by_curriculum"] is False


if __name__ == "__main__":
    test_amp_style_schedule_and_task_ratio_gate()
    test_runner_keeps_top_task_checkpoints()
    test_runner_wires_amp_schedule_config_to_algorithm()
    test_r2_amp_config_declares_schedule_and_topk_fields()
    test_r2_amp_config_declares_motion_experts()
    test_r2_env_exposes_amp_expert_routing_contract()
    test_runner_wires_amp_motion_experts()
    test_amp_ppo_routes_by_expert_id()
    test_amp_ppo_resolves_expert_ids_before_collector_mutation()
    test_evaluate_uses_routed_amp_discriminator()
    test_evaluate_dtw_is_opt_in()
    test_evaluate_can_record_termination_reason_diagnostics()
    test_evaluate_can_record_state_trace_diagnostics()
    test_play_supports_finite_recorded_diagnostic_runs()
    test_evaluate_supports_forced_disturbance_sweep_metrics()
    test_expert_hard_gate_ablation_json_and_docs_contract()
    test_interrupt_disturb_release_can_bypass_terrain_curriculum_gate()
    test_staged_disturb_release_config_and_code_contract()
    test_second_staged_disturb_release_experiment_is_run_focused()
    test_run_recovery_staged_disturb_release_uses_per_stage_gates()
    test_eval_manifold_staged_disturb_release_uses_command_profile_mixture()
    test_eval_manifold_conservative_disturb_release_json_contract()
    test_selective_walk_eval_manifold_conservative_disturb_release_json_contract()
    test_run_disturb_sweep_helper_contract()
    test_selective_walk_followup_eval_plan_script_contract()
    test_selective_walk_followup_train_plan_script_contract()
    test_selective_walk_followup_eval_runner_requires_recommended_plan()
    test_selective_walk_followup_eval_summary_script_contract()
    test_selective_walk_followup_readiness_audit_contract()
    test_evaluate_checkpoint_load_disables_training_log_dir()
    test_task_registry_none_log_root_keeps_default_resume_lookup()
    test_next_batch_command_hold_ablation_json_contract()
