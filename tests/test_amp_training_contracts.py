import sys
import types
import ast
import json
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "rsl_rl"))
isaacgym_stub = types.ModuleType("isaacgym")
isaacgym_torch_utils_stub = types.ModuleType("isaacgym.torch_utils")
sys.modules.setdefault("isaacgym", isaacgym_stub)
sys.modules.setdefault("isaacgym.torch_utils", isaacgym_torch_utils_stub)

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module


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


def test_run_disturb_sweep_helper_contract():
    script = (ROOT_DIR / "scripts/run_run_disturb_sweep.ps1").read_text(
        encoding="utf-8"
    )
    assert "--preset run" in script
    assert "--eval_disturb_ratio" in script
    for token in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
        assert token in script


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
    test_evaluate_supports_forced_disturbance_sweep_metrics()
    test_expert_hard_gate_ablation_json_and_docs_contract()
    test_interrupt_disturb_release_can_bypass_terrain_curriculum_gate()
    test_staged_disturb_release_config_and_code_contract()
    test_second_staged_disturb_release_experiment_is_run_focused()
    test_run_recovery_staged_disturb_release_uses_per_stage_gates()
    test_eval_manifold_staged_disturb_release_uses_command_profile_mixture()
    test_eval_manifold_conservative_disturb_release_json_contract()
    test_run_disturb_sweep_helper_contract()
    test_next_batch_command_hold_ablation_json_contract()
