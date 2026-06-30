"""Audit whether the selective-walk follow-up is ready for formal evaluation.

The audit is deliberately filesystem-based: it answers whether a trained run
and planned evaluation outputs exist, without launching training or Isaac Gym.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_selective_walk_followup_eval import build_eval_plan


DEFAULT_RUN_NAME_CONTAINS = "selective_walk_eval_manifold_conservative_disturb_release"
DEFAULT_LOGS_ROOT = "logs/r2_amp"
DEFAULT_EVAL_ROOT = "outputs/eval"


def _relative_load_run(logs_root: Path, run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(logs_root)).replace("\\", "/")
    except ValueError:
        return str(run_dir).replace("\\", "/")


def _checkpoint_id(path: Path) -> Optional[str]:
    name = path.name
    if name == "model_best_task.pt":
        return "best_task"
    if name == "model_best_mixed.pt":
        return "best_mixed"
    if name.startswith("model_top_task_") and name.endswith(".pt"):
        return f"top_task_{name[len('model_top_task_'):-len('.pt')]}"
    if name.startswith("model_") and name.endswith(".pt"):
        return name[len("model_") : -len(".pt")]
    return None


def _find_runs(logs_root: Path, run_name_contains: str) -> List[Path]:
    if not logs_root.exists():
        return []
    return sorted(
        path
        for path in logs_root.rglob("*")
        if path.is_dir() and run_name_contains in path.name
    )


def _find_checkpoints(logs_root: Path, run_dir: Path) -> List[Dict[str, str]]:
    checkpoints = []
    for path in sorted(run_dir.glob("model*.pt")):
        checkpoint = _checkpoint_id(path)
        if checkpoint is None:
            continue
        checkpoints.append(
            {
                "checkpoint": checkpoint,
                "path": str(path),
                "run_dir": str(run_dir),
                "load_run": _relative_load_run(logs_root, run_dir),
            }
        )
    return checkpoints


def _select_recommended_checkpoint(checkpoints: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not checkpoints:
        return None
    for checkpoint in checkpoints:
        if checkpoint["checkpoint"] == "8000":
            return checkpoint
    numeric = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["checkpoint"].isdigit()
    ]
    if numeric:
        return max(numeric, key=lambda checkpoint: int(checkpoint["checkpoint"]))
    for checkpoint in checkpoints:
        if checkpoint["checkpoint"] == "best_task":
            return checkpoint
    return checkpoints[0]


def _classify_progress(run_dir: Path, checkpoint_count: int) -> Dict[str, object]:
    if checkpoint_count > 0:
        train_log = run_dir / "train.log"
        return {
            "progress_status": "checkpoint_present",
            "artifact_source": "trained_run",
            "train_log_bytes": train_log.stat().st_size if train_log.exists() else 0,
        }
    train_log = run_dir / "train.log"
    if not train_log.exists():
        return {
            "progress_status": "no_train_log",
            "artifact_source": "unknown_no_train_log",
            "train_log_bytes": 0,
        }
    text = train_log.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    has_iteration = any(
        token in text
        for token in (
            "Learning iteration",
            "Total timesteps",
            "Computation:",
            "Mean episode",
        )
    )
    if has_iteration:
        status = "training_progress_no_checkpoint"
        source = "training_run_no_checkpoint"
    elif lines and all(
        line.startswith("Loading model from:") or line.startswith("load_path:")
        for line in lines
    ):
        # Older evaluate.py calls initialized a runner with the default log_dir
        # before loading checkpoints, leaving train-style directories with only
        # these two load lines. They are eval artifacts, not failed training.
        status = "load_only_no_training_progress"
        source = "evaluate_checkpoint_load_log_dir"
    else:
        status = "train_log_without_checkpoint"
        source = "unknown_train_log_without_checkpoint"
    return {
        "progress_status": status,
        "artifact_source": source,
        "train_log_bytes": train_log.stat().st_size,
    }


def _default_output_prefix(eval_root: Path, checkpoint: str) -> str:
    return str(eval_root / f"selective_walk_followup_{checkpoint}")


def _audit_planned_outputs(output_prefix: str) -> List[Dict[str, object]]:
    planned = build_eval_plan(
        load_run="<audit-placeholder>",
        checkpoint="<audit-placeholder>",
        output_prefix=output_prefix,
    )
    rows = []
    for command in planned:
        output_dir = Path(command.output_dir)
        metrics_path = output_dir / "metrics.csv"
        rows.append(
            {
                "label": command.label,
                "output_dir": str(output_dir),
                "status": "present" if metrics_path.exists() else "missing",
                "metrics_csv": str(metrics_path),
            }
        )
    return rows


def audit_readiness(
    *,
    logs_root: str = DEFAULT_LOGS_ROOT,
    eval_root: str = DEFAULT_EVAL_ROOT,
    run_name_contains: str = DEFAULT_RUN_NAME_CONTAINS,
    output_prefix: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, object]:
    logs_root_path = Path(logs_root)
    eval_root_path = Path(eval_root)
    runs = _find_runs(logs_root_path, run_name_contains)
    run_rows = []
    checkpoint_rows = []
    for run in runs:
        checkpoints = _find_checkpoints(logs_root_path, run)
        progress = _classify_progress(run, len(checkpoints))
        run_rows.append(
            {
                "run_dir": str(run),
                "checkpoint_count": len(checkpoints),
                **progress,
            }
        )
        checkpoint_rows.extend(checkpoints)

    recommended_checkpoint = _select_recommended_checkpoint(checkpoint_rows)
    selected_checkpoint = (
        recommended_checkpoint["checkpoint"] if recommended_checkpoint is not None else "8000"
    )
    planned_prefix = output_prefix or _default_output_prefix(eval_root_path, selected_checkpoint)
    eval_rows = _audit_planned_outputs(planned_prefix)
    present_eval = sum(1 for row in eval_rows if row["status"] == "present")
    missing_eval = len(eval_rows) - present_eval
    recommended_eval_plan = []
    if recommended_checkpoint is not None:
        recommended_eval_plan = [
            command._asdict()
            for command in build_eval_plan(
                load_run=recommended_checkpoint["load_run"],
                checkpoint=recommended_checkpoint["checkpoint"],
                output_prefix=planned_prefix,
            )
        ]

    audit = {
        "logs_root": str(logs_root_path),
        "eval_root": str(eval_root_path),
        "run_name_contains": run_name_contains,
        "runs_found": len(run_rows),
        "runs": run_rows,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "recommended_checkpoint": (
            recommended_checkpoint["checkpoint"] if recommended_checkpoint is not None else None
        ),
        "recommended_load_run": (
            recommended_checkpoint["load_run"] if recommended_checkpoint is not None else None
        ),
        "selected_output_prefix": planned_prefix,
        "recommended_eval_plan": recommended_eval_plan,
        "planned_eval_outputs": len(eval_rows),
        "present_eval_outputs": present_eval,
        "missing_eval_outputs": missing_eval,
        "evaluation_complete": missing_eval == 0,
        "ready_for_evaluation": len(checkpoint_rows) > 0,
        "ready_for_completion": len(checkpoint_rows) > 0 and missing_eval == 0,
        "eval_outputs": eval_rows,
    }
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Audit selective-walk follow-up checkpoint/evaluation readiness."
    )
    parser.add_argument("--logs_root", default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--run_name_contains", default=DEFAULT_RUN_NAME_CONTAINS)
    parser.add_argument("--output_prefix")
    parser.add_argument("--output_json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = audit_readiness(
        logs_root=args.logs_root,
        eval_root=args.eval_root,
        run_name_contains=args.run_name_contains,
        output_prefix=args.output_prefix,
        output_json=args.output_json,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
