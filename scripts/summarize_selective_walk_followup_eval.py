"""Summarize the selective-walk follow-up evaluation output set.

This script complements ``plan_selective_walk_followup_eval.py``. It reads the
planned output directories after evaluation jobs finish and makes missing
directories explicit instead of silently treating a partial eval as complete.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


EXPECTED_LABELS = [
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

SUMMARY_FIELDS = [
    "label",
    "status",
    "output_dir",
    "rows",
    "task_return_mean",
    "fall_rate",
    "survival_time_mean_s",
    "worst_task_preset",
    "worst_task_return",
    "worst_fall_preset",
    "worst_fall_rate",
]


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _read_metrics(metrics_path: Path) -> List[Dict[str, object]]:
    with metrics_path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _summarize_present_label(label: str, output_dir: Path) -> Dict[str, object]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return {
            "label": label,
            "status": "missing_metrics",
            "output_dir": str(output_dir),
            "rows": 0,
            "task_return_mean": None,
            "fall_rate": None,
            "survival_time_mean_s": None,
            "worst_task_preset": "",
            "worst_task_return": None,
            "worst_fall_preset": "",
            "worst_fall_rate": None,
        }

    rows = _read_metrics(metrics_path)
    task_values = [_float_or_none(row.get("task_return_mean")) for row in rows]
    fall_values = [_float_or_none(row.get("fall_rate")) for row in rows]
    survival_values = [_float_or_none(row.get("survival_time_mean_s")) for row in rows]

    task_pairs = [
        (row.get("preset_name", ""), value)
        for row, value in zip(rows, task_values)
        if value is not None
    ]
    fall_pairs = [
        (row.get("preset_name", ""), value)
        for row, value in zip(rows, fall_values)
        if value is not None
    ]
    worst_task_preset, worst_task_return = min(
        task_pairs,
        key=lambda item: item[1],
        default=("", None),
    )
    worst_fall_preset, worst_fall_rate = max(
        fall_pairs,
        key=lambda item: item[1],
        default=("", None),
    )
    return {
        "label": label,
        "status": "present",
        "output_dir": str(output_dir),
        "rows": len(rows),
        "task_return_mean": _mean(task_values),
        "fall_rate": _mean(fall_values),
        "survival_time_mean_s": _mean(survival_values),
        "worst_task_preset": worst_task_preset,
        "worst_task_return": worst_task_return,
        "worst_fall_preset": worst_fall_preset,
        "worst_fall_rate": worst_fall_rate,
    }


def _summarize_label(label: str, output_prefix: Path) -> Dict[str, object]:
    output_dir = Path(f"{output_prefix}_{label}")
    if not output_dir.exists():
        return {
            "label": label,
            "status": "missing",
            "output_dir": str(output_dir),
            "rows": 0,
            "task_return_mean": None,
            "fall_rate": None,
            "survival_time_mean_s": None,
            "worst_task_preset": "",
            "worst_task_return": None,
            "worst_fall_preset": "",
            "worst_fall_rate": None,
        }
    return _summarize_present_label(label, output_dir)


def _write_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["rows"]
    with (output_dir / "selective_walk_followup_eval_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "selective_walk_followup_eval_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)


def summarize_eval_outputs(
    *,
    output_prefix: str,
    output_dir: Optional[str] = None,
    require_all: bool = True,
) -> Dict[str, object]:
    prefix = Path(output_prefix)
    rows = [_summarize_label(label, prefix) for label in EXPECTED_LABELS]
    present = sum(1 for row in rows if row["status"] == "present")
    missing = sum(1 for row in rows if row["status"] != "present")
    summary = {
        "output_prefix": output_prefix,
        "expected_outputs": len(EXPECTED_LABELS),
        "present_outputs": present,
        "missing_outputs": missing,
        "complete": missing == 0,
        "rows": rows,
    }
    if output_dir is not None:
        _write_summary(Path(output_dir), summary)
    if require_all and missing:
        missing_labels = [row["label"] for row in rows if row["status"] != "present"]
        raise FileNotFoundError(
            "missing selective-walk follow-up eval outputs: "
            + ", ".join(missing_labels)
        )
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize selective-walk follow-up evaluation outputs."
    )
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Write a partial summary instead of failing when outputs are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_eval_outputs(
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        require_all=not args.allow_missing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
