"""Summarize evaluator termination and terminal-state diagnostic artifacts."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "preset",
    "task_return_mean",
    "fall_rate",
    "survival_time_mean_s",
    "contact_base_link_rate",
    "orientation_roll_pitch_rate",
    "timeout_rate",
    "contact_mean_survival_s",
    "orientation_mean_survival_s",
    "mean_base_z_terminal",
    "mean_abs_roll_terminal",
    "mean_abs_pitch_terminal",
    "max_contact_force_terminal",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required diagnostic artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"Diagnostic field {key!r} is missing or empty")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot summarize an empty terminal-state series")
    return sum(materialized) / len(materialized)


def _termination_key(row: Dict[str, str]) -> str:
    reason = row.get("termination_reason", "unknown") or "unknown"
    detail = row.get("termination_detail", "") or ""
    return f"{reason}:{detail}" if detail else reason


def summarize_failure_diagnostics(
    input_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Write compact failure summaries from one evaluator output directory."""
    source_dir = Path(input_dir)
    target_dir = Path(output_dir) if output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = _read_csv(source_dir / "metrics.csv")
    termination_rows = _read_csv(source_dir / "termination_reasons.csv")
    trace_rows = _read_csv(source_dir / "state_trace.csv")
    terminal_rows = [row for row in trace_rows if _float(row, "steps_until_done") == 0.0]

    metrics_by_preset = {row["preset_name"]: row for row in metrics_rows}
    presets = sorted(metrics_by_preset)
    if not presets:
        raise ValueError("metrics.csv contains no preset rows")
    if len(metrics_by_preset) != len(metrics_rows):
        raise ValueError("metrics.csv contains duplicate preset rows")

    expected_presets = set(presets)
    termination_presets = {row.get("preset_name", "") for row in termination_rows}
    trace_presets = {row.get("preset_name", "") for row in trace_rows}
    if termination_presets != expected_presets:
        raise ValueError("termination preset set does not match metrics.csv")
    if trace_presets != expected_presets:
        raise ValueError("state-trace preset set does not match metrics.csv")

    termination_by_preset: Dict[str, Dict[str, Dict[str, float]]] = {
        preset: {} for preset in presets
    }
    for row in termination_rows:
        preset = row.get("preset_name", "")
        if preset not in termination_by_preset:
            continue
        termination_by_preset[preset][_termination_key(row)] = {
            "count": int(_float(row, "count")),
            "rate": _float(row, "rate"),
            "mean_survival_time_s": _float(row, "mean_survival_time_s"),
        }

    terminal_rows_by_preset: Dict[str, List[Dict[str, str]]] = {}
    for preset in presets:
        expected_episodes = int(_float(metrics_by_preset[preset], "num_episodes"))
        preset_termination_rows = [
            row for row in termination_rows if row.get("preset_name") == preset
        ]
        termination_count = sum(
            int(_float(row, "count")) for row in preset_termination_rows
        )
        if termination_count != expected_episodes:
            raise ValueError(
                f"{preset} termination count {termination_count} does not match "
                f"num_episodes {expected_episodes}"
            )
        termination_rate = sum(
            _float(row, "rate") for row in preset_termination_rows
        )
        if not math.isclose(termination_rate, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{preset} termination rates do not sum to 1.0")

        preset_terminal_rows = [
            row for row in terminal_rows if row.get("preset_name") == preset
        ]
        terminal_episode_ids = {
            row.get("episode_index", "") for row in preset_terminal_rows
        }
        if (
            len(preset_terminal_rows) != expected_episodes
            or len(terminal_episode_ids) != expected_episodes
            or "" in terminal_episode_ids
        ):
            raise ValueError(
                f"{preset} terminal-state rows must contain exactly one row per episode"
            )
        terminal_rows_by_preset[preset] = preset_terminal_rows

    terminal_state_summary: Dict[str, Dict[str, float]] = {}
    summary_rows: List[Dict[str, object]] = []
    for preset in presets:
        preset_terminal_rows = terminal_rows_by_preset[preset]
        terminal_summary = {
            "terminal_trace_rows": len(preset_terminal_rows),
            "mean_base_z": _mean(_float(row, "base_z") for row in preset_terminal_rows),
            "mean_abs_roll": _mean(
                abs(_float(row, "roll")) for row in preset_terminal_rows
            ),
            "mean_abs_pitch": _mean(
                abs(_float(row, "pitch")) for row in preset_terminal_rows
            ),
            "max_abs_roll": max(
                abs(_float(row, "roll")) for row in preset_terminal_rows
            ),
            "max_abs_pitch": max(
                abs(_float(row, "pitch")) for row in preset_terminal_rows
            ),
            "mean_lin_vel_error": _mean(
                _float(row, "lin_vel_error") for row in preset_terminal_rows
            ),
            "mean_abs_yaw_vel_error": _mean(
                abs(_float(row, "yaw_vel_error")) for row in preset_terminal_rows
            ),
            "max_contact_force": max(
                _float(row, "contact_force_max") for row in preset_terminal_rows
            ),
        }
        terminal_state_summary[preset] = terminal_summary

        metrics = metrics_by_preset[preset]
        reasons = termination_by_preset[preset]
        contact = reasons.get("contact:base_link", {})
        orientation = reasons.get("orientation:roll_pitch", {})
        timeout = reasons.get("timeout", {})
        summary_rows.append(
            {
                "preset": preset,
                "task_return_mean": _float(metrics, "task_return_mean"),
                "fall_rate": _float(metrics, "fall_rate"),
                "survival_time_mean_s": _float(metrics, "survival_time_mean_s"),
                "contact_base_link_rate": float(contact.get("rate", 0.0)),
                "orientation_roll_pitch_rate": float(orientation.get("rate", 0.0)),
                "timeout_rate": float(timeout.get("rate", 0.0)),
                "contact_mean_survival_s": contact.get("mean_survival_time_s"),
                "orientation_mean_survival_s": orientation.get("mean_survival_time_s"),
                "mean_base_z_terminal": terminal_summary["mean_base_z"],
                "mean_abs_roll_terminal": terminal_summary["mean_abs_roll"],
                "mean_abs_pitch_terminal": terminal_summary["mean_abs_pitch"],
                "max_contact_force_terminal": terminal_summary["max_contact_force"],
            }
        )

    metrics_summary = {
        preset: {
            "task_return_mean": _float(row, "task_return_mean"),
            "fall_rate": _float(row, "fall_rate"),
            "survival_time_mean_s": _float(row, "survival_time_mean_s"),
            "base_height_violation_rate": _float(row, "base_height_violation_rate"),
            "roll_pitch_violation_rate": _float(row, "roll_pitch_violation_rate"),
        }
        for preset, row in metrics_by_preset.items()
    }
    summary = {
        "presets": presets,
        "rows": summary_rows,
        "termination_by_preset": termination_by_preset,
        "metrics_by_preset": metrics_summary,
        "terminal_state_summary": terminal_state_summary,
    }

    with (target_dir / "failure_diagnostics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (target_dir / "failure_diagnostics_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize one evaluate.py failure-diagnostic output directory."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_failure_diagnostics(args.input_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
