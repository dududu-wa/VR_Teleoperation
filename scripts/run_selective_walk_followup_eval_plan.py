"""Dry-run or execute the formal selective-walk follow-up evaluation plan.

This script consumes the readiness audit output. By default it only prints the
recommended commands; it runs them only when --execute is passed explicitly.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional


DEFAULT_AUDIT_JSON = (
    "outputs/eval/June30_selective_walk_followup_readiness_audit/readiness_audit.json"
)


def load_audit(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_execution(audit: Dict[str, object]) -> Dict[str, object]:
    commands = list(audit.get("recommended_eval_plan") or [])
    if not commands:
        return {
            "can_execute": False,
            "reason": "No recommended eval commands; run readiness audit after a real checkpoint exists.",
            "commands": [],
        }
    return {
        "can_execute": True,
        "reason": "recommended eval commands are available",
        "commands": commands,
    }


def format_dry_run_lines(commands: List[Dict[str, str]]) -> List[str]:
    lines: List[str] = []
    for command in commands:
        lines.append(f"# {command['label']}")
        lines.append(command["command"])
        lines.append("")
    return lines


def _subprocess_runner(command: str) -> None:
    subprocess.run(command, shell=True, check=True)


def run_eval_plan(
    audit: Dict[str, object],
    *,
    execute: bool = False,
    command_runner: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    plan = prepare_execution(audit)
    commands = plan["commands"]
    lines = format_dry_run_lines(commands)
    if not plan["can_execute"]:
        if execute:
            raise ValueError(plan["reason"])
        return {
            "planned": 0,
            "executed": 0,
            "lines": [plan["reason"]],
        }
    if not execute:
        return {
            "planned": len(commands),
            "executed": 0,
            "lines": lines,
        }

    runner = command_runner or _subprocess_runner
    for command in commands:
        runner(command["command"])
    return {
        "planned": len(commands),
        "executed": len(commands),
        "lines": lines,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run or execute the selective-walk follow-up eval plan."
    )
    parser.add_argument("--audit_json", default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = load_audit(args.audit_json)
    try:
        result = run_eval_plan(audit, execute=args.execute)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for line in result["lines"]:
        print(line)


if __name__ == "__main__":
    main()
