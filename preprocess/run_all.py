from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

HERE = Path(__file__).resolve().parent

STAGES = [
    ("create_dataset", "create_dataset.py"),
    ("split", "split.py"),
    ("coordinate_trans", "coordinate_trans.py"),
    ("str_to_list", "str_to_list.py"),
    ("minus_trans", "minus_trans.py"),
    ("pos_trans", "pos_trans.py"),
]


def run_stage(python_bin: str, stage_name: str, script_path: Path) -> None:
    print(f"\n[RUN] {stage_name} :: {python_bin} {script_path}")
    subprocess.run([python_bin, str(script_path)], check=True)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REMOTE preprocessing pipeline runner")
    parser.add_argument(
        "--skip",
        action="append",
        choices=[name for name, _ in STAGES],
        default=[],
        help="Stage name to skip (can be repeated)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for each stage (default: current interpreter)",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    skip = set(args.skip or [])

    for stage_name, relative_script in STAGES:
        if stage_name in skip:
            print(f"[SKIP] {stage_name}")
            continue
        script_path = HERE / relative_script
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        run_stage(args.python, stage_name, script_path)

    print("\nAll requested preprocessing stages finished.")


if __name__ == "__main__":
    main()

