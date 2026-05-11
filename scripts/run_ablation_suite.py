from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.training.ablation_runner import run_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablations.yaml")
    parser.add_argument("--make-table", action="store_true")
    args = parser.parse_args()

    result = run_ablation_suite(args.config)
    if args.make_table:
        command = [sys.executable, "scripts/make_results_table.py", "--runs", *result["run_dirs"]]
        subprocess.run(command, check=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
