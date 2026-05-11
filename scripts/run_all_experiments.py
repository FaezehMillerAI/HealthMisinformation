from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import load_config
from finfact_cgse.training.neural_runner import run_neural_experiment
from finfact_cgse.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    baseline_config = "configs/smoke.yaml" if args.mode == "smoke" else "configs/default.yaml"
    neural_config = "configs/neural_smoke.yaml" if args.mode == "smoke" else "configs/neural.yaml"

    baseline_result = run_experiment(load_config(baseline_config))
    neural_result = run_neural_experiment(load_config(neural_config))

    subprocess.run(
        [
            sys.executable,
            "scripts/make_results_table.py",
            "--runs",
            baseline_result["output_dir"],
            neural_result["output_dir"],
        ],
        check=True,
    )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "baseline": baseline_result,
                "neural": neural_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
