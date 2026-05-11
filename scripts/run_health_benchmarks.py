from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import load_config
from finfact_cgse.training.neural_runner import run_neural_experiment
from finfact_cgse.training.runner import run_experiment

HEALTH_CONFIGS = {
    "pubhealth_classic": "configs/health_pubhealth_classic.yaml",
    "pubhealth_neural": "configs/health_pubhealth_neural.yaml",
    "health_claims_classic": "configs/health_claims_classic.yaml",
    "health_claims_neural": "configs/health_claims_neural.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_dirs = []

    for key in ["pubhealth_classic", "health_claims_classic"]:
        config = load_config(HEALTH_CONFIGS[key])
        if args.smoke:
            config["data"]["max_train_samples"] = config["smoke"]["train_samples"]
            config["data"]["max_val_samples"] = config["smoke"]["val_samples"]
            config["experiment_name"] += "_smoke"
            config["output_dir"] += "_smoke"
        result = run_experiment(config)
        run_dirs.append(result["output_dir"])

    if not args.skip_neural:
        for key in ["pubhealth_neural", "health_claims_neural"]:
            config = load_config(HEALTH_CONFIGS[key])
            if args.smoke:
                config["data"]["max_train_samples"] = config["smoke"]["train_samples"]
                config["data"]["max_val_samples"] = config["smoke"]["val_samples"]
                config["experiment_name"] += "_smoke"
                config["output_dir"] += "_smoke"
            result = run_neural_experiment(config)
            run_dirs.append(result["output_dir"])

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_emnlp_report.py",
            "--runs",
            *run_dirs,
            "--output-dir",
            "outputs/health_report_smoke" if args.smoke else "outputs/health_report",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
