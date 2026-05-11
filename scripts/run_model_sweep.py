from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import deep_merge_dicts, load_config
from finfact_cgse.training.neural_runner import run_neural_experiment
from finfact_cgse.utils.io import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sweep_config = load_config(args.config)
    base_config = load_config(sweep_config["base_config"])
    output_root = ensure_dir(sweep_config["output_root"] + ("_smoke" if args.smoke else ""))
    run_dirs = []
    runs = []

    for model_spec in sweep_config["models"]:
        config = deep_merge_dicts(base_config, model_spec.get("overrides", {}))
        config["neural"]["model_name"] = model_spec["model_name"]
        config["experiment_name"] = f"{base_config['experiment_name']}_{model_spec['name']}" + ("_smoke" if args.smoke else "")
        config["output_dir"] = str(output_root / model_spec["name"])
        if args.smoke:
            config["data"]["max_train_samples"] = config["smoke"]["train_samples"]
            config["data"]["max_val_samples"] = config["smoke"]["val_samples"]

        result = run_neural_experiment(config)
        run_dirs.append(result["output_dir"])
        runs.append(
            {
                "name": model_spec["name"],
                "model_name": model_spec["model_name"],
                "output_dir": result["output_dir"],
                "metrics": result["metrics"],
            }
        )

    report_output_dir = sweep_config["report_output_dir"] + ("_smoke" if args.smoke else "")
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_emnlp_report.py",
            "--runs",
            *run_dirs,
            "--output-dir",
            report_output_dir,
            "--metric",
            sweep_config.get("metric", "micro_f1"),
        ],
        check=True,
    )

    summary = {
        "sweep_config": args.config,
        "report_output_dir": report_output_dir,
        "runs": runs,
    }
    save_json(summary, Path(report_output_dir) / "model_sweep_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
