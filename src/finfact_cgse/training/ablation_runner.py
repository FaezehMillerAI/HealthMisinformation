from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from finfact_cgse.config import load_config
from finfact_cgse.training.neural_runner import run_neural_experiment
from finfact_cgse.training.runner import run_experiment
from finfact_cgse.utils.io import ensure_dir, save_json


def deep_merge(base: Dict, updates: Dict) -> Dict:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_ablation_suite(ablation_config_path: str | Path) -> Dict:
    ablation_config = load_config(ablation_config_path)
    output_root = ensure_dir(ablation_config["output_root"])
    baseline_base = load_config(ablation_config["baseline_config"])
    neural_base = load_config(ablation_config["neural_config"])

    results: List[Dict] = []
    run_dirs: List[str] = []

    for variant in ablation_config["variants"]:
        family = variant["family"]
        base_config = baseline_base if family == "baseline" else neural_base
        config = deep_merge(base_config, variant.get("overrides", {}))
        config["experiment_name"] = variant["name"]
        config["output_dir"] = str(output_root / variant["name"])

        if family == "baseline":
            result = run_experiment(config)
        else:
            result = run_neural_experiment(config)

        results.append(
            {
                "name": variant["name"],
                "family": family,
                "output_dir": result["output_dir"],
                "metrics": result["metrics"],
            }
        )
        run_dirs.append(result["output_dir"])

    summary = {
        "ablation_config": str(ablation_config_path),
        "runs": results,
        "run_dirs": run_dirs,
    }
    save_json(summary, output_root / "ablation_summary.json")
    return summary
