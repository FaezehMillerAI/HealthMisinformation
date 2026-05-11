from __future__ import annotations

from pathlib import Path
from typing import Dict

from finfact_cgse.data.dataset import load_train_val_examples, maybe_truncate
from finfact_cgse.models.pipeline import CGSEPipeline
from finfact_cgse.utils.io import ensure_dir, save_json, save_jsonl


def run_experiment(config: Dict) -> Dict:
    output_dir = ensure_dir(config["output_dir"])
    train_examples, val_examples = load_train_val_examples(config)
    train_examples = maybe_truncate(train_examples, config["data"]["max_train_samples"])
    val_examples = maybe_truncate(val_examples, config["data"]["max_val_samples"])

    pipeline = CGSEPipeline(config)
    pipeline.fit(train_examples)
    metrics = pipeline.evaluate(val_examples)
    predictions = [record.to_dict() for record in pipeline.predict(val_examples)]

    pipeline.save(str(output_dir / "model.joblib"))
    save_json(metrics, output_dir / "metrics.json")
    save_jsonl(predictions, output_dir / "predictions.jsonl")
    save_json(
        {
            "train_size": len(train_examples),
            "val_size": len(val_examples),
            "experiment_name": config["experiment_name"],
        },
        output_dir / "run_summary.json",
    )
    return {
        "metrics": metrics,
        "model_path": str(output_dir / "model.joblib"),
        "prediction_path": str(output_dir / "predictions.jsonl"),
        "output_dir": str(output_dir),
    }


def run_smoke(config: Dict) -> Dict:
    smoke_config = {
        **config,
        "output_dir": "outputs/smoke",
        "data": {
            **config["data"],
            "max_train_samples": config["smoke"]["train_samples"],
            "max_val_samples": config["smoke"]["val_samples"],
        },
    }
    return run_experiment(smoke_config)
