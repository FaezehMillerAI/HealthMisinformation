from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import load_config
from finfact_cgse.data.dataset import load_train_val_examples, maybe_truncate
from finfact_cgse.models.neural_pipeline import NeuralCGSEPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    _, val_examples = load_train_val_examples(config)
    val_examples = maybe_truncate(val_examples, config["data"]["max_val_samples"])
    pipeline = NeuralCGSEPipeline.load(args.model_dir, config=config)
    metrics = pipeline.evaluate(val_examples)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
