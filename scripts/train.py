from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import load_config
from finfact_cgse.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = run_experiment(load_config(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
