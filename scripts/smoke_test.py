from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.config import load_config
from finfact_cgse.training.runner import run_smoke


def main() -> None:
    config = load_config("configs/smoke.yaml")
    result = run_smoke(config)
    print("SMOKE_TEST_OK")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
