from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.data.dataset import FinFactExample
from finfact_cgse.models.pipeline import CGSEPipeline
from finfact_cgse.utils.text import split_sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--digest", default="")
    args = parser.parse_args()

    example = FinFactExample(
        example_id="inference-example",
        claim=args.claim,
        digest=args.digest,
        context=args.context,
        evidence_sentences=[],
        context_sentences=split_sentences(args.context),
        label="unknown",
        metadata={},
    )
    pipeline = CGSEPipeline.load(args.model_path)
    prediction = pipeline.predict([example])[0]
    print(json.dumps(prediction.to_dict(), indent=2))


if __name__ == "__main__":
    main()
