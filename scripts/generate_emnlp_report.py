from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finfact_cgse.reporting import (
    latex_table,
    load_ablation_summary,
    load_run,
    markdown_table,
    narrative_section,
    report_payload,
)
from finfact_cgse.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--ablation-summary")
    parser.add_argument("--output-dir", default="outputs/report")
    parser.add_argument("--metric", default="micro_f1")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    runs = [load_run(path) for path in args.runs]
    ablation_summary = load_ablation_summary(args.ablation_summary) if args.ablation_summary else None

    markdown = "# EMNLP Results Section\n\n"
    markdown += "## Main Results\n\n"
    markdown += markdown_table(runs) + "\n\n"
    markdown += "## LaTeX Table\n\n```latex\n" + latex_table(runs) + "\n```\n\n"
    markdown += narrative_section(runs, ablation_summary=ablation_summary, metric=args.metric) + "\n"

    latex = latex_table(runs) + "\n"
    narrative = narrative_section(runs, ablation_summary=ablation_summary, metric=args.metric) + "\n"
    payload = report_payload(runs, ablation_summary=ablation_summary, metric=args.metric)

    (Path(output_dir) / "results_section.md").write_text(markdown, encoding="utf-8")
    (Path(output_dir) / "results_table.tex").write_text(latex, encoding="utf-8")
    (Path(output_dir) / "results_narrative.md").write_text(narrative, encoding="utf-8")
    (Path(output_dir) / "report_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("REPORT_GENERATED")
    print(str(Path(output_dir) / "results_section.md"))
    print(str(Path(output_dir) / "results_table.tex"))
    print(str(Path(output_dir) / "results_narrative.md"))
    print(str(Path(output_dir) / "report_payload.json"))


if __name__ == "__main__":
    main()
