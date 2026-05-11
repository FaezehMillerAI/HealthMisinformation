from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


METRIC_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("micro_f1", "Micro-F1"),
    ("macro_f1", "Macro-F1"),
    ("rationale_f1", "Rationale-F1"),
    ("rouge1_f1", "ROUGE1-F1"),
    ("avg_sufficiency_confidence", "Suff."),
    ("avg_comprehensiveness_delta", "Comp."),
]


def format_value(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    for run_path in args.runs:
        run_dir = Path(run_path)
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "run": summary.get("experiment_name", run_dir.name),
                **{metric: metrics.get(metric, 0.0) for metric, _ in METRIC_COLUMNS},
            }
        )

    markdown_lines = []
    markdown_lines.append("| Run | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * (len(METRIC_COLUMNS) + 1)) + " |")
    for row in rows:
        markdown_lines.append(
            "| "
            + row["run"]
            + " | "
            + " | ".join(format_value(row[metric]) for metric, _ in METRIC_COLUMNS)
            + " |"
        )

    latex_lines = []
    latex_lines.append("\\begin{tabular}{l" + "c" * len(METRIC_COLUMNS) + "}")
    latex_lines.append("\\toprule")
    latex_lines.append("Run & " + " & ".join(label for _, label in METRIC_COLUMNS) + " \\\\")
    latex_lines.append("\\midrule")
    for row in rows:
        latex_lines.append(
            row["run"]
            + " & "
            + " & ".join(format_value(row[metric]) for metric, _ in METRIC_COLUMNS)
            + " \\\\"
        )
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    print("MARKDOWN_TABLE")
    print("\n".join(markdown_lines))
    print("\nLATEX_TABLE")
    print("\n".join(latex_lines))


if __name__ == "__main__":
    main()
