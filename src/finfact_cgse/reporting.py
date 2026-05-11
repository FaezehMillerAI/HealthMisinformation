from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PRIMARY_METRICS = [
    ("accuracy", "Accuracy"),
    ("micro_f1", "Micro-F1"),
    ("macro_f1", "Macro-F1"),
    ("rationale_f1", "Rationale-F1"),
    ("rouge1_f1", "ROUGE1-F1"),
    ("avg_sufficiency_confidence", "Suff."),
    ("avg_comprehensiveness_delta", "Comp."),
]

METRIC_LABELS = {key: label for key, label in PRIMARY_METRICS}


@dataclass
class RunResult:
    name: str
    output_dir: str
    metrics: Dict[str, float]
    summary: Dict


def load_run(run_dir: str | Path) -> RunResult:
    run_dir = Path(run_dir)
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return RunResult(
        name=summary.get("experiment_name", run_dir.name),
        output_dir=str(run_dir),
        metrics=metrics,
        summary=summary,
    )


def load_ablation_summary(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(runs: Sequence[RunResult]) -> str:
    lines = []
    lines.append("| Run | " + " | ".join(label for _, label in PRIMARY_METRICS) + " |")
    lines.append("| " + " | ".join(["---"] * (len(PRIMARY_METRICS) + 1)) + " |")
    for run in runs:
        lines.append(
            "| "
            + run.name
            + " | "
            + " | ".join(format_metric(run.metrics.get(metric, 0.0)) for metric, _ in PRIMARY_METRICS)
            + " |"
        )
    return "\n".join(lines)


def latex_table(runs: Sequence[RunResult]) -> str:
    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(PRIMARY_METRICS) + "}")
    lines.append("\\toprule")
    lines.append("Run & " + " & ".join(label for _, label in PRIMARY_METRICS) + " \\\\")
    lines.append("\\midrule")
    for run in runs:
        lines.append(
            run.name
            + " & "
            + " & ".join(format_metric(run.metrics.get(metric, 0.0)) for metric, _ in PRIMARY_METRICS)
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def best_run(runs: Sequence[RunResult], metric: str = "micro_f1") -> RunResult:
    return max(runs, key=lambda run: run.metrics.get(metric, float("-inf")))


def summarize_ablation_effects(ablation_summary: Dict, metric: str = "micro_f1") -> List[Dict]:
    runs = ablation_summary["runs"]
    grouped: Dict[str, Dict[str, Dict]] = {}
    for run in runs:
        family = run["family"]
        grouped.setdefault(family, {})[run["name"]] = run

    summaries = []
    for family, family_runs in grouped.items():
        anchor_name = f"{family}_full"
        anchor = family_runs.get(anchor_name)
        if not anchor:
            continue
        anchor_score = anchor["metrics"].get(metric, 0.0)
        for name, run in family_runs.items():
            if name == anchor_name:
                continue
            delta = run["metrics"].get(metric, 0.0) - anchor_score
            summaries.append(
                {
                    "family": family,
                    "variant": name,
                    "metric": metric,
                    "delta": delta,
                    "score": run["metrics"].get(metric, 0.0),
                }
            )
    summaries.sort(key=lambda item: (item["family"], item["delta"]))
    return summaries


def ablation_bullets(ablation_summary: Dict, metric: str = "micro_f1") -> List[str]:
    bullets = []
    items = summarize_ablation_effects(ablation_summary, metric=metric)
    items.sort(key=lambda item: (item["family"], -abs(item["delta"]), item["variant"]))
    for item in items:
        direction = "improves" if item["delta"] > 0 else "reduces" if item["delta"] < 0 else "matches"
        bullets.append(
            f"{item['variant']} {direction} {METRIC_LABELS.get(metric, metric)} by {abs(item['delta']):.3f} "
            f"relative to {item['family']}_full (score={item['score']:.3f})."
        )
    return bullets


def narrative_section(
    primary_runs: Sequence[RunResult],
    ablation_summary: Dict | None = None,
    metric: str = "micro_f1",
) -> str:
    if not primary_runs:
        return ""

    leader = best_run(primary_runs, metric=metric)
    metric_label = METRIC_LABELS.get(metric, metric)
    lines = []
    lines.append("## Results Narrative")
    lines.append("")
    lines.append(
        f"The strongest primary run under {metric_label} is `{leader.name}` with "
        f"{metric_label}={leader.metrics.get(metric, 0.0):.3f}, "
        f"ROUGE1-F1={leader.metrics.get('rouge1_f1', 0.0):.3f}, and "
        f"Rationale-F1={leader.metrics.get('rationale_f1', 0.0):.3f}."
    )
    lines.append("")

    if len(primary_runs) >= 2:
        ordered = sorted(primary_runs, key=lambda run: run.metrics.get(metric, 0.0), reverse=True)
        reference = ordered[0]
        challenger = ordered[1]
        delta = reference.metrics.get(metric, 0.0) - challenger.metrics.get(metric, 0.0)
        lines.append(
            f"Compared with `{challenger.name}`, `{reference.name}` changes {metric_label} by {delta:.3f}. "
            f"This comparison should be interpreted jointly with explanation quality, where "
            f"`{reference.name}` obtains ROUGE1-F1={reference.metrics.get('rouge1_f1', 0.0):.3f}."
        )
        lines.append("")

    if ablation_summary is not None:
        lines.append("Ablation findings:")
        for bullet in ablation_bullets(ablation_summary, metric=metric):
            lines.append(f"- {bullet}")
    return "\n".join(lines)


def report_payload(
    primary_runs: Sequence[RunResult],
    ablation_summary: Dict | None = None,
    metric: str = "micro_f1",
) -> Dict:
    return {
        "primary_runs": [
            {
                "name": run.name,
                "output_dir": run.output_dir,
                "metrics": run.metrics,
            }
            for run in primary_runs
        ],
        "best_run": {
            "name": best_run(primary_runs, metric=metric).name,
            "metric": metric,
            "score": best_run(primary_runs, metric=metric).metrics.get(metric, 0.0),
        } if primary_runs else None,
        "ablation_summary": ablation_summary,
    }
