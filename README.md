# FinFact-CGSE

`FinFact-CGSE` is a modular research codebase that implements a practical version of the proposed Counterfactually-Grounded Structured Explanations (CGSE) methodology for misinformation detection experiments and EMNLP-style reporting.

The pipeline is designed to be:

- reproducible for research iteration,
- lightweight enough for Google Colab,
- modular enough to extend into stronger neural submissions later,
- faithful to the proposal structure you drafted from the FinNLP paper,
- flexible enough to run both financial and health misinformation benchmarks.

## What is implemented

The codebase provides:

- dataset normalization for `Fin-Fact`,
- health-domain experiment configs for `OpenMed/PubHealth-Processed` and `ClassyB/Health_Misinformation`,
- multi-model sweep configs for comparing several neural backbones on the same dataset,
- weak rationale supervision by aligning evidence text to context sentences,
- rationale-aware label classification,
- a stronger transformer-based classifier for paper-facing experiments,
- structured explanation generation,
- sufficiency and comprehensiveness faithfulness checks,
- heuristic counterfactual augmentation,
- train, evaluate, predict, smoke-test, ablation-suite, results-table, and report-generation entrypoints.

## Repository layout

```text
configs/                 YAML experiment settings
scripts/                 CLI entrypoints
src/finfact_cgse/        Core library code
outputs/                 Saved runs, metrics, and models
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Quick start

Train a compact classical experiment:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a saved classical model:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --model-path outputs/default/model.joblib
```

Train the stronger neural CGSE variant:

```bash
python scripts/train_neural.py --config configs/neural.yaml
```

Evaluate the neural variant:

```bash
python scripts/evaluate_neural.py \
  --config configs/neural.yaml \
  --model-dir outputs/neural
```

Generate a raw prediction from text:

```bash
python scripts/predict.py \
  --model-path outputs/default/model.joblib \
  --claim "Company X reported a 12% rise in quarterly revenue." \
  --context "The earnings release states revenue increased 12% year over year in Q2."
```

Run the smoke test:

```bash
python scripts/smoke_test.py
```

Build a Markdown and LaTeX results table from finished runs:

```bash
python scripts/make_results_table.py --runs outputs/smoke outputs/neural_smoke
```

Run both baseline and neural experiments together:

```bash
python scripts/run_all_experiments.py --mode smoke
```

Run the ablation suite and emit a comparison table:

```bash
python scripts/run_ablation_suite.py --config configs/ablations.yaml --make-table
```

Generate a paper-ready EMNLP results section from completed runs:

```bash
python scripts/generate_emnlp_report.py \
  --runs outputs/smoke outputs/neural_smoke \
  --ablation-summary outputs/ablations/ablation_summary.json \
  --output-dir outputs/report
```

Run the two health-domain benchmarks end to end:

```bash
python scripts/run_health_benchmarks.py
```

Run a multi-model neural sweep on one dataset:

```bash
python scripts/run_model_sweep.py --config configs/model_sweep_health_claims.yaml
```

## Google Colab

In Colab, the following is usually enough:

```python
!git clone <your-github-repo-url>
%cd finfact-cgse
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
!python scripts/smoke_test.py
!python scripts/train.py --config configs/default.yaml
!python scripts/train_neural.py --config configs/neural.yaml
!python scripts/run_ablation_suite.py --config configs/ablations.yaml --make-table
!python scripts/generate_emnlp_report.py --runs outputs/smoke outputs/neural_smoke --ablation-summary outputs/ablations/ablation_summary.json --output-dir outputs/report
!python scripts/run_health_benchmarks.py
!python scripts/run_model_sweep.py --config configs/model_sweep_health_claims.yaml
```

## Methodology mapping

This implementation operationalizes the proposal as follows:

1. `PseudoRationaleAligner` converts evidence text into weak rationale supervision by matching evidence sentences against context sentences.
2. `CounterfactualGenerator` creates controlled claim edits, primarily around numbers, years, and polarity.
3. `CGSEPipeline` builds rationale-aware features from the claim, digest, selected rationale sentences, and context, then trains a classifier.
4. `NeuralCGSEPipeline` upgrades the label predictor to a contextual Hugging Face encoder while keeping the rationale serialization explicit.
5. `StructuredExplanationBuilder` emits auditable explanations and optional counterfactual narratives.
6. `generate_emnlp_report.py` turns run artifacts into a draft-ready Markdown section, LaTeX table, narrative summary, and JSON payload.

## Notes

- The released `Fin-Fact` dataset currently exposes a single `train` split, so the financial scripts create reproducible train/validation splits internally.
- `OpenMed/PubHealth-Processed` ships explicit `train` and `validation` splits and is supported directly.
- `ClassyB/Health_Misinformation` is supported as a second health benchmark with internal train/validation splitting.
- The default stack uses `sentence-transformers/all-MiniLM-L6-v2` for fast Colab-friendly sentence embeddings.
- `configs/neural.yaml` is the paper-facing neural default, while `configs/neural_smoke.yaml` is the faster verification setup.
- `configs/ablations.yaml` defines a reproducible smoke-scale ablation suite over digest, rationale, and counterfactual components.
- `configs/model_sweep_health_claims.yaml` and `configs/model_sweep_pubhealth.yaml` compare multiple neural backbones on the same benchmark.
- For health misinformation, this repo currently supports `OpenMed/PubHealth-Processed` and `ClassyB/Health_Misinformation` as the two directly usable datasets in this environment. The discovered `MedHelp` dataset here is not a misinformation benchmark, and the older SciFact script loaders exposed through `datasets` are currently deprecated.
