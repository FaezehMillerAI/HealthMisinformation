from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from finfact_cgse.utils.text import (
    compact_text,
    normalize_label,
    parse_evidence_sentences,
    safe_literal,
    split_sentences,
    stringify_digest,
)


@dataclass
class FinFactExample:
    example_id: str
    claim: str
    digest: str
    context: str
    evidence_sentences: List[str]
    context_sentences: List[str]
    label: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def claim_with_digest(self) -> str:
        return compact_text([self.claim, self.digest])


def row_to_example(row: Dict, example_id: str) -> FinFactExample:
    digest = stringify_digest(row.get("sci_digest"))
    context = compact_text([str(row.get("justification") or "")])
    context_sentences = split_sentences(context)
    metadata = {
        "url": row.get("url"),
        "author": row.get("author"),
        "posted": row.get("posted"),
        "issues": safe_literal(row.get("issues")),
        "visualization_bias": row.get("visualization_bias"),
    }
    return FinFactExample(
        example_id=example_id,
        claim=str(row.get("claim") or "").strip(),
        digest=digest,
        context=context,
        evidence_sentences=parse_evidence_sentences(row.get("evidence")),
        context_sentences=context_sentences,
        label=normalize_label(str(row.get("label") or "")),
        metadata=metadata,
    )


def row_to_pubhealth_example(row: Dict, example_id: str) -> FinFactExample | None:
    valid_labels = {"false", "true", "mixture", "unproven"}
    label = normalize_label(str(row.get("label") or ""))
    if label not in valid_labels:
        return None
    context = compact_text([str(row.get("main_text") or "")])
    return FinFactExample(
        example_id=example_id,
        claim=str(row.get("claim") or "").strip(),
        digest=compact_text([str(row.get("subjects") or "")]),
        context=context,
        evidence_sentences=split_sentences(str(row.get("explanation") or "")),
        context_sentences=split_sentences(context),
        label=label,
        metadata={"claim_id": row.get("claim_id")},
    )


def row_to_health_misinformation_example(row: Dict, example_id: str) -> FinFactExample:
    raw_label = str(row.get("True or Misinformation") or "").strip().lower()
    label = {"true": "true", "misinformation": "false"}.get(raw_label, raw_label)
    evidence = str(row.get("Reason") or "").strip()
    context = evidence
    return FinFactExample(
        example_id=example_id,
        claim=str(row.get("Statement") or "").strip(),
        digest="health misinformation benchmark",
        context=context,
        evidence_sentences=split_sentences(evidence),
        context_sentences=split_sentences(context),
        label=label,
        metadata={},
    )


def load_examples(config: Dict, split: str = "train") -> List[FinFactExample]:
    dataset_name = config["data"]["dataset_name"]
    adapter = config["data"].get("adapter", "finfact")
    raw_dataset = load_dataset(dataset_name)

    if adapter == "finfact":
        rows = raw_dataset[split]
        return [row_to_example(row, f"finfact-{idx}") for idx, row in enumerate(rows)]

    if adapter == "pubhealth":
        rows = raw_dataset[split]
        examples = [row_to_pubhealth_example(row, f"pubhealth-{split}-{idx}") for idx, row in enumerate(rows)]
        return [example for example in examples if example is not None]

    if adapter == "health_misinformation":
        rows = raw_dataset[split]
        return [row_to_health_misinformation_example(row, f"healthmis-{split}-{idx}") for idx, row in enumerate(rows)]

    raise ValueError(f"Unsupported data adapter: {adapter}")


def load_train_val_examples(config: Dict) -> Tuple[List[FinFactExample], List[FinFactExample]]:
    data_config = config["data"]
    if data_config.get("use_dataset_splits", False):
        train_split = data_config.get("train_split_name", "train")
        val_split = data_config.get("validation_split_name", "validation")
        train_examples = load_examples(config, split=train_split)
        val_examples = load_examples(config, split=val_split)
        return train_examples, val_examples

    examples = load_examples(config, split=data_config.get("train_split_name", "train"))
    return split_examples(
        examples,
        test_size=data_config["test_size"],
        seed=config["seed"],
    )


def split_examples(
    examples: Sequence[FinFactExample],
    test_size: float,
    seed: int,
) -> Tuple[List[FinFactExample], List[FinFactExample]]:
    labels = [example.label for example in examples]
    train_examples, val_examples = train_test_split(
        list(examples),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_examples, val_examples


def maybe_truncate(examples: Iterable[FinFactExample], max_samples: int | None) -> List[FinFactExample]:
    examples = list(examples)
    if max_samples is None:
        return examples
    return examples[:max_samples]
