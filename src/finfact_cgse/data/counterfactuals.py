from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from finfact_cgse.data.dataset import FinFactExample


POLARITY_PATTERNS = [
    (re.compile(r"\bincreased\b", re.IGNORECASE), "decreased"),
    (re.compile(r"\bgrew\b", re.IGNORECASE), "fell"),
    (re.compile(r"\brise\b", re.IGNORECASE), "drop"),
    (re.compile(r"\bprofit\b", re.IGNORECASE), "loss"),
]

NUMBER_RE = re.compile(r"(?<!\w)(\d[\d,]*(?:\.\d+)?)(%?)(?!\w)")
YEAR_RE = re.compile(r"\b(19|20)(\d{2})\b")


@dataclass
class CounterfactualExample:
    source_id: str
    edited_claim: str
    target_label: str
    edit_description: str


class CounterfactualGenerator:
    def __init__(self, max_per_example: int = 1) -> None:
        self.max_per_example = max_per_example

    def generate(self, example: FinFactExample) -> List[CounterfactualExample]:
        outputs: List[CounterfactualExample] = []
        if example.label == "true":
            outputs.extend(self._flip_true_to_false(example))
        elif example.label == "neutral":
            outputs.extend(self._force_resolution(example))
        return outputs[: self.max_per_example]

    def _flip_true_to_false(self, example: FinFactExample) -> List[CounterfactualExample]:
        claim = example.claim
        number_match = NUMBER_RE.search(claim)
        if number_match:
            original = number_match.group(1)
            edited_value = self._perturb_number(original)
            edited_claim = claim.replace(original, edited_value, 1)
            return [
                CounterfactualExample(
                    source_id=example.example_id,
                    edited_claim=edited_claim,
                    target_label="false",
                    edit_description=f"Changed numeric value from {original} to {edited_value}.",
                )
            ]
        year_match = YEAR_RE.search(claim)
        if year_match:
            original = year_match.group(0)
            edited_value = str(int(original) + 1)
            edited_claim = claim.replace(original, edited_value, 1)
            return [
                CounterfactualExample(
                    source_id=example.example_id,
                    edited_claim=edited_claim,
                    target_label="false",
                    edit_description=f"Shifted the referenced year from {original} to {edited_value}.",
                )
            ]
        for pattern, replacement in POLARITY_PATTERNS:
            if pattern.search(claim):
                edited_claim = pattern.sub(replacement, claim, count=1)
                return [
                    CounterfactualExample(
                        source_id=example.example_id,
                        edited_claim=edited_claim,
                        target_label="false",
                        edit_description=f"Swapped polarity-bearing term to '{replacement}'.",
                    )
                ]
        return []

    def _force_resolution(self, example: FinFactExample) -> List[CounterfactualExample]:
        context_hint = example.context_sentences[0] if example.context_sentences else example.context[:200]
        edited_claim = f"{example.claim} Specifically, {context_hint}"
        return [
            CounterfactualExample(
                source_id=example.example_id,
                edited_claim=edited_claim,
                target_label="true",
                edit_description="Injected a supporting context fragment to reduce ambiguity.",
            )
        ]

    @staticmethod
    def _perturb_number(raw_value: str) -> str:
        numeric = float(raw_value.replace(",", ""))
        if numeric == 0:
            numeric = 1.0
        edited = numeric * 1.15
        if raw_value.isdigit():
            return str(int(round(edited)))
        return f"{edited:.2f}".rstrip("0").rstrip(".")
