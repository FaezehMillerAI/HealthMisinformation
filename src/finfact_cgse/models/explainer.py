from __future__ import annotations

from typing import Dict, List

from finfact_cgse.data.dataset import FinFactExample
from finfact_cgse.utils.text import extract_entities, extract_key_numbers


LABEL_VERB = {
    "true": "supports",
    "false": "contradicts",
    "neutral": "does not fully resolve",
    "unproven": "does not fully establish",
    "mixture": "partially supports",
}


class StructuredExplanationBuilder:
    def __init__(self, max_rationale_sentences: int = 3) -> None:
        self.max_rationale_sentences = max_rationale_sentences

    def build(
        self,
        example: FinFactExample,
        predicted_label: str,
        rationale_sentences: List[str],
        confidence: float,
        counterfactual_note: str | None = None,
    ) -> str:
        entities = extract_entities(example.claim)[:3]
        numbers = extract_key_numbers(example.claim)[:3]
        focus_parts = []
        if entities:
            focus_parts.append(f"entities: {', '.join(entities)}")
        if numbers:
            focus_parts.append(f"numbers: {', '.join(numbers)}")
        focus_text = "; ".join(focus_parts) if focus_parts else "core financial assertion"

        selected = rationale_sentences[: self.max_rationale_sentences]
        rationale_text = " ".join(selected) if selected else "No high-confidence rationale sentence was isolated from the context."
        verdict = LABEL_VERB.get(predicted_label, "relates to")

        explanation = (
            f"The claim centers on {focus_text}. "
            f"The most relevant context states: {rationale_text} "
            f"This evidence {verdict} the claim, so the predicted label is {predicted_label.upper()} "
            f"with confidence {confidence:.3f}."
        )
        if counterfactual_note:
            explanation += f" Counterfactual check: {counterfactual_note}"
        return explanation

    def build_counterfactual_note(self, edit_description: str, target_label: str) -> str:
        return f"If the edited detail followed this change, the label would likely move toward {target_label.upper()}: {edit_description}"
