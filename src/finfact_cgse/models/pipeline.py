from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import joblib
import numpy as np

from finfact_cgse.data.counterfactuals import CounterfactualGenerator
from finfact_cgse.data.dataset import FinFactExample
from finfact_cgse.evaluation.metrics import classification_metrics, rationale_metrics, rouge1_f1
from finfact_cgse.models.classifier import RationaleAwareClassifier
from finfact_cgse.models.explainer import StructuredExplanationBuilder
from finfact_cgse.models.rationale import PseudoRationaleAligner


@dataclass
class PredictionRecord:
    example_id: str
    gold_label: str
    predicted_label: str
    confidence: float
    rationale_sentences: List[str]
    explanation: str
    sufficiency_confidence: float
    comprehensiveness_delta: float

    def to_dict(self) -> Dict:
        return asdict(self)


class CGSEPipeline:
    def __init__(self, config: Dict) -> None:
        self.config = config
        rationale_config = config["rationale"]
        classifier_config = config["classifier"]
        explanation_config = config["explanations"]
        counterfactual_config = config["counterfactuals"]

        self.aligner = PseudoRationaleAligner(**rationale_config)
        self.classifier = RationaleAwareClassifier(**classifier_config)
        self.explainer = StructuredExplanationBuilder(**explanation_config)
        self.counterfactual_generator = CounterfactualGenerator(
            max_per_example=counterfactual_config["max_per_example"]
        )
        self.feature_flags = config.get(
            "feature_flags",
            {"use_digest": True, "use_rationale": True, "use_context": True},
        )
        self.labels_: List[str] = []

    def fit(self, examples: Sequence[FinFactExample]) -> None:
        training_rows = []
        labels = []

        for example in examples:
            annotation = self.aligner.annotate_with_query(
                example,
                query_text=self._query_text(example),
                use_evidence=True,
            )
            training_rows.append(self._build_feature_row(example, annotation))
            labels.append(example.label)

            if self.config["counterfactuals"]["enabled"]:
                for counterfactual in self.counterfactual_generator.generate(example):
                    cf_example = FinFactExample(
                        example_id=f"{example.example_id}-cf",
                        claim=counterfactual.edited_claim,
                        digest=example.digest,
                        context=example.context,
                        evidence_sentences=example.evidence_sentences,
                        context_sentences=example.context_sentences,
                        label=counterfactual.target_label,
                        metadata={**example.metadata, "counterfactual_edit": counterfactual.edit_description},
                    )
                    cf_annotation = self.aligner.annotate_with_query(
                        cf_example,
                        query_text=self._query_text(cf_example),
                        use_evidence=False,
                    )
                    training_rows.append(self._build_feature_row(cf_example, cf_annotation))
                    labels.append(cf_example.label)

        self.labels_ = sorted(set(labels))
        self.classifier.fit(training_rows, labels)

    def evaluate(self, examples: Sequence[FinFactExample]) -> Dict:
        predictions = self.predict(examples)
        y_true = [example.label for example in examples]
        y_pred = [prediction.predicted_label for prediction in predictions]
        rationale_gold = [
            self.aligner.annotate_with_query(
                example,
                query_text=self._query_text(example),
                use_evidence=True,
            )["pseudo_gold_mask"]
            for example in examples
        ]
        rationale_pred = [
            self.aligner.annotate_with_query(
                example,
                query_text=self._query_text(example),
                use_evidence=False,
            )["predicted_mask"]
            for example in examples
        ]
        references = [" ".join(example.evidence_sentences) for example in examples]
        predicted_text = [prediction.explanation for prediction in predictions]

        metrics = classification_metrics(y_true, y_pred)
        metrics.update(rationale_metrics(rationale_gold, rationale_pred))
        metrics["rouge1_f1"] = rouge1_f1(references, predicted_text)
        metrics["avg_sufficiency_confidence"] = float(np.mean([record.sufficiency_confidence for record in predictions]))
        metrics["avg_comprehensiveness_delta"] = float(np.mean([record.comprehensiveness_delta for record in predictions]))
        return metrics

    def predict(self, examples: Sequence[FinFactExample]) -> List[PredictionRecord]:
        outputs = []
        for example in examples:
            annotation = self.aligner.annotate_with_query(
                example,
                query_text=self._query_text(example),
                use_evidence=False,
            )
            rationale_sentences = annotation["rationale_sentences"] if self.feature_flags.get("use_rationale", True) else []
            full_row = self._build_feature_row(example, annotation)
            probabilities = self.classifier.predict_proba([full_row])[0]
            label_order = self.classifier.model.classes_.tolist()
            prediction_index = int(np.argmax(probabilities))
            predicted_label = label_order[prediction_index]
            confidence = float(probabilities[prediction_index])

            sufficiency_row = self._build_feature_row(example, annotation, rationale_only=True)
            sufficiency_probabilities = self.classifier.predict_proba([sufficiency_row])[0]
            sufficiency_confidence = float(sufficiency_probabilities[prediction_index])

            without_rationale_row = self._build_feature_row(example, annotation, remove_rationale=True)
            without_rationale_probabilities = self.classifier.predict_proba([without_rationale_row])[0]
            comprehensiveness_delta = float(confidence - without_rationale_probabilities[prediction_index])

            counterfactual_note = None
            counterfactuals = self.counterfactual_generator.generate(example)
            if self.config["counterfactuals"]["enabled"] and counterfactuals:
                counterfactual_note = self.explainer.build_counterfactual_note(
                    counterfactuals[0].edit_description,
                    counterfactuals[0].target_label,
                )

            explanation = self.explainer.build(
                example=example,
                predicted_label=predicted_label,
                rationale_sentences=rationale_sentences,
                confidence=confidence,
                counterfactual_note=counterfactual_note,
            )

            outputs.append(
                PredictionRecord(
                    example_id=example.example_id,
                    gold_label=example.label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    rationale_sentences=rationale_sentences,
                    explanation=explanation,
                    sufficiency_confidence=sufficiency_confidence,
                    comprehensiveness_delta=comprehensiveness_delta,
                )
            )
        return outputs

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CGSEPipeline":
        return joblib.load(path)

    def _query_text(self, example: FinFactExample) -> str:
        if self.feature_flags.get("use_digest", True):
            return example.claim_with_digest
        return example.claim

    def _build_feature_row(
        self,
        example: FinFactExample,
        annotation: Dict,
        rationale_only: bool = False,
        remove_rationale: bool = False,
    ) -> Dict:
        rationale_sentences = annotation["rationale_sentences"]
        if not self.feature_flags.get("use_rationale", True):
            rationale_sentences = []
        context_sentences = example.context_sentences or [example.context]
        rationale_set = set(rationale_sentences)

        if rationale_only:
            context_text = " ".join(rationale_sentences)
        elif remove_rationale:
            context_text = " ".join(sentence for sentence in context_sentences if sentence not in rationale_set)
        else:
            context_text = example.context if self.feature_flags.get("use_context", True) else ""

        feature_text = " [SEP] ".join(
            part
            for part in [
                example.claim,
                example.digest if self.feature_flags.get("use_digest", True) else "",
                " ".join(rationale_sentences),
                context_text,
            ]
            if part
        )
        scores = annotation["scores"]
        return {
            "feature_text": feature_text,
            "rationale_count": len(rationale_sentences),
            "mean_rationale_score": float(np.mean(scores)) if scores else 0.0,
            "max_rationale_score": float(np.max(scores)) if scores else 0.0,
        }
