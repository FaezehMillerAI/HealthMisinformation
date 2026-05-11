from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from finfact_cgse.data.counterfactuals import CounterfactualGenerator
from finfact_cgse.data.dataset import FinFactExample
from finfact_cgse.evaluation.metrics import classification_metrics, rationale_metrics, rouge1_f1
from finfact_cgse.models.explainer import StructuredExplanationBuilder
from finfact_cgse.models.rationale import PseudoRationaleAligner
from finfact_cgse.models.pipeline import PredictionRecord
from finfact_cgse.utils.io import ensure_dir, save_json
from finfact_cgse.utils.seed import set_seed


DEFAULT_LABELS = ["false", "neutral", "true"]


@dataclass
class NeuralTrainingExample:
    example_id: str
    input_text: str
    claim: str
    digest: str
    context: str
    label: str
    rationale_sentences: List[str]
    rationale_scores: List[float]
    pseudo_gold_mask: List[int]
    predicted_mask: List[int]
    evidence_text: str
    counterfactual_note: str | None

    def to_dict(self) -> Dict:
        return asdict(self)


class _EncodedTextDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class NeuralCGSEPipeline:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self._initialize_runtime(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.neural_config["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.neural_config["model_name"],
            num_labels=len(self.labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )
        self.model.to(self.device)

    def fit(self, train_examples: Sequence[FinFactExample]) -> Dict:
        set_seed(self.config["seed"])
        records = self._prepare_training_records(train_examples, include_counterfactuals=True)
        train_dataset = self._build_dataset(records, include_labels=True)
        dataloader = DataLoader(train_dataset, batch_size=self.neural_config["batch_size"], shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.neural_config["learning_rate"]),
            weight_decay=float(self.neural_config["weight_decay"]),
        )
        total_steps = max(1, len(dataloader) * int(self.neural_config["epochs"]))
        warmup_steps = int(total_steps * float(self.neural_config["warmup_ratio"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()
        running_losses = []
        total_epochs = int(self.neural_config["epochs"])
        for epoch_index in range(total_epochs):
            epoch_bar = tqdm(
                dataloader,
                desc=f"training epoch {epoch_index + 1}/{total_epochs}",
                leave=False,
            )
            for batch in epoch_bar:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.neural_config["gradient_clip_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_value = float(loss.detach().cpu())
                running_losses.append(loss_value)
                epoch_bar.set_postfix(loss=f"{loss_value:.4f}")
        return {"train_loss": float(np.mean(running_losses)) if running_losses else 0.0}

    def evaluate(self, examples: Sequence[FinFactExample]) -> Dict:
        records = self._prepare_training_records(examples, include_counterfactuals=False)
        predictions = self.predict_from_records(records)
        y_true = [record.label for record in records]
        y_pred = [prediction.predicted_label for prediction in predictions]
        metrics = classification_metrics(y_true, y_pred)
        metrics.update(rationale_metrics(
            [record.pseudo_gold_mask for record in records],
            [record.predicted_mask for record in records],
        ))
        metrics["rouge1_f1"] = rouge1_f1(
            [record.evidence_text for record in records],
            [prediction.explanation for prediction in predictions],
        )
        metrics["avg_sufficiency_confidence"] = float(np.mean([prediction.sufficiency_confidence for prediction in predictions]))
        metrics["avg_comprehensiveness_delta"] = float(np.mean([prediction.comprehensiveness_delta for prediction in predictions]))
        return metrics

    def predict(self, examples: Sequence[FinFactExample]) -> List[PredictionRecord]:
        records = self._prepare_training_records(examples, include_counterfactuals=False)
        return self.predict_from_records(records)

    def predict_from_records(self, records: Sequence[NeuralTrainingExample]) -> List[PredictionRecord]:
        full_probs = self._predict_probabilities([record.input_text for record in records])
        sufficiency_probs = self._predict_probabilities([self._serialize_rationale_only(record) for record in records])
        comprehensiveness_probs = self._predict_probabilities([self._serialize_without_rationale(record) for record in records])

        outputs: List[PredictionRecord] = []
        for idx, record in enumerate(records):
            probs = full_probs[idx]
            predicted_index = int(np.argmax(probs))
            predicted_label = self.id_to_label[predicted_index]
            confidence = float(probs[predicted_index])
            sufficiency_confidence = float(sufficiency_probs[idx][predicted_index])
            comprehensiveness_delta = float(confidence - comprehensiveness_probs[idx][predicted_index])
            explanation = self.explainer.build(
                example=FinFactExample(
                    example_id=record.example_id,
                    claim=record.claim,
                    digest=record.digest,
                    context=record.context,
                    evidence_sentences=[],
                    context_sentences=[],
                    label=record.label,
                    metadata={},
                ),
                predicted_label=predicted_label,
                rationale_sentences=record.rationale_sentences,
                confidence=confidence,
                counterfactual_note=record.counterfactual_note,
            )
            outputs.append(
                PredictionRecord(
                    example_id=record.example_id,
                    gold_label=record.label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    rationale_sentences=record.rationale_sentences,
                    explanation=explanation,
                    sufficiency_confidence=sufficiency_confidence,
                    comprehensiveness_delta=comprehensiveness_delta,
                )
            )
        return outputs

    def save(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        save_json(
            {
                "config": self.config,
                "labels": self.labels,
            },
            Path(output_dir) / "cgse_neural_metadata.json",
        )

    @classmethod
    def load(cls, model_dir: str | Path, config: Dict | None = None) -> "NeuralCGSEPipeline":
        metadata_path = Path(model_dir) / "cgse_neural_metadata.json"
        if config is None:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            config = metadata["config"]
        pipeline = cls.__new__(cls)
        pipeline._initialize_runtime(config)
        pipeline.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        pipeline.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        pipeline.model.to(pipeline.device)
        return pipeline

    def _prepare_training_records(
        self,
        examples: Sequence[FinFactExample],
        include_counterfactuals: bool,
    ) -> List[NeuralTrainingExample]:
        records: List[NeuralTrainingExample] = []
        for example in examples:
            annotation = self.aligner.annotate_with_query(
                example,
                query_text=self._query_text(example),
                use_evidence=True,
            )
            records.append(self._record_from_example(example, annotation, counterfactual_note=True))
            if include_counterfactuals and self.config["counterfactuals"]["enabled"]:
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
                    records.append(self._record_from_example(cf_example, cf_annotation, counterfactual_note=False))
        return records

    def _record_from_example(
        self,
        example: FinFactExample,
        annotation: Dict,
        counterfactual_note: bool,
    ) -> NeuralTrainingExample:
        note = None
        if counterfactual_note and self.config["counterfactuals"]["enabled"]:
            counterfactuals = self.counterfactual_generator.generate(example)
            if counterfactuals:
                note = self.explainer.build_counterfactual_note(
                    counterfactuals[0].edit_description,
                    counterfactuals[0].target_label,
                )
        rationale_sentences = annotation["rationale_sentences"]
        if not self.feature_flags.get("use_rationale", True):
            rationale_sentences = []
        rationale_scores = [annotation["scores"][idx] for idx in annotation["top_indices"][: len(rationale_sentences)]]
        input_text = self._serialize_input(example, rationale_sentences)
        return NeuralTrainingExample(
            example_id=example.example_id,
            input_text=input_text,
            claim=example.claim,
            digest=example.digest,
            context=example.context,
            label=example.label,
            rationale_sentences=rationale_sentences,
            rationale_scores=rationale_scores,
            pseudo_gold_mask=annotation["pseudo_gold_mask"],
            predicted_mask=annotation["predicted_mask"],
            evidence_text=" ".join(example.evidence_sentences),
            counterfactual_note=note,
        )

    def _build_dataset(self, records: Sequence[NeuralTrainingExample], include_labels: bool) -> _EncodedTextDataset:
        encodings = self.tokenizer(
            [record.input_text for record in records],
            padding=True,
            truncation=True,
            max_length=int(self.neural_config["max_length"]),
            return_tensors="pt",
        )
        labels = None
        if include_labels:
            labels = torch.tensor([self.label_to_id[record.label] for record in records], dtype=torch.long)
        return _EncodedTextDataset(encodings=encodings, labels=labels)

    def _predict_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        dataset = _EncodedTextDataset(
            encodings=self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=int(self.neural_config["max_length"]),
                return_tensors="pt",
            )
        )
        dataloader = DataLoader(dataset, batch_size=self.neural_config["batch_size"], shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def _serialize_input(self, example: FinFactExample, rationale_sentences: Sequence[str]) -> str:
        rationale_text = " ".join(rationale_sentences)
        digest_text = example.digest if self.feature_flags.get("use_digest", True) else "N/A"
        context_text = example.context if self.feature_flags.get("use_context", True) else "N/A"
        return (
            f"Claim: {example.claim}\n"
            f"Scientific Digest: {digest_text or 'N/A'}\n"
            f"Selected Rationale: {rationale_text or 'N/A'}\n"
            f"Context: {context_text}"
        )

    def _serialize_rationale_only(self, record: NeuralTrainingExample) -> str:
        return (
            f"Claim: {record.claim}\n"
            f"Scientific Digest: {record.digest if self.feature_flags.get('use_digest', True) else 'N/A'}\n"
            f"Selected Rationale: {' '.join(record.rationale_sentences) or 'N/A'}\n"
            f"Context: {' '.join(record.rationale_sentences) or 'N/A'}"
        )

    def _serialize_without_rationale(self, record: NeuralTrainingExample) -> str:
        context = record.context
        for sentence in record.rationale_sentences:
            context = context.replace(sentence, " ")
        context = " ".join(context.split()) or "N/A"
        return (
            f"Claim: {record.claim}\n"
            f"Scientific Digest: {record.digest if self.feature_flags.get('use_digest', True) else 'N/A'}\n"
            f"Selected Rationale: N/A\n"
            f"Context: {context}"
        )

    def _query_text(self, example: FinFactExample) -> str:
        if self.feature_flags.get("use_digest", True):
            return example.claim_with_digest
        return example.claim

    @staticmethod
    def _resolve_device(configured_device: str) -> torch.device:
        if configured_device == "cpu":
            return torch.device("cpu")
        if configured_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _initialize_runtime(self, config: Dict) -> None:
        self.config = config
        self.aligner = PseudoRationaleAligner(**config["rationale"])
        self.explainer = StructuredExplanationBuilder(**config["explanations"])
        self.counterfactual_generator = CounterfactualGenerator(
            max_per_example=config["counterfactuals"]["max_per_example"]
        )
        self.neural_config = config["neural"]
        self.feature_flags = config.get(
            "feature_flags",
            {"use_digest": True, "use_rationale": True, "use_context": True},
        )
        self.labels = config.get("label_list", DEFAULT_LABELS)
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.device = self._resolve_device(self.neural_config["device"])
