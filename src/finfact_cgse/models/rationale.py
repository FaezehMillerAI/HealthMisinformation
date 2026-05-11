from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from finfact_cgse.data.dataset import FinFactExample


class PseudoRationaleAligner:
    def __init__(
        self,
        encoder_name: str,
        top_k: int = 3,
        evidence_weight: float = 0.7,
        claim_weight: float = 0.3,
    ) -> None:
        self.encoder_name = encoder_name
        self.top_k = top_k
        self.evidence_weight = evidence_weight
        self.claim_weight = claim_weight
        self.encoder = SentenceTransformer(encoder_name)

    def annotate(self, example: FinFactExample, use_evidence: bool = True) -> Dict:
        return self.annotate_with_query(example, query_text=example.claim_with_digest, use_evidence=use_evidence)

    def annotate_with_query(self, example: FinFactExample, query_text: str, use_evidence: bool = True) -> Dict:
        sentences = example.context_sentences or [example.context]
        sentence_embeddings = np.asarray(np.nan_to_num(
            self.encoder.encode(sentences, normalize_embeddings=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ), dtype=np.float64)
        claim_embedding = np.asarray(np.nan_to_num(
            self.encoder.encode(query_text, normalize_embeddings=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ), dtype=np.float64)
        with np.errstate(all="ignore"):
            claim_scores = np.nan_to_num(np.dot(sentence_embeddings, claim_embedding), nan=0.0, posinf=0.0, neginf=0.0)

        if use_evidence and example.evidence_sentences:
            evidence_embeddings = np.asarray(np.nan_to_num(
                self.encoder.encode(
                    example.evidence_sentences,
                    normalize_embeddings=True,
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ), dtype=np.float64)
            with np.errstate(all="ignore"):
                evidence_scores = np.nan_to_num(sentence_embeddings @ evidence_embeddings.T, nan=0.0, posinf=0.0, neginf=0.0)
            evidence_scores = np.nan_to_num(evidence_scores.max(axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            evidence_scores = np.zeros(len(sentences))

        scores = (self.claim_weight * claim_scores) + (self.evidence_weight * evidence_scores)
        top_indices = np.argsort(scores)[::-1][: self.top_k].tolist()
        top_indices = sorted(top_indices)
        rationale_sentences = [sentences[index] for index in top_indices]

        evidence_hits = {
            sentence.strip().lower()
            for sentence in example.evidence_sentences
            if sentence.strip()
        }
        pseudo_gold_mask = [int(sentence.strip().lower() in evidence_hits) for sentence in sentences]
        predicted_mask = [1 if idx in top_indices else 0 for idx in range(len(sentences))]

        return {
            "rationale_sentences": rationale_sentences,
            "top_indices": top_indices,
            "scores": scores.tolist(),
            "pseudo_gold_mask": pseudo_gold_mask,
            "predicted_mask": predicted_mask,
        }
