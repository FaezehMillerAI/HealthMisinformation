from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def rationale_metrics(pseudo_gold_masks: Iterable[List[int]], predicted_masks: Iterable[List[int]]) -> Dict[str, float]:
    gold = []
    pred = []
    for gold_mask, pred_mask in zip(pseudo_gold_masks, predicted_masks):
        if not gold_mask:
            continue
        gold.extend(gold_mask)
        pred.extend(pred_mask)
    if not gold:
        return {"rationale_precision": 0.0, "rationale_recall": 0.0, "rationale_f1": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "rationale_precision": precision,
        "rationale_recall": recall,
        "rationale_f1": f1,
    }


def rouge1_f1(references: List[str], predictions: List[str]) -> float:
    scores = []
    for reference, prediction in zip(references, predictions):
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0
