from __future__ import annotations

from typing import Dict, Iterable, List

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class RationaleAwareClassifier:
    def __init__(self, max_features: int = 6000, ngram_max: int = 2, c: float = 4.0, max_iter: int = 2000) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, ngram_max),
        )
        self.model = LogisticRegression(
            C=c,
            max_iter=max_iter,
        )

    def fit(self, rows: Iterable[Dict], labels: List[str]) -> None:
        rows = list(rows)
        text_features = self.vectorizer.fit_transform(row["feature_text"] for row in rows)
        dense_features = np.asarray(
            [
                [
                    row["rationale_count"],
                    row["mean_rationale_score"],
                    row["max_rationale_score"],
                ]
                for row in rows
            ],
            dtype=float,
        )
        design = hstack([text_features, dense_features])
        self.model.fit(design, labels)

    def predict(self, rows: Iterable[Dict]) -> List[str]:
        return self.model.predict(self._transform(rows)).tolist()

    def predict_proba(self, rows: Iterable[Dict]) -> np.ndarray:
        return self.model.predict_proba(self._transform(rows))

    def _transform(self, rows: Iterable[Dict]):
        rows = list(rows)
        text_features = self.vectorizer.transform(row["feature_text"] for row in rows)
        dense_features = np.asarray(
            [
                [
                    row["rationale_count"],
                    row["mean_rationale_score"],
                    row["max_rationale_score"],
                ]
                for row in rows
            ],
            dtype=float,
        )
        return hstack([text_features, dense_features])

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RationaleAwareClassifier":
        return joblib.load(path)
