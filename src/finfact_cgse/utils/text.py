from __future__ import annotations

import ast
import re
from typing import List, Sequence


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
NUMBER_RE = re.compile(r"(?<!\w)(\$?\d[\d,]*(?:\.\d+)?%?)(?!\w)")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")


def safe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def stringify_digest(raw_digest) -> str:
    parsed = safe_literal(raw_digest)
    if isinstance(parsed, list):
        return " ".join(str(item).strip() for item in parsed if str(item).strip())
    return str(raw_digest or "").strip()


def parse_evidence_sentences(raw_evidence) -> List[str]:
    parsed = safe_literal(raw_evidence)
    if isinstance(parsed, list):
        sentences = []
        for item in parsed:
            if isinstance(item, dict) and item.get("sentence"):
                sentences.append(str(item["sentence"]).strip())
            elif isinstance(item, str) and item.strip():
                sentences.append(item.strip())
        return [sentence for sentence in sentences if sentence]
    if isinstance(parsed, str):
        return split_sentences(parsed)
    return []


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    pieces = SENTENCE_SPLIT_RE.split(text)
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    mapping = {"nei": "neutral", "not enough information": "neutral"}
    return mapping.get(label, label)


def compact_text(parts: Sequence[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def extract_key_numbers(text: str) -> List[str]:
    return NUMBER_RE.findall(text or "")


def extract_years(text: str) -> List[str]:
    return YEAR_RE.findall(text or "")


def extract_entities(text: str) -> List[str]:
    seen = []
    for match in ENTITY_RE.findall(text or ""):
        candidate = match.strip()
        if candidate not in seen:
            seen.append(candidate)
    return seen
