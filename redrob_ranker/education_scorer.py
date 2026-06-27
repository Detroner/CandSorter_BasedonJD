from __future__ import annotations

from typing import Any, Dict, Mapping

from .utils import safe_get


FIELD_WEIGHTS = [
    (("computer science", "machine learning", "artificial intelligence", "information retrieval", "data science", "statistics", "mathematics"), 1.00),
    (("information technology", "electronics", "electrical", "ece", "ee"), 0.82),
    (("engineering", "technology"), 0.68),
]


def _field_weight(field: str) -> float:
    lowered = field.lower()
    for terms, weight in FIELD_WEIGHTS:
        if any(term in lowered for term in terms):
            return weight
    if any(term in lowered for term in ["physics", "operations research", "economics"]):
        return 0.58
    return 0.42


def score_education(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    education = safe_get(candidate, "education", default=[]) or []
    if not education:
        return {"score": 50.0, "best_institution": "", "best_field": "", "best_tier": "unknown"}

    tier_scores = {
        "tier_1": 100.0,
        "tier_2": 80.0,
        "tier_3": 60.0,
        "tier_4": 40.0,
        "unknown": 50.0,
    }
    best = (0.0, "", "", "unknown")
    for edu in education:
        tier = str(edu.get("tier", "unknown"))
        field = str(edu.get("field_of_study", ""))
        tier_score = tier_scores.get(tier, 50.0)
        score = tier_score * _field_weight(field)
        if score > best[0]:
            best = (score, str(edu.get("institution", "")), field, tier)

    return {
        "score": max(0.0, min(100.0, best[0])),
        "best_institution": best[1],
        "best_field": best[2],
        "best_tier": best[3],
    }

