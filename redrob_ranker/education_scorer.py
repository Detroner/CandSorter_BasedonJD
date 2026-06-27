from __future__ import annotations

from typing import Any, Dict, Mapping

from utils import safe_get


TIER_SCORES = {
    "tier_1": 92.0,
    "tier_2": 82.0,
    "tier_3": 70.0,
    "tier_4": 55.0,
    "unknown": 45.0,
}

TECH_FIELDS = {
    "computer science",
    "artificial intelligence",
    "data science",
    "information technology",
    "mathematics",
    "statistics",
}


def score_education(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    items = safe_get(candidate, "education", default=[]) or []
    if not items:
        return {"score": 50.0, "active": False}

    best = 0.0
    for edu in items:
        field = str(edu.get("field_of_study", "")).strip().lower()
        tier = str(edu.get("tier", "unknown")).strip().lower() or "unknown"
        score = TIER_SCORES.get(tier, TIER_SCORES["unknown"])
        if field in TECH_FIELDS:
            score += 8.0
        best = max(best, score)
    return {"score": min(100.0, best), "active": True}

