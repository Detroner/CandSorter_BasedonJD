from __future__ import annotations

from typing import Any, Dict, Mapping

from utils import DataCompletenessReport, days_since, safe_float, safe_get, score_from_z


def score_signals(
    candidate: Mapping[str, Any],
    corpus_stats: Mapping[str, Mapping[str, float]],
    completeness: DataCompletenessReport,
) -> Dict[str, Any]:
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    score = 50.0
    note_parts = []

    response = safe_float(signals.get("recruiter_response_rate"), -1)
    if response >= 0.60:
        score += 10.0
        note_parts.append("healthy recruiter response")
    elif 0 <= response < 0.20:
        score -= 8.0
        note_parts.append("low recruiter response")

    active_days = days_since(signals.get("last_active_date"), default=999)
    if active_days <= 30:
        score += 8.0
        note_parts.append("active recently")
    elif active_days > 120:
        score -= 6.0
        note_parts.append("stale platform activity")

    github = safe_float(signals.get("github_activity_score"), -1)
    github_missing = github <= 0
    if github >= 60:
        score += 6.0
        note_parts.append("strong github activity")
    elif github_missing and not completeness.component_active("github_activity_score"):
        note_parts.append("github unavailable across corpus")
    elif github > 0:
        github_stats = corpus_stats.get("github_activity_score", {"mean": 0.0, "std": 1.0})
        score += 10.0 * (score_from_z(github, github_stats["mean"], github_stats["std"]) - 0.6)

    connections = safe_float(signals.get("connection_count"), -1)
    if connections >= 0:
        conn_stats = corpus_stats.get("connection_count", {"mean": 0.0, "std": 1.0})
        score += 6.0 * (score_from_z(connections, conn_stats["mean"], conn_stats["std"]) - 0.6)

    if github_missing:
        score *= completeness.missing_penalty("github_activity_score", mild=0.98, strong=0.95)

    score = max(0.0, min(100.0, score))
    return {
        "score": score,
        "signal_note": ", ".join(note_parts) if note_parts else "limited behavioral signal",
        "active_days": active_days,
    }
