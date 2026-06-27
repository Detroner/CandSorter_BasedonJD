from __future__ import annotations

from typing import Any, Dict, Mapping

from .utils import DataCompletenessReport, days_since, safe_bool, safe_float, safe_get, safe_int, score_from_z


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _field_multiplier(field: str, value: Any, completeness: DataCompletenessReport) -> float:
    if not _missing(value):
        return 1.0
    return completeness.missing_penalty(field)


def score_signals(
    candidate: Mapping[str, Any],
    corpus_stats: Mapping[str, Mapping[str, float]],
    completeness: DataCompletenessReport,
) -> Dict[str, Any]:
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    multiplier = 1.0
    notes = []

    last_active = signals.get("last_active_date")
    if _missing(last_active):
        multiplier *= _field_multiplier("last_active_date", last_active, completeness)
        active_days = 999
    else:
        active_days = days_since(last_active)
        if active_days > 180:
            multiplier *= 0.40
            notes.append("inactive 180d+")
        elif active_days > 90:
            multiplier *= 0.70
            notes.append("stale activity")
        elif active_days > 30:
            multiplier *= 0.85
            notes.append("recent enough activity")
        else:
            notes.append("active recently")

    open_to_work = signals.get("open_to_work_flag")
    if _missing(open_to_work):
        multiplier *= _field_multiplier("open_to_work_flag", open_to_work, completeness)
    elif not safe_bool(open_to_work):
        multiplier *= 0.82
        notes.append("not explicitly open")

    response_rate = safe_float(signals.get("recruiter_response_rate"), -1)
    if response_rate < 0:
        multiplier *= _field_multiplier("recruiter_response_rate", signals.get("recruiter_response_rate"), completeness)
    elif response_rate < 0.2 and active_days > 60:
        multiplier *= 0.62
        notes.append("low recruiter response")

    interview_rate = safe_float(signals.get("interview_completion_rate"), -1)
    if interview_rate >= 0 and interview_rate < 0.5:
        multiplier *= 0.72
        notes.append("low interview completion")

    verified_email = signals.get("verified_email")
    verified_phone = signals.get("verified_phone")
    if not safe_bool(verified_email, True) and not safe_bool(verified_phone, True):
        multiplier *= 0.72
        notes.append("unverified contact")

    applications = safe_int(signals.get("applications_submitted_30d"), 0)
    if applications > 20 and 0 <= response_rate < 0.1:
        multiplier *= 0.50
        notes.append("high-application low-response pattern")

    github = safe_float(signals.get("github_activity_score"), -1)
    github_norm = 0.0
    if github >= 0:
        stats = corpus_stats.get("github_activity_score", {"mean": 0.0, "std": 1.0})
        github_norm = score_from_z(github, stats["mean"], stats["std"])

    assessments = signals.get("skill_assessment_scores") or {}
    if assessments:
        assessment_norm = sum(max(0.0, min(100.0, safe_float(v))) for v in assessments.values()) / (100.0 * len(assessments))
    else:
        assessment_norm = 0.0
    activity_cluster = max(github_norm, assessment_norm)

    saved = safe_int(signals.get("saved_by_recruiters_30d"), 0)
    appearances = safe_int(signals.get("search_appearance_30d"), 0)
    engagement_cluster = max(min(1.0, saved / 10.0), min(1.0, appearances / 100.0))

    availability_cluster = 1.0 if safe_bool(open_to_work) or applications > 0 else 0.0

    positive = 1.0 + min(0.08, activity_cluster * 0.04) + min(0.08, engagement_cluster * 0.05) + min(0.04, availability_cluster * 0.03)
    multiplier = min(1.0, multiplier * positive)
    multiplier = max(0.30, min(1.0, multiplier))

    if saved > 5:
        notes.append(f"{saved} recruiter saves")
    elif github >= 60:
        notes.append(f"GitHub {github:.0f}/100")
    elif response_rate >= 0:
        notes.append(f"response rate {response_rate:.2f}")

    return {
        "multiplier": multiplier,
        "active_days": active_days,
        "activity_cluster": activity_cluster,
        "engagement_cluster": engagement_cluster,
        "availability_cluster": availability_cluster,
        "signal_note": notes[0] if notes else "limited behavioral signal",
    }

