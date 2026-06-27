from __future__ import annotations

from typing import Any, Mapping

from utils import safe_float, safe_get


def _format_skills(features: Mapping[str, Any]) -> str:
    strong = features.get("skill", {}).get("strong_evidence_skills", []) or []
    matched = features.get("skill", {}).get("matched_skills", []) or []
    skills = strong or matched
    if not skills:
        return "limited explicit JD skill evidence"
    if len(skills) == 1:
        return f"evidence in {skills[0]}"
    return f"evidence in {skills[0]} and {skills[1]}"


def make_reasoning(candidate: Mapping[str, Any], features: Mapping[str, Any], rank: int) -> str:
    del rank
    profile = safe_get(candidate, "profile", default={}) or {}
    career = features.get("career", {})
    location = features.get("location", {})
    signals = features.get("signals", {})
    skill = features.get("skill", {})
    filter_result = features.get("filter", {}) or {}

    years = safe_float(profile.get("years_of_experience"), 0.0)
    title = str(profile.get("current_title", "candidate")).strip() or "candidate"
    location_note = location.get("location_note", "location fit unclear")
    notice = location.get("notice_days", 180)
    skill_note = _format_skills(features)
    domain_terms = career.get("domain_terms", []) or []
    domain_note = ", ".join(domain_terms[:2]) if domain_terms else "limited search/ranking evidence"
    score = float(features.get("final_score", 0.0))
    honeypot_flags = filter_result.get("honeypot_flags", []) or []

    if honeypot_flags and score <= 0.01:
        fit = "Disqualified"
        body = f"{title} with {years:.1f}y; profile flagged for suspicious signals ({'; '.join(honeypot_flags[:2])})."
    elif skill.get("interest_only"):
        fit = "Adjacent candidate only"
        body = f"{title} with {years:.1f}y; interest in AI/Python is present but execution evidence is weak. {location_note}; notice {notice}d."
    elif score >= 0.70:
        fit = "Grounded fit"
        body = f"{title} with {years:.1f}y; {skill_note}. Career evidence: {domain_note}. {location_note}; {signals.get('signal_note', 'limited behavioral signal')}."
    elif score >= 0.45:
        fit = "Partial fit"
        body = f"{title} with {years:.1f}y; {skill_note}. Career evidence: {domain_note}. {location_note}; {signals.get('signal_note', 'limited behavioral signal')}."
    else:
        fit = "Weak fit"
        body = f"{title} with {years:.1f}y; {skill_note}. Career evidence: {domain_note}. {location_note}; {signals.get('signal_note', 'limited behavioral signal')}."

    return f"{fit}: {body}".replace("\n", " ").strip()
