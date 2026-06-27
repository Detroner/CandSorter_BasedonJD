from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .utils import company_size_midpoint, is_consulting_company, safe_float, safe_get


def _template_idx(candidate_id: str) -> int:
    idx = int(hashlib.md5(candidate_id.encode("utf-8")).hexdigest(), 16) % 4
    assert idx < 4
    return idx


def _company_type(candidate: Mapping[str, Any]) -> str:
    profile = safe_get(candidate, "profile", default={}) or {}
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    size = company_size_midpoint(profile.get("current_company_size"))
    if is_consulting_company(company, industry):
        return "consulting background"
    if size <= 200:
        return "startup/product background"
    return "product/tech background"


def _top_skills(features: Mapping[str, Any]) -> tuple[str, str]:
    skills = features.get("skill", {}).get("matched_skills", []) or []
    first = skills[0] if len(skills) >= 1 else "Python/ML"
    second = skills[1] if len(skills) >= 2 else "production systems"
    return first, second


def _concern_or_strength(candidate: Mapping[str, Any], features: Mapping[str, Any], rank: int) -> str:
    location = features.get("location", {})
    career = features.get("career", {})
    signals = features.get("signals", {})
    notice = location.get("notice_days", 180)
    if notice > 60:
        return f"concern: {notice}d notice"
    if signals.get("active_days", 0) > 90:
        return "concern: stale platform activity"
    if career.get("consulting_roles", 0) and career.get("product_roles", 0) == 0:
        return "concern: mostly consulting-side history"
    if career.get("domain_roles", 0) >= 2:
        return "strength: repeated search/ranking work"
    if features.get("skill", {}).get("score", 0) >= 75:
        return "strength: strong JD skill coverage"
    if rank > 50:
        return "included as an adjacent but weaker match"
    return "balanced fit with some gaps"


def make_reasoning(candidate: Mapping[str, Any], features: Mapping[str, Any], rank: int) -> str:
    profile = safe_get(candidate, "profile", default={}) or {}
    career = features.get("career", {})
    location = features.get("location", {})
    signals = features.get("signals", {})
    candidate_id = str(candidate.get("candidate_id", ""))
    years = safe_float(profile.get("years_of_experience"), 0.0)
    title = str(profile.get("current_title", "candidate")).strip() or "candidate"
    company = str(profile.get("current_company", "current company")).strip() or "current company"
    company_type = _company_type(candidate)
    top_skill, top_skill2 = _top_skills(features)
    concern = _concern_or_strength(candidate, features, rank)
    signal_note = signals.get("signal_note", "limited behavioral signal")
    location_note = location.get("location_note", "location fit unclear")
    notice = location.get("notice_days", 180)
    domain_terms = career.get("domain_terms", []) or []
    domain_note = "/".join(domain_terms[:3]) + " domain work" if domain_terms else "general ML/backend career evidence"
    production_roles = career.get("production_roles", 0)
    production_phrase = (
        f"with {production_roles} production-evidence role(s)"
        if production_roles
        else "with limited explicit production evidence"
    )

    idx = _template_idx(candidate_id)
    if idx == 0:
        text = f"{years:.1f}y {title} with {company_type}: {top_skill}, {top_skill2}. {concern}; {signal_note}."
    elif idx == 1:
        text = f"Career as {title} at {company} shows {domain_note}; {years:.1f} yrs total. {top_skill} is the strongest JD skill signal; {concern}."
    elif idx == 2:
        text = f"{title}: {top_skill} plus {top_skill2} {production_phrase}. {company_type}, {location_note}; {signal_note}."
    else:
        text = f"{years:.1f} yrs, currently {title}. {domain_note}; {concern}. Notice: {notice}d."

    if rank <= 10:
        prefix = "Top-tier fit: "
    elif rank <= 50:
        prefix = "Strong fit: "
    else:
        prefix = "Borderline top-100 fit: "
    return (prefix + text).replace("\n", " ").strip()
