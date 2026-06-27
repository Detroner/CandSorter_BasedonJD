from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from .career_scorer import domain_score_for_text, product_ship_score
from .utils import (
    REFERENCE_DATE,
    is_consulting_company,
    safe_bool,
    safe_date,
    safe_float,
    safe_get,
    safe_int,
    text_blob,
    title_is_engineering,
)


TECHNICAL_TEXT_TERMS = [
    "python",
    "machine learning",
    "ml",
    "ai engineer",
    "data scientist",
    "backend",
    "software",
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "nlp",
    "embedding",
    "vector",
    "production",
]


def _technical_evidence(blob: str) -> bool:
    return any(term in blob for term in TECHNICAL_TEXT_TERMS) or domain_score_for_text(blob) >= 0.35


def _domain_disqualified(blob: str) -> bool:
    excluded_count = sum(blob.count(term) for term in ["computer vision", "image classification", "speech recognition", "robotics"])
    ir_count = sum(blob.count(term) for term in ["nlp", "retrieval", "ranking", "recommendation", "search", "embedding", "relevance"])
    return excluded_count >= 2 and ir_count == 0


def _duration_plausible(role: Mapping[str, Any]) -> bool:
    start = safe_date(role.get("start_date"))
    end = safe_date(role.get("end_date")) or REFERENCE_DATE
    if not start or not end or end < start:
        return False
    actual_months = max(0, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    claimed = safe_int(role.get("duration_months"), 0)
    return claimed <= actual_months + 4


def _company_founding_flag(role: Mapping[str, Any]) -> bool:
    text = str(role.get("description", ""))
    found = re.search(r"founded\s+(?:in\s+)?(20\d{2}|19\d{2})", text, flags=re.I)
    start = safe_date(role.get("start_date"))
    if not found or not start:
        return False
    founding_year = safe_int(found.group(1), 0)
    return start.year < founding_year


def honeypot_flags(candidate: Mapping[str, Any]) -> List[str]:
    flags: List[str] = []
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    skills = safe_get(candidate, "skills", default=[]) or []
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    assessments = signals.get("skill_assessment_scores") or {}

    expert_zero = [
        s
        for s in skills
        if str(s.get("proficiency", "")).lower() == "expert" and safe_int(s.get("duration_months"), 0) == 0
    ]
    if expert_zero:
        flags.append("expert skill with zero duration")

    for skill in skills:
        name = str(skill.get("name", "")).lower()
        if str(skill.get("proficiency", "")).lower() != "expert":
            continue
        for assessed_name, score in assessments.items():
            assessed = str(assessed_name).lower()
            if (assessed == name or assessed in name or name in assessed) and safe_float(score, 100) < 40:
                flags.append("expert claim contradicted by low assessment")
                break

    for role in roles:
        if not _duration_plausible(role):
            flags.append("implausible career date duration")
            break
        if _company_founding_flag(role):
            flags.append("tenure starts before company founding")
            break

    signup = safe_date(signals.get("signup_date"))
    last_active = safe_date(signals.get("last_active_date"))
    if signup and last_active and signup > last_active:
        flags.append("signup after last active date")
    if safe_float(signals.get("profile_completeness_score"), 0) >= 99 and not last_active:
        flags.append("complete profile with missing activity")

    years = safe_float(profile.get("years_of_experience"), 0)
    duration_years = sum(safe_int(role.get("duration_months"), 0) for role in roles) / 12.0
    if duration_years > years + 5 and len(roles) <= 3:
        flags.append("career duration exceeds stated experience")

    return sorted(set(flags))


def apply_filters(candidate: Mapping[str, Any], jd_config: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    title = profile.get("current_title", "")
    country = str(profile.get("country", "")).strip().lower()
    blob = text_blob(candidate)
    flags = honeypot_flags(candidate)
    reasons: List[str] = []

    if country and country != "india" and not safe_bool(signals.get("willing_to_relocate"), False):
        reasons.append("outside India and not willing to relocate")

    has_technical_title = title_is_engineering(title)
    has_technical_history = any(title_is_engineering(role.get("title", "")) for role in roles)
    if not has_technical_title and not has_technical_history and not _technical_evidence(blob):
        reasons.append("no technical engineering evidence")

    if roles:
        consulting_roles = [
            role
            for role in roles
            if is_consulting_company(role.get("company"), role.get("industry"))
        ]
        productish = any(product_ship_score(role.get("description")) > 0.45 for role in roles)
        if len(consulting_roles) == len(roles) and not productish and domain_score_for_text(blob) < 0.3:
            reasons.append("consulting-only career without product shipping evidence")

    if _domain_disqualified(blob):
        reasons.append("primary domain is CV, speech, or robotics without NLP/IR exposure")

    if "langchain" in blob and domain_score_for_text(blob) < 0.25 and "production" not in blob:
        reasons.append("recent framework-only AI exposure without production retrieval depth")

    return {
        "passed": not reasons,
        "reason": "; ".join(reasons),
        "honeypot_flags": flags,
    }

