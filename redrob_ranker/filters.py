from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from career_scorer import domain_score_for_text, product_ship_score
from utils import (
    INTEREST_ONLY_PATTERNS,
    get_reference_date,
    is_consulting_company,
    list_skill_names,
    safe_bool,
    safe_date,
    safe_date_info,
    safe_float,
    safe_get,
    safe_int,
    text_blob,
    title_is_engineering,
    title_is_explicitly_non_engineering,
)


FOUNDING_YEAR_RE = re.compile(r"founded\s+(?:in\s+)?(20\d{2}|19\d{2})", re.IGNORECASE)
INTEREST_ONLY_RE = re.compile("|".join(re.escape(p) for p in INTEREST_ONLY_PATTERNS), re.IGNORECASE)

DIRECT_TECH_SKILLS = {
    "python",
    "pyspark",
    "machine learning",
    "nlp",
    "retrieval",
    "ranking",
    "semantic search",
    "faiss",
    "vector search",
    "backend",
    "fastapi",
    "django",
    "flask",
    "microservices",
    "rest apis",
    "grpc",
}


def _structured_technical_evidence(candidate: Mapping[str, Any]) -> bool:
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    skill_names = {name.lower() for name in list_skill_names(candidate)}
    if title_is_engineering(profile.get("current_title", "")):
        return True
    if any(title_is_engineering(role.get("title", "")) for role in roles):
        return True
    direct_hits = DIRECT_TECH_SKILLS.intersection(skill_names)
    if len(direct_hits) >= 3:
        return True
    if len(direct_hits) >= 2 and any(domain_score_for_text(str(role.get("description", ""))) >= 0.25 for role in roles):
        return True
    supporting_hits = skill_names.intersection({"spark", "airflow", "kafka", "apache beam", "databricks", "dbt", "hadoop"})
    if len(supporting_hits) >= 2 and any(title_is_engineering(role.get("title", "")) for role in roles):
        return True
    for role in roles:
        desc = str(role.get("description", ""))
        if product_ship_score(desc) > 0.45 and domain_score_for_text(desc) >= 0.25:
            return True
    return False


def _profile_is_inconsistent(candidate: Mapping[str, Any]) -> bool:
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    current_title = profile.get("current_title", "")
    if title_is_engineering(current_title) or title_is_explicitly_non_engineering(current_title):
        title_kinds = []
        for role in roles:
            role_title = role.get("title", "")
            if title_is_engineering(role_title):
                title_kinds.append("eng")
            elif title_is_explicitly_non_engineering(role_title):
                title_kinds.append("non_eng")
        if title_kinds:
            current_kind = "eng" if title_is_engineering(current_title) else "non_eng"
            opposite = "non_eng" if current_kind == "eng" else "eng"
            return title_kinds.count(opposite) >= max(1, len(title_kinds))
    return False


def _interest_only_candidate(candidate: Mapping[str, Any]) -> bool:
    blob = text_blob(candidate)
    if not INTEREST_ONLY_RE.search(blob):
        return False
    return not _structured_technical_evidence(candidate)


def _domain_disqualified(blob: str) -> bool:
    excluded_count = sum(blob.count(term) for term in ["computer vision", "image classification", "speech recognition", "robotics"])
    ir_count = sum(blob.count(term) for term in ["nlp", "retrieval", "ranking", "recommendation", "search", "embedding", "relevance"])
    return excluded_count >= 2 and ir_count == 0


def _duration_plausible(role: Mapping[str, Any]) -> bool:
    start = safe_date(role.get("start_date"))
    end = safe_date(role.get("end_date")) or get_reference_date()
    if not start or not end or end < start:
        return False
    actual_months = max(0, (end.year - start.year) * 12 + (end.month - start.month) + 1)
    claimed = safe_int(role.get("duration_months"), 0)
    return claimed <= actual_months + 4


def _company_founding_flag(role: Mapping[str, Any]) -> bool:
    text = str(role.get("description", ""))
    found = FOUNDING_YEAR_RE.search(text)
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
        start_date, start_status = safe_date_info(role.get("start_date"))
        end_date, end_status = safe_date_info(role.get("end_date"))
        if start_status == "invalid" or end_status == "invalid":
            flags.append("malformed career date")
            break
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

    if _interest_only_candidate(candidate):
        reasons.append("interest in AI/Python without execution evidence")

    has_technical_title = title_is_engineering(title)
    has_technical_history = any(title_is_engineering(role.get("title", "")) for role in roles)
    has_structured_evidence = _structured_technical_evidence(candidate)
    if not has_technical_title and not has_technical_history and not has_structured_evidence:
        reasons.append("no technical engineering evidence")

    if title_is_explicitly_non_engineering(title) and not has_structured_evidence:
        reasons.append("current role is non-engineering without technical proof")

    if _profile_is_inconsistent(candidate) and not has_technical_title:
        reasons.append("profile title/history inconsistency without technical proof")

    if roles:
        consulting_roles = [
            role
            for role in roles
            if is_consulting_company(role.get("company"), role.get("industry"))
        ]
        productish = any(product_ship_score(role.get("description")) > 0.45 for role in roles)
        if len(consulting_roles) == len(roles) and not productish and domain_score_for_text(blob) < 0.35:
            reasons.append("consulting-only career without product shipping evidence")

    if _domain_disqualified(blob):
        reasons.append("primary domain is CV, speech, or robotics without NLP/IR exposure")

    if "langchain" in blob and domain_score_for_text(blob) < 0.30 and "production" not in blob:
        reasons.append("recent framework-only AI exposure without production retrieval depth")

    return {
        "passed": not reasons,
        "reason": "; ".join(dict.fromkeys(reasons)),
        "honeypot_flags": flags,
    }
