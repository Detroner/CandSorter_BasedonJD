from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from utils import (
    INTEREST_ONLY_PATTERNS,
    company_size_midpoint,
    has_interest_only_language,
    is_consulting_company,
    safe_float,
    safe_get,
    safe_int,
    title_is_engineering,
    title_is_explicitly_non_engineering,
)


_GAP = r"[\w\s,.;/\-&]{0,120}?"
TECH_ACTIONS = r"(?:built|deployed|shipped|implemented|designed|owned|scaled|optimized|launched)"
TECH_OBJECTS = r"(?:pipeline|service|api|model|ranking system|retrieval system|search system|backend|microservice|embedding pipeline)"

PRODUCTION_PATTERNS = [
    rf"\b{TECH_ACTIONS}{_GAP}\b{TECH_OBJECTS}\b",
    r"reduced latency",
    r"served\s+\d+",
    r"\d+\s*[kmb]\+?\s+users",
    r"real users",
]

HIGH_DOMAIN_PATTERNS = [
    rf"\bbuilt{_GAP}\bsearch\b",
    rf"\bdeployed{_GAP}\branking\b",
    rf"\bdesigned{_GAP}\bretrieval\b",
    rf"\bimplemented{_GAP}\branking\b",
    rf"\bvector{_GAP}\bsearch\b",
    rf"\bembedding{_GAP}\bpipeline\b",
    rf"\bhybrid{_GAP}\bretrieval\b",
    r"candidate[- ]jd matching",
]

MEDIUM_DOMAIN_PATTERNS = [
    r"ranking system",
    r"recommendation engine",
    r"recommender system",
    r"retrieval pipeline",
    r"\bsemantic search\b",
    r"\bdense retrieval\b",
    r"\blearning[- ]to[- ]rank\b",
    r"\bquery understanding\b",
    r"\bmatching system\b",
    r"\benterprise search\b",
]

EXCLUDED_DOMAIN_PATTERNS = [
    r"searched for",
    r"search for data",
    r"search keywords",
    r"seo strategy",
    r"ranked on the first page of search",
    r"content creation",
]

TECH_INDUSTRIES = [
    "software",
    "internet",
    "saas",
    "ai",
    "machine learning",
    "data",
    "analytics",
    "hr tech",
    "fintech",
    "e-commerce",
    "marketplace",
]


def _compile(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


PRODUCTION_COMPILED = _compile(PRODUCTION_PATTERNS)
HIGH_DOMAIN_COMPILED = _compile(HIGH_DOMAIN_PATTERNS)
MEDIUM_DOMAIN_COMPILED = _compile(MEDIUM_DOMAIN_PATTERNS)
INTEREST_ONLY_COMPILED = _compile(INTEREST_ONLY_PATTERNS)
EXCLUDED_DOMAIN_COMPILED = _compile(EXCLUDED_DOMAIN_PATTERNS)


def _matches(compiled: Sequence[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in compiled if pattern.search(text))


def product_ship_score(description: Any) -> float:
    text = str(description or "").lower()
    if not text:
        return 0.0
    if any(pattern.search(text) for pattern in INTEREST_ONLY_COMPILED):
        return 0.0
    hits = _matches(PRODUCTION_COMPILED, text)
    return min(1.0, hits / 2.0)


def domain_score_for_text(text: str) -> float:
    lowered = text.lower()
    if any(pattern.search(lowered) for pattern in EXCLUDED_DOMAIN_COMPILED):
        return 0.0
    if any(pattern.search(lowered) for pattern in INTEREST_ONLY_COMPILED):
        return 0.0
    high = _matches(HIGH_DOMAIN_COMPILED, lowered)
    medium = _matches(MEDIUM_DOMAIN_COMPILED, lowered)
    broad = sum(
        1
        for term in ["retrieval", "ranking", "recommendation", "matching", "embeddings", "vector search"]
        if term in lowered
    )
    action = 1 if re.search(rf"\b{TECH_ACTIONS}\b", lowered) else 0
    substance = high * 0.45 + medium * 0.18 + min(0.12, broad * 0.03) + action * 0.12
    if action == 0 and high == 0:
        substance = min(substance, 0.18)
    return min(1.0, substance)


def _role_is_productish(role: Mapping[str, Any]) -> bool:
    industry = str(role.get("industry", "")).lower()
    company = str(role.get("company", "")).lower()
    size = company_size_midpoint(role.get("company_size"))
    if any(term in industry for term in TECH_INDUSTRIES):
        return True
    if size is not None and size <= 500 and not is_consulting_company(company, industry) and title_is_engineering(role.get("title", "")):
        return True
    return product_ship_score(role.get("description")) > 0.4 and not is_consulting_company(company, industry)


def score_experience(candidate: Mapping[str, Any], jd_config: Mapping[str, Any], corpus_stats: Mapping[str, Mapping[str, float]]) -> float:
    years = safe_float(safe_get(candidate, "profile", "years_of_experience"), 0.0)
    exp_min = safe_float(jd_config.get("experience_min"), 5)
    exp_max = safe_float(jd_config.get("experience_max"), 9)
    if exp_min <= years <= exp_max:
        band = 1.0
    elif exp_min - 1 <= years <= exp_max + 1:
        band = 0.90
    elif exp_min - 2 <= years <= exp_max + 2:
        band = 0.68
    elif exp_min - 3 <= years <= exp_max + 4:
        band = 0.42
    else:
        band = 0.18
    return 100.0 * band


def score_career(candidate: Mapping[str, Any], jd_config: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")
    years = safe_float(profile.get("years_of_experience"), 0.0)

    score = 15.0
    reasons: List[str] = []
    domain_roles = 0
    total_domain = 0.0
    product_roles = 0
    production_roles = 0
    consulting_roles = 0
    startup_roles = 0
    technical_roles = 0
    recent_short_roles = 0
    domain_terms: List[str] = []
    interest_only_roles = 0
    required_skills = {str(skill).lower() for skill in (jd_config.get("required_skills") or [])}

    if title_is_engineering(current_title):
        score += 12.0
        technical_roles += 1
    elif title_is_explicitly_non_engineering(current_title):
        score -= 10.0
    current_kind = "eng" if title_is_engineering(current_title) else "non_eng" if title_is_explicitly_non_engineering(current_title) else "unknown"
    opposite_title_roles = 0

    for role in roles:
        title = str(role.get("title", ""))
        desc = str(role.get("description", ""))
        text = f"{title} {role.get('industry', '')} {desc}".lower()

        if title_is_engineering(title):
            technical_roles += 1
            score += 3.0
            if current_kind == "non_eng":
                opposite_title_roles += 1
        elif title_is_explicitly_non_engineering(title):
            score -= 1.5
            if current_kind == "eng":
                opposite_title_roles += 1

        if has_interest_only_language(text):
            interest_only_roles += 1

        if _role_is_productish(role):
            product_roles += 1
            score += 4.0

        if is_consulting_company(role.get("company"), role.get("industry")):
            consulting_roles += 1

        role_size = company_size_midpoint(role.get("company_size"))
        if role_size is not None and role_size <= 200:
            startup_roles += 1
            score += 1.5

        ship = product_ship_score(desc)
        if ship > 0:
            production_roles += 1
            score += 8.0 * ship

        role_domain = domain_score_for_text(text)
        if role_domain > 0:
            domain_roles += 1
            total_domain += role_domain
            score += 12.0 * role_domain
            for term in ["retrieval", "ranking", "recommendation", "search", "matching", "embedding"]:
                if term in text and term not in domain_terms:
                    domain_terms.append(term)

        jd_overlap = sum(1 for term in required_skills if term and re.search(rf"\b{re.escape(term)}\b", text))
        if jd_overlap:
            score += min(8.0, jd_overlap * 2.0)

        title_lower = title.lower()
        if any(term in title_lower for term in ["founding", "lead", "principal", "staff", "head", "architect"]):
            score += 3.0
        if safe_int(role.get("duration_months"), 0) < 18:
            recent_short_roles += 1

    if domain_roles >= 2 and total_domain >= 0.45:
        score += 14.0
        reasons.append("multi-role search/ranking evidence")
    elif domain_roles == 1 and total_domain >= 0.70 and production_roles:
        score += 7.0
        reasons.append("strong single-role domain evidence")

    if product_roles:
        reasons.append("product-company exposure")
    if production_roles:
        reasons.append("production shipping evidence")

    jd_text = str(jd_config.get("text", "")).lower()
    consulting_penalty_enabled = (
        "consulting firm" in jd_text
        or "consulting-only" in jd_text
        or "not preferred consulting" in jd_text
        or any(name in jd_text for name in ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"])
    )
    if consulting_penalty_enabled and roles and consulting_roles == len(roles) and production_roles == 0:
        score -= 18.0
    elif consulting_penalty_enabled and roles and consulting_roles == len(roles):
        score -= 7.0

    if len(roles) >= 3 and recent_short_roles >= 3 and years < 6:
        score -= 8.0

    if roles and opposite_title_roles >= max(1, len(roles) // 2):
        score -= 10.0

    if production_roles == 0:
        score -= 8.0

    if technical_roles == 0:
        score -= 18.0

    if interest_only_roles and technical_roles == 0 and domain_roles == 0:
        score -= 10.0

    if technical_roles < 1 and domain_roles < 1 and product_roles < 1:
        score -= 10.0

    if any(term in f"{current_title} {current_industry}".lower() for term in ["research", "academic"]) and production_roles == 0:
        score -= 10.0

    score = max(0.0, min(100.0, score))
    return {
        "score": score,
        "domain_score": min(1.0, total_domain),
        "domain_roles": domain_roles,
        "domain_terms": domain_terms[:4],
        "product_roles": product_roles,
        "production_roles": production_roles,
        "consulting_roles": consulting_roles,
        "startup_roles": startup_roles,
        "technical_roles": technical_roles,
        "interest_only_roles": interest_only_roles,
        "reasons": reasons[:4],
    }
