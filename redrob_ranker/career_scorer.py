from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple

from .utils import (
    company_size_midpoint,
    is_consulting_company,
    safe_float,
    safe_get,
    safe_int,
    score_from_z,
    title_is_engineering,
)


PRODUCTION_PATTERNS = [
    r"\bdeployed\b",
    r"\bshipped\b",
    r"\bscaled\b",
    r"\blaunched\b",
    r"reduced latency",
    r"a/?b test",
    r"served\s+\d+",
    r"\d+\s*[kmb]\+?\s+users",
    r"built and maintained",
    r"\bproduction\b",
    r"real users",
    r"end[- ]to[- ]end",
    r"owned .* system",
    r"improved .* metric",
]

HIGH_DOMAIN_PATTERNS = [
    r"built\s+[^.]{0,80}\bsearch\b",
    r"deployed\s+[^.]{0,80}\branking\b",
    r"shipped\s+[^.]{0,80}\brecommendation\b",
    r"designed\s+[^.]{0,80}\bretrieval\b",
    r"scaled\s+[^.]{0,80}\bmatching\b",
    r"implemented\s+[^.]{0,80}\branking\b",
    r"production\s+[^.]{0,80}\bsearch\b",
    r"vector\s+[^.]{0,40}\bsearch\b",
    r"embedding\s+[^.]{0,50}\bpipeline\b",
    r"hybrid\s+[^.]{0,40}\bretrieval\b",
    r"candidate[- ]jd matching",
]

MEDIUM_DOMAIN_PATTERNS = [
    r"search relevance",
    r"ranking system",
    r"recommendation engine",
    r"recommender system",
    r"retrieval pipeline",
    r"a/?b test[^.]{0,80}ranking",
    r"\bsemantic search\b",
    r"\bdense retrieval\b",
    r"\blearning[- ]to[- ]rank\b",
    r"\bquery understanding\b",
    r"\bmatching system\b",
]

EXCLUDED_DOMAIN_PATTERNS = [
    r"searched for",
    r"search for data",
    r"data entry",
    r"searched records",
    r"searching records",
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


def _matches(patterns: List[str], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.I))


def product_ship_score(description: Any) -> float:
    text = str(description or "").lower()
    if not text:
        return 0.0
    hits = _matches(PRODUCTION_PATTERNS, text)
    return min(1.0, hits / 3.0)


def domain_score_for_text(text: str) -> float:
    lowered = text.lower()
    if any(re.search(pattern, lowered, flags=re.I) for pattern in EXCLUDED_DOMAIN_PATTERNS):
        return 0.0
    high = _matches(HIGH_DOMAIN_PATTERNS, lowered)
    medium = _matches(MEDIUM_DOMAIN_PATTERNS, lowered)
    broad = sum(
        1
        for term in ["retrieval", "ranking", "recommendation", "relevance", "matching", "embeddings", "vector search"]
        if term in lowered
    )
    return min(1.0, high * 0.40 + medium * 0.22 + min(0.25, broad * 0.06))


def _role_is_productish(role: Mapping[str, Any]) -> bool:
    industry = str(role.get("industry", "")).lower()
    company = str(role.get("company", "")).lower()
    size = company_size_midpoint(role.get("company_size"))
    if any(term in industry for term in TECH_INDUSTRIES):
        return True
    if size <= 500 and not is_consulting_company(company, industry):
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
    elif exp_min - 2 <= years <= exp_max + 3:
        band = 0.75
    elif 2.5 <= years <= 14:
        band = 0.58
    else:
        band = 0.35
    stats = corpus_stats.get("years_of_experience", {"mean": 0.0, "std": 1.0})
    z_band = score_from_z(years, stats["mean"], stats["std"])
    return 100.0 * (0.78 * band + 0.22 * z_band)


def score_career(candidate: Mapping[str, Any], jd_config: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    roles = safe_get(candidate, "career_history", default=[]) or []
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")
    years = safe_float(profile.get("years_of_experience"), 0.0)

    score = 20.0
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

    if title_is_engineering(current_title):
        score += 13
        technical_roles += 1
    else:
        score -= 12

    for role in roles:
        title = str(role.get("title", ""))
        desc = str(role.get("description", ""))
        text = f"{title} {role.get('industry', '')} {desc}".lower()

        if title_is_engineering(title):
            technical_roles += 1
            score += 2.5

        if _role_is_productish(role):
            product_roles += 1
            score += 5.0

        if is_consulting_company(role.get("company"), role.get("industry")):
            consulting_roles += 1

        if company_size_midpoint(role.get("company_size")) <= 200:
            startup_roles += 1
            score += 2.0

        ship = product_ship_score(desc)
        if ship > 0:
            production_roles += 1
            score += 10.0 * ship

        role_domain = domain_score_for_text(text)
        if role_domain > 0:
            domain_roles += 1
            total_domain += role_domain
            score += 13.0 * role_domain
            for term in ["retrieval", "ranking", "recommendation", "search", "matching", "embedding"]:
                if term in text and term not in domain_terms:
                    domain_terms.append(term)

        title_lower = title.lower()
        if any(term in title_lower for term in ["founding", "lead", "principal", "staff", "head", "architect"]):
            score += 4.0
        if safe_int(role.get("duration_months"), 0) < 18:
            recent_short_roles += 1

    if domain_roles >= 2 and total_domain >= 0.4:
        score += 16.0
        reasons.append("multi-role search/ranking evidence")
    elif domain_roles == 1 and total_domain >= 0.65 and production_roles:
        score += 8.0
        reasons.append("strong single-role domain evidence")

    if product_roles:
        reasons.append("product-company exposure")
    if production_roles:
        reasons.append("production shipping evidence")
    if startup_roles:
        reasons.append("startup-sized team exposure")

    if roles and consulting_roles == len(roles) and production_roles == 0:
        score -= 22.0
    elif roles and consulting_roles == len(roles):
        score -= 9.0

    if len(roles) >= 3 and recent_short_roles >= 3 and years < 6:
        score -= 9.0

    if production_roles == 0:
        score -= 10.0

    if product_roles == 0 and company_size_midpoint(profile.get("current_company_size")) > 10000:
        score -= 5.0

    if any(term in f"{current_title} {current_industry}".lower() for term in ["research", "academic"]) and production_roles == 0:
        score -= 14.0

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
        "reasons": reasons[:4],
        "current_company": current_company,
    }

