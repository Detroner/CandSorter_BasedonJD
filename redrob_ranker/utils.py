from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


REFERENCE_DATE = date(2026, 6, 10)


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidates from .jsonl, .jsonl.gz, or a JSON array sample file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item:
                yield item
        return

    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return default if cur is None else cur


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def safe_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value).date()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%Y"):
            try:
                parsed = datetime.strptime(text[: len(fmt)], fmt)
                return parsed.date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def days_since(value: Any, default: int = 999) -> int:
    parsed = safe_date(value)
    if not parsed:
        return default
    return max(0, (REFERENCE_DATE - parsed).days)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_from_z(value: Any, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 0.6
    z = (safe_float(value) - mean) / std
    if z > 1.5:
        return 1.0
    if z > 0.5:
        return 0.8
    if z >= -0.5:
        return 0.6
    if z >= -1.5:
        return 0.35
    return 0.15


def normalize_score(value: Any, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((safe_float(value) - lo) / (hi - lo))


def text_blob(candidate: Mapping[str, Any]) -> str:
    profile = safe_get(candidate, "profile", default={}) or {}
    career = safe_get(candidate, "career_history", default=[]) or []
    skills = safe_get(candidate, "skills", default=[]) or []
    parts: List[str] = [
        str(profile.get("headline", "")),
        str(profile.get("summary", "")),
        str(profile.get("current_title", "")),
        str(profile.get("current_industry", "")),
    ]
    for role in career:
        parts.extend(
            [
                str(role.get("title", "")),
                str(role.get("industry", "")),
                str(role.get("description", "")),
                str(role.get("company", "")),
            ]
        )
    for skill in skills:
        parts.append(str(skill.get("name", "")))
    return " ".join(parts).lower()


ENGINEERING_TITLE_TERMS = (
    "engineer",
    "developer",
    "architect",
    "scientist",
    "machine learning",
    "ml ",
    " ai ",
    "artificial intelligence",
    "data",
    "research",
    "backend",
    "software",
    "sde",
    "technical lead",
    "tech lead",
    "principal",
)


def title_is_engineering(title: Any) -> bool:
    text = f" {str(title or '').lower()} "
    return any(term in text for term in ENGINEERING_TITLE_TERMS)


CONSULTING_FIRMS = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "lti",
    "ltimindtree",
    "mindtree",
    "mphasis",
    "tech mahindra",
    "deloitte",
    "pwc",
    "ey",
    "kpmg",
}


def is_consulting_company(company: Any, industry: Any = "") -> bool:
    company_text = str(company or "").lower()
    industry_text = str(industry or "").lower()
    return any(name in company_text for name in CONSULTING_FIRMS) or "it services" in industry_text or "consulting" in industry_text


def company_size_midpoint(size: Any) -> int:
    text = str(size or "").strip()
    mapping = {
        "1-10": 5,
        "11-50": 30,
        "51-200": 125,
        "201-500": 350,
        "501-1000": 750,
        "1001-5000": 3000,
        "5001-10000": 7500,
        "10001+": 15000,
    }
    return mapping.get(text, 1000)


def _field_values(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    skills = safe_get(candidate, "skills", default=[]) or []
    return {
        "years_of_experience": profile.get("years_of_experience"),
        "github_activity_score": signals.get("github_activity_score"),
        "connection_count": signals.get("connection_count"),
        "notice_period_days": signals.get("notice_period_days"),
        "endorsements_received": signals.get("endorsements_received"),
        "last_active_date": signals.get("last_active_date"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "skill_assessment_scores": signals.get("skill_assessment_scores"),
        "skill_count": len(skills),
        "career_history": safe_get(candidate, "career_history", default=[]),
    }


def compute_prepass(path: str | Path) -> tuple[Dict[str, Dict[str, float]], "DataCompletenessReport"]:
    continuous_fields = [
        "years_of_experience",
        "github_activity_score",
        "connection_count",
        "notice_period_days",
        "endorsements_received",
    ]
    values: Dict[str, List[float]] = {field: [] for field in continuous_fields}
    null_counts: Counter[str] = Counter()
    total = 0

    for candidate in iter_candidates(path):
        total += 1
        fields = _field_values(candidate)
        for field, val in fields.items():
            if val is None or val == "" or val == {} or val == []:
                null_counts[field] += 1
        for field in continuous_fields:
            val = fields.get(field)
            if val is None or val == "" or val == -1:
                continue
            values[field].append(safe_float(val))

    stats: Dict[str, Dict[str, float]] = {}
    for field, vals in values.items():
        if vals:
            stats[field] = {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) or 1.0,
            }
        else:
            stats[field] = {"mean": 0.0, "std": 1.0}
    return stats, DataCompletenessReport(total=total, null_rates={k: v / max(total, 1) for k, v in null_counts.items()})


class DataCompletenessReport:
    def __init__(self, total: int, null_rates: Dict[str, float]):
        self.total = total
        self.null_rates = null_rates

    def rate(self, field: str) -> float:
        return self.null_rates.get(field, 0.0)

    def tier(self, field: str) -> str:
        rate = self.rate(field)
        if rate > 0.50:
            return "structural"
        if rate >= 0.01:
            return "individual"
        return "meaningful"

    def missing_penalty(self, field: str, mild: float = 0.94, strong: float = 0.82) -> float:
        tier = self.tier(field)
        if tier == "structural":
            return 1.0
        if tier == "individual":
            return mild
        return strong

