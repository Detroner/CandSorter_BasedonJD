from __future__ import annotations

import gzip
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple


_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T\s].*)?$")
_CANDIDATE_ID_RE = re.compile(r"^CAND_\d{7}$")

INTEREST_ONLY_PATTERNS = (
    "interested in transitioning",
    "building competence",
    "curious about ai",
    "curious about how ai tools could augment",
    "self-learner level",
    "self-directed ml projects",
    "learning modern ml",
    "experimented with chatgpt",
    "emerging ai capabilities",
    "interested in expanding into broader backend",
)


def iter_candidates(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    name = p.name.lower()

    if name.endswith(".json") and not name.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {p}, got {type(data).__name__}")
        for item in data:
            if item:
                yield item
        return

    opener = gzip.open if name.endswith(".gz") else open
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


def valid_candidate_id(candidate_id: Any) -> bool:
    return bool(_CANDIDATE_ID_RE.match(str(candidate_id or "").strip()))


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


@lru_cache(maxsize=65536)
def _parse_date_cached(value: str) -> Optional[date]:
    text = value.strip()
    match = _ISO_DATE_RE.match(text)
    if match:
        year, month, day = (int(match.group(i)) for i in range(1, 4))
        try:
            return date(year, month, day)
        except ValueError:
            return None

    for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def safe_date_info(value: Any) -> Tuple[Optional[date], str]:
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, date) and not isinstance(value, datetime):
        return value, "ok"
    if isinstance(value, datetime):
        return value.date(), "ok"
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value).date(), "ok"
        except (OverflowError, OSError, ValueError):
            return None, "invalid"
    if isinstance(value, str):
        parsed = _parse_date_cached(value)
        return parsed, "ok" if parsed else "invalid"
    return None, "invalid"


def safe_date(value: Any) -> Optional[date]:
    parsed, _ = safe_date_info(value)
    return parsed


def get_reference_date() -> date:
    override = os.getenv("REDROB_REFERENCE_DATE", "").strip()
    if override:
        parsed = _parse_date_cached(override)
        if parsed:
            return parsed
    return date.today()


def days_since(value: Any, default: int = 999, reference_date: Optional[date] = None) -> int:
    parsed = safe_date(value)
    if not parsed:
        return default
    reference = reference_date or get_reference_date()
    return max(0, (reference - parsed).days)


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


def has_interest_only_language(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(pattern in lowered for pattern in INTEREST_ONLY_PATTERNS)


ENGINEERING_TITLE_TERMS = (
    "engineer",
    "developer",
    "architect",
    "scientist",
    "machine learning",
    "ml ",
    " ai ",
    "artificial intelligence",
    "backend",
    "software",
    "sde",
    "technical lead",
    "tech lead",
    "platform",
    "data engineer",
)


NON_SOFTWARE_ENGINEERING_TERMS = (
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "chemical engineer",
    "industrial engineer",
)


NON_ENGINEERING_TITLE_TERMS = (
    "marketing",
    "sales",
    "hr",
    "accountant",
    "support",
    "operations manager",
    "content writer",
    "graphic designer",
    "business analyst",
)


def title_is_engineering(title: Any) -> bool:
    text = f" {str(title or '').lower()} "
    if any(term in text for term in NON_SOFTWARE_ENGINEERING_TERMS):
        return False
    return any(term in text for term in ENGINEERING_TITLE_TERMS)


def title_is_explicitly_non_engineering(title: Any) -> bool:
    text = f" {str(title or '').lower()} "
    return any(term in text for term in NON_ENGINEERING_TITLE_TERMS)


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


def company_size_midpoint(size: Any) -> Optional[int]:
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
    return mapping.get(text)


def list_skill_names(candidate: Mapping[str, Any]) -> List[str]:
    skills = safe_get(candidate, "skills", default=[]) or []
    names = []
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _field_values(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    skills = safe_get(candidate, "skills", default=[]) or []
    return {
        "years_of_experience": profile.get("years_of_experience"),
        "location": profile.get("location"),
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
        "education": safe_get(candidate, "education", default=[]),
    }


def compute_prepass(path: str | Path) -> tuple[Dict[str, Dict[str, float]], "DataCompletenessReport"]:
    # Null-rate tracking covers both continuous and structural fields like education;
    # only the continuous subset below receives corpus mean/std statistics.
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

    def component_active(self, field: str) -> bool:
        return self.tier(field) != "structural"


def extract_text_from_docx(path: str | Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    chunks = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()
