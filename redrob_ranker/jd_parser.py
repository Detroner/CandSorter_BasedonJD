from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from utils import extract_text_from_docx


SKILL_HINTS = ["python", "machine learning", "nlp", "retrieval", "ranking", "search", "recommendation", "embeddings", "vector search", "backend"]
LOCATION_HINTS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore", "bengaluru", "india"]


def parse_job_description(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".docx":
        text = extract_text_from_docx(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()

    parse_warnings: List[str] = []
    exp_patterns = [
        r"experience required[:\s]*?(\d+)\s*(?:-|–|to)\s*(\d+)\s*years",
        r"required experience[:\s]*?(\d+)\s*(?:-|–|to)\s*(\d+)\s*years",
        r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*years of experience",
    ]
    exp_match = None
    for pattern in exp_patterns:
        exp_match = re.search(pattern, lowered)
        if exp_match:
            break
    if exp_match:
        exp_min = float(exp_match.group(1))
        exp_max = float(exp_match.group(2))
    else:
        exp_min, exp_max = 5.0, 9.0
        parse_warnings.append("experience range defaulted")

    notice_patterns = [
        r"buy out up to\s*(\d+)\s*days",
        r"notice period of up to\s*(\d+)\s*days",
        r"up to\s*(\d+)\s*days\s*notice",
        r"notice[:\s]*.*?(\d+)\s*days",
    ]
    notice_match = None
    for pattern in notice_patterns:
        notice_match = re.search(pattern, lowered)
        if notice_match:
            break
    max_notice = int(notice_match.group(1)) if notice_match else 30
    if not notice_match:
        parse_warnings.append("notice period defaulted")

    required_skills: List[str] = []
    for skill in SKILL_HINTS:
        if skill in lowered and skill not in required_skills:
            required_skills.append(skill)
    if "python" not in required_skills:
        required_skills.insert(0, "python")

    preferred_locations = [loc for loc in LOCATION_HINTS if loc in lowered]
    if not preferred_locations:
        preferred_locations = ["india"]

    return {
        "text": text,
        "experience_min": exp_min,
        "experience_max": exp_max,
        "max_notice_days": max_notice,
        "required_skills": required_skills,
        "preferred_locations": preferred_locations,
        "parse_warnings": parse_warnings,
    }
