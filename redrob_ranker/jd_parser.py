from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET


DEFAULT_CONSULTING_FIRMS = [
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
    "HCL",
    "LTI",
    "Mphasis",
    "Tech Mahindra",
]


def _read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras: List[str] = []
    for para in root.findall(".//w:p", ns):
        text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
        if text:
            paras.append(text)
    return "\n".join(paras)


def read_text(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".docx":
        return _read_docx_text(p)
    return p.read_text(encoding="utf-8")


def _city_hits(text: str) -> List[str]:
    cities = [
        "Pune",
        "Noida",
        "Delhi NCR",
        "Gurgaon",
        "Faridabad",
        "Hyderabad",
        "Mumbai",
        "Bangalore",
        "Bengaluru",
        "Chennai",
    ]
    hits = []
    lowered = text.lower()
    for city in cities:
        if city.lower() in lowered:
            hits.append(city)
    return hits


def parse_job_description(path: str | Path) -> Dict[str, object]:
    text = read_text(path)
    lowered = text.lower()

    preferred_locations: List[str] = []
    location_line = ""
    for line in text.splitlines():
        if "location" in line.lower():
            location_line = line
            preferred_locations.extend(_city_hits(line))
            break
    if not preferred_locations:
        preferred_locations = ["Pune", "Noida"]

    acceptable_locations = []
    for line in text.splitlines():
        if "welcome to apply" in line.lower() or "tier-1 indian cities" in line.lower():
            acceptable_locations.extend(_city_hits(line))
    for city in ["Hyderabad", "Mumbai", "Delhi NCR", "Gurgaon", "Faridabad", "Bangalore", "Bengaluru"]:
        if city.lower() in lowered and city not in acceptable_locations and city not in preferred_locations:
            acceptable_locations.append(city)

    exp_min, exp_max = 5, 9
    exp_match = re.search(r"(\d+)\s*[–-]\s*(\d+)\s*years", text, flags=re.I)
    if exp_match:
        exp_min, exp_max = int(exp_match.group(1)), int(exp_match.group(2))

    notice_buyout_days = 30
    notice_match = re.search(r"buy out up to\s*(\d+)\s*days", text, flags=re.I)
    if notice_match:
        notice_buyout_days = int(notice_match.group(1))

    consulting_firms = DEFAULT_CONSULTING_FIRMS[:]
    consulting_match = re.search(r"consulting firms\s*\(([^)]+)\)", text, flags=re.I)
    if consulting_match:
        consulting_firms = [part.strip() for part in consulting_match.group(1).split(",") if part.strip()]

    disqualifier_domains = ["computer vision", "speech", "robotics"]
    domain_match = re.search(r"primary expertise is\s+([^.\n]+)", text, flags=re.I)
    if domain_match:
        raw = domain_match.group(1)
        disqualifier_domains = [p.strip(" .") for p in re.split(r",|or", raw) if p.strip(" .")]

    return {
        "raw_text": text,
        "location_line": location_line,
        "preferred_locations": sorted(set(preferred_locations)),
        "acceptable_locations": sorted(set(acceptable_locations)),
        "notice_buyout_days": notice_buyout_days,
        "experience_min": exp_min,
        "experience_max": exp_max,
        "consulting_firms": consulting_firms,
        "disqualifier_domains": disqualifier_domains,
    }

