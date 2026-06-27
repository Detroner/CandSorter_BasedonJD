from __future__ import annotations

from typing import Any, Dict, Mapping

from utils import safe_bool, safe_get, safe_int


PREFERRED_INDIAN_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore", "bengaluru"}


def score_location(candidate: Mapping[str, Any], jd_config: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    location = str(profile.get("location", "")).lower()
    country = str(profile.get("country", "")).lower()
    raw_notice = signals.get("notice_period_days")
    notice_days = safe_int(raw_notice, 180)
    if notice_days < 0:
        notice_days = 180
    willing = safe_bool(signals.get("willing_to_relocate"), False)

    score = 35.0
    location_note = "location fit unclear"

    if country == "india":
        score = 80.0
        location_note = "based in India"
        if any(city in location for city in PREFERRED_INDIAN_CITIES):
            score = 92.0
            location_note = "preferred India location"
    elif willing:
        score = 62.0
        location_note = "outside India but willing to relocate"
    else:
        score = 5.0
        location_note = "outside India without relocation"

    max_notice = int(jd_config.get("max_notice_days", 30))
    if notice_days <= max_notice:
        score += 6.0
    elif notice_days <= 60:
        score -= 4.0
    else:
        score -= 18.0

    return {
        "score": max(0.0, min(100.0, score)),
        "notice_days": notice_days,
        "location_note": location_note,
    }
