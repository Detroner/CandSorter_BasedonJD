from __future__ import annotations

from typing import Any, Dict, Mapping

from .utils import safe_bool, safe_get, safe_int


CITY_ALIASES = {
    "delhi ncr": ["delhi", "new delhi", "gurgaon", "gurugram", "faridabad", "noida"],
    "bangalore": ["bangalore", "bengaluru"],
    "bengaluru": ["bangalore", "bengaluru"],
}


def _contains_city(location: str, city: str) -> bool:
    loc = location.lower()
    city_lower = city.lower()
    if city_lower in loc:
        return True
    return any(alias in loc for alias in CITY_ALIASES.get(city_lower, []))


def score_location(candidate: Mapping[str, Any], jd_config: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    location = str(profile.get("location", ""))
    country = str(profile.get("country", "")).lower()
    willing = safe_bool(signals.get("willing_to_relocate"), False)
    preferred = jd_config.get("preferred_locations", []) or []
    acceptable = jd_config.get("acceptable_locations", []) or []

    if any(_contains_city(location, city) for city in preferred):
        loc_score = 100.0
        location_note = "preferred location"
    elif any(_contains_city(location, city) for city in acceptable):
        loc_score = 82.0
        location_note = "JD-welcome location"
    elif country == "india" and willing:
        loc_score = 65.0
        location_note = "India-based and willing to relocate"
    elif country == "india":
        loc_score = 42.0
        location_note = "India-based but relocation unclear"
    elif willing:
        loc_score = 18.0
        location_note = "outside India but willing to relocate"
    else:
        loc_score = 5.0
        location_note = "outside India"

    notice = safe_int(signals.get("notice_period_days"), 180)
    buyout = safe_int(jd_config.get("notice_buyout_days"), 30)
    if notice <= buyout:
        notice_score = 100.0
        notice_note = "within buyout window"
    elif notice <= 60:
        notice_score = 72.0
        notice_note = "moderate notice concern"
    elif notice <= 90:
        notice_score = 42.0
        notice_note = "long notice concern"
    else:
        notice_score = 18.0
        notice_note = "very long notice"

    return {
        "score": 0.70 * loc_score + 0.30 * notice_score,
        "location_score": loc_score,
        "notice_score": notice_score,
        "location_note": location_note,
        "notice_note": notice_note,
        "notice_days": notice,
    }

