from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from career_scorer import score_career, score_experience
from education_scorer import score_education
from filters import apply_filters
from jd_parser import parse_job_description
from location_scorer import score_location
from reasoning import make_reasoning
from signals_scorer import score_signals
from skill_scorer import score_skills
from utils import compute_prepass, valid_candidate_id, iter_candidates


REQUIRED_TOP_LEVEL = ("candidate_id", "profile", "career_history", "education", "skills", "redrob_signals")

HARD_FILTER_SCORE = 0.0100
MALFORMED_SCORE = 0.0050
HONEYPOT_CEILING = 0.0099

COMPONENT_WEIGHTS = {
    "skill": 0.28,
    "career": 0.28,
    "experience": 0.12,
    "location": 0.10,
    "education": 0.12,
    "signals": 0.10,
}


def _compute_base_score(
    skill: Mapping[str, Any],
    career: Mapping[str, Any],
    experience: float,
    location: Mapping[str, Any],
    education: Mapping[str, Any],
    signals: Mapping[str, Any],
    completeness,
) -> float:
    scores = {
        "skill": skill["score"],
        "career": career["score"],
        "experience": experience,
        "location": location["score"],
        "education": education["score"],
        "signals": signals["score"],
    }
    active = {
        "skill": True,
        "career": True,
        "experience": True,
        "location": completeness.component_active("location"),
        "education": education.get("active", True) and completeness.component_active("education"),
        "signals": True,
    }
    active_weight = sum(COMPONENT_WEIGHTS[key] for key, on in active.items() if on)
    if active_weight <= 0:
        return 0.0
    total = sum(COMPONENT_WEIGHTS[key] / active_weight * scores[key] for key, on in active.items() if on)
    return total / 100.0


def _candidate_num(candidate_id: str) -> int:
    if valid_candidate_id(candidate_id):
        return int(candidate_id.split("_")[-1])
    return 10**12


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    if not valid_candidate_id(candidate.get("candidate_id")):
        raise ValueError("candidate_id must match CAND_0000000 format")
    for field in REQUIRED_TOP_LEVEL:
        if field not in candidate:
            raise ValueError(f"missing required field: {field}")
    if not isinstance(candidate.get("profile"), Mapping):
        raise ValueError("profile must be a mapping")
    if not isinstance(candidate.get("career_history"), list):
        raise ValueError("career_history must be a list")
    if not isinstance(candidate.get("education"), list):
        raise ValueError("education must be a list")
    if not isinstance(candidate.get("skills"), list):
        raise ValueError("skills must be a list")
    if not isinstance(candidate.get("redrob_signals"), Mapping):
        raise ValueError("redrob_signals must be a mapping")


def score_candidate(
    candidate: Mapping[str, Any],
    jd_config: Mapping[str, Any],
    corpus_stats: Mapping[str, Mapping[str, float]],
    completeness,
) -> Dict[str, Any]:
    _validate_candidate(candidate)
    candidate_id = str(candidate.get("candidate_id", ""))

    filter_result = apply_filters(candidate, jd_config)
    if not filter_result["passed"]:
        return {
            "candidate_id": candidate_id,
            "score": HARD_FILTER_SCORE,
            "candidate": candidate,
            "features": {"filter": filter_result, "final_score": HARD_FILTER_SCORE, "location": {}, "signals": {}, "career": {}, "skill": {}},
            "reasoning": f"Filtered: {filter_result['reason']}",
        }

    skill = score_skills(candidate)
    career = score_career(candidate, jd_config)
    experience = score_experience(candidate, jd_config, corpus_stats)
    location = score_location(candidate, jd_config)
    education = score_education(candidate)
    signals = score_signals(candidate, corpus_stats, completeness)

    base = _compute_base_score(skill, career, experience, location, education, signals, completeness)
    score = base
    honeypot_flags = filter_result.get("honeypot_flags", [])
    if honeypot_flags:
        score *= 0.75 ** len(honeypot_flags)
        if len(honeypot_flags) >= 2:
            score = min(score, HONEYPOT_CEILING)

    score = round(max(MALFORMED_SCORE, min(0.9999, score)), 4)
    features = {
        "filter": filter_result,
        "skill": skill,
        "career": career,
        "experience": experience,
        "location": location,
        "education": education,
        "signals": signals,
        "base_score": base,
        "final_score": score,
    }
    return {
        "candidate_id": candidate_id,
        "score": score,
        "candidate": candidate,
        "features": features,
        "reasoning": "",
    }


def rank_candidates(
    candidates_path: str | Path,
    jd_path: str | Path,
    out_path: str | Path,
    validator_path: str | Path | None = None,
    top_n: int = 100_000,  # changed: default covers full 100k dataset
) -> list[Dict[str, Any]]:
    del validator_path
    start = time.time()
    jd_config = parse_job_description(jd_path)
    corpus_stats, completeness = compute_prepass(candidates_path)
    prepass_elapsed = time.time() - start
    print(f"Pre-pass complete: {completeness.total} candidates in {prepass_elapsed:.1f}s")

    # changed: collect all records in a plain list; no heap cap
    records: list[Dict[str, Any]] = []
    malformed = 0
    scored = 0

    for candidate in iter_candidates(candidates_path):
        candidate_id = str(candidate.get("candidate_id", ""))
        try:
            record = score_candidate(candidate, jd_config, corpus_stats, completeness)
        except Exception as exc:
            malformed += 1
            record = {
                "candidate_id": candidate_id or f"MALFORMED_{malformed:07d}",
                "score": MALFORMED_SCORE,
                "candidate": candidate,
                "features": {"error": str(exc), "final_score": MALFORMED_SCORE},
                "reasoning": "Skipped: malformed record",
            }

        scored += 1
        records.append(record)

    # Sort all records descending by score, then ascending by candidate_id for ties
    records.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    # Apply top_n slice after sorting (keeps CLI --top-n flag useful for testing)
    ranked = records[:top_n]

    for idx, record in enumerate(ranked, start=1):
        record["rank"] = idx
        if not record.get("reasoning"):
            record["reasoning"] = make_reasoning(record["candidate"], record["features"], idx)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for record in ranked:
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "rank": record["rank"],
                    "score": f"{record['score']:.4f}",
                    "reasoning": record["reasoning"],
                }
            )

    elapsed = time.time() - start
    print(f"Wrote {len(ranked)} rows to {out} in {elapsed:.1f}s; malformed={malformed}; scored={scored}")
    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the senior AI engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl, candidates.jsonl.gz, or sample JSON array.")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx/.md/.txt.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--validator", default=None, help="Optional validate_submission.py path.")
    parser.add_argument("--top-n", type=int, default=100_000, help="Number of ranked rows to emit (default: 100000 = full dataset).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rank_candidates(args.candidates, args.jd, args.out, args.validator, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())