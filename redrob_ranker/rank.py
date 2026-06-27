from __future__ import annotations

import argparse
import csv
import heapq
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from redrob_ranker.career_scorer import score_career, score_experience
from redrob_ranker.education_scorer import score_education
from redrob_ranker.filters import apply_filters
from redrob_ranker.jd_parser import parse_job_description
from redrob_ranker.location_scorer import score_location
from redrob_ranker.reasoning import make_reasoning
from redrob_ranker.signals_scorer import score_signals
from redrob_ranker.skill_scorer import score_skills
from redrob_ranker.utils import compute_prepass, iter_candidates, safe_get


REQUIRED_TOP_LEVEL = ("candidate_id", "profile", "career_history", "education", "skills", "redrob_signals")


def _candidate_num(candidate_id: str) -> int:
    try:
        return int(candidate_id.split("_")[-1])
    except (ValueError, AttributeError):
        return 10**12


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    for field in REQUIRED_TOP_LEVEL:
        if field not in candidate:
            raise ValueError(f"missing required field: {field}")
    if not isinstance(candidate.get("career_history"), list):
        raise ValueError("career_history must be a list")
    if not isinstance(candidate.get("skills"), list):
        raise ValueError("skills must be a list")


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
            "score": 0.0100,
            "candidate": candidate,
            "features": {"filter": filter_result},
            "reasoning": f"Filtered: {filter_result['reason']}",
        }

    skill = score_skills(candidate)
    career = score_career(candidate, jd_config)
    experience = score_experience(candidate, jd_config, corpus_stats)
    location = score_location(candidate, jd_config)
    education = score_education(candidate)
    signals = score_signals(candidate, corpus_stats, completeness)

    # Career and skill evidence are primary; behavioral signals only modify availability.
    base = (
        0.30 * skill["score"]
        + 0.35 * career["score"]
        + 0.15 * experience
        + 0.10 * location["score"]
        + 0.10 * education["score"]
    ) / 100.0

    score = base * signals["multiplier"]
    honeypot_flags = filter_result.get("honeypot_flags", [])
    if len(honeypot_flags) >= 2:
        score *= 0.10

    score = round(max(0.005, min(0.9999, score)), 4)
    features = {
        "filter": filter_result,
        "skill": skill,
        "career": career,
        "experience": experience,
        "location": location,
        "education": education,
        "signals": signals,
        "base_score": base,
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
    top_n: int = 100,
) -> list[Dict[str, Any]]:
    start = time.time()
    jd_config = parse_job_description(jd_path)
    corpus_stats, completeness = compute_prepass(candidates_path)
    prepass_elapsed = time.time() - start
    print(f"Pre-pass complete: {completeness.total} candidates in {prepass_elapsed:.1f}s")

    heap: list[tuple[tuple[float, int], Dict[str, Any]]] = []
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
                "score": 0.0050,
                "candidate": candidate,
                "features": {"error": str(exc)},
                "reasoning": "Skipped: malformed record",
            }

        scored += 1
        heap_key = (record["score"], -_candidate_num(record["candidate_id"]))
        if len(heap) < top_n:
            heapq.heappush(heap, (heap_key, record))
        elif heap_key > heap[0][0]:
            heapq.heapreplace(heap, (heap_key, record))

    ranked = [record for _, record in heap]
    ranked.sort(key=lambda row: (-row["score"], row["candidate_id"]))

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

    if validator_path:
        validator = Path(validator_path)
        if validator.exists():
            subprocess.run([sys.executable, str(validator), str(out)], check=True)
        else:
            print(f"Validator not found: {validator}", file=sys.stderr)

    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the senior AI engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl, candidates.jsonl.gz, or sample JSON array.")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx/.md/.txt.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--validator", default=None, help="Optional validate_submission.py path.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of ranked rows to emit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rank_candidates(args.candidates, args.jd, args.out, args.validator, args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

