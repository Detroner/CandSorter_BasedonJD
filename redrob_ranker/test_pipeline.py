from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from .career_scorer import score_career
from .filters import honeypot_flags
from .rank import rank_candidates
from .reasoning import make_reasoning
from .utils import DataCompletenessReport


JD_TEXT = """
Job Description: Senior AI Engineer
Location: Pune/Noida, India | Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply.
Experience Required: 5-9 years
Notice period: We can buy out up to 30 days.
People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) are not preferred.
People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure are not a fit.
"""


def candidate(candidate_id: str = "CAND_0000001", scorey: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "ML Engineer building retrieval systems",
            "summary": "Built and deployed semantic search and ranking systems for real users.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.5,
            "current_title": "ML Engineer",
            "current_company": "ProductCo",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": "ML Engineer",
                "start_date": "2023-01-01",
                "end_date": None,
                "duration_months": 41,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Deployed ranking system and shipped recommendation engine for 500k users.",
            },
            {
                "company": "MarketplaceCo",
                "title": "Software Engineer",
                "start_date": "2020-01-01",
                "end_date": "2022-12-31",
                "duration_months": 36,
                "is_current": False,
                "industry": "Internet",
                "company_size": "201-500",
                "description": "Designed retrieval pipeline and production search relevance experiments.",
            },
        ],
        "education": [
            {
                "institution": "IIT Test",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 25, "duration_months": 70},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 14, "duration_months": 30},
            {"name": "NDCG", "proficiency": "advanced", "endorsements": 8, "duration_months": 24},
        ],
        "redrob_signals": {
            "profile_completeness_score": 95,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 12,
            "skill_assessment_scores": {"Python": 82, "FAISS": 79},
            "connection_count": 300,
            "endorsements_received": 50,
            "notice_period_days": 20,
            "expected_salary_range_inr_lpa": {"min": 35, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "search_appearance_30d": 55,
            "saved_by_recruiters_30d": 6,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


class PipelineTests(unittest.TestCase):
    def test_junk_input_scores_below_hard_filter_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jd = tmp_path / "jd.txt"
            jd.write_text(JD_TEXT, encoding="utf-8")
            candidates = tmp_path / "candidates.json"
            junk = {"candidate_id": "CAND_9999999", "profile": {"years_of_experience": "6.9"}, "skills": []}
            candidates.write_text(json.dumps([candidate("CAND_0000001"), junk]), encoding="utf-8")
            out = tmp_path / "out.csv"
            ranked = rank_candidates(candidates, jd, out, top_n=2)
            by_id = {row["candidate_id"]: row for row in ranked}
            self.assertEqual(by_id["CAND_9999999"]["score"], 0.005)

    def test_honeypot_detection(self) -> None:
        c = candidate()
        c["skills"].append({"name": "Qdrant", "proficiency": "expert", "endorsements": 3, "duration_months": 0})
        c["redrob_signals"]["skill_assessment_scores"]["Qdrant"] = 22
        flags = honeypot_flags(c)
        self.assertGreaterEqual(len(flags), 2)

    def test_tier5_rescue(self) -> None:
        c = candidate()
        c["skills"] = [{"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 48}]
        result = score_career(c, {"experience_min": 5, "experience_max": 9})
        self.assertGreaterEqual(result["domain_roles"], 2)
        self.assertGreater(result["score"], 60)

    def test_reasoning_diversity(self) -> None:
        base = candidate()
        features = {
            "skill": {"matched_skills": ["FAISS", "NDCG"], "score": 80},
            "career": {"domain_terms": ["retrieval", "ranking"], "production_roles": 2, "domain_roles": 2, "consulting_roles": 0, "product_roles": 2},
            "location": {"notice_days": 20, "location_note": "preferred location"},
            "signals": {"signal_note": "active recently", "active_days": 5},
        }
        openings = set()
        for i in range(20):
            c = json.loads(json.dumps(base))
            c["candidate_id"] = f"CAND_{i + 1:07d}"
            c["profile"]["current_title"] = f"ML Engineer {i}"
            c["profile"]["current_company"] = f"ProductCo{i}"
            text = make_reasoning(c, features, i + 1)
            opening = " ".join(text.split()[:10])
            self.assertNotIn(opening, openings)
            openings.add(opening)

    def test_score_monotonic_and_floating_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jd = tmp_path / "jd.txt"
            jd.write_text(JD_TEXT, encoding="utf-8")
            rows = []
            for i in range(12):
                c = candidate(f"CAND_{i + 1:07d}")
                c["profile"]["years_of_experience"] = 5 + (i % 5)
                c["redrob_signals"]["github_activity_score"] = 80 - i
                rows.append(c)
            candidates = tmp_path / "candidates.json"
            candidates.write_text(json.dumps(rows), encoding="utf-8")
            out = tmp_path / "out.csv"
            rank_candidates(candidates, jd, out, top_n=10)
            with open(out, encoding="utf-8", newline="") as f:
                scored = list(csv.DictReader(f))
            scores = [float(row["score"]) for row in scored]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for score in scores:
                self.assertEqual(score, round(score, 4))
            for prev, nxt in zip(scored, scored[1:]):
                if float(prev["score"]) == float(nxt["score"]):
                    self.assertLess(prev["candidate_id"], nxt["candidate_id"])


if __name__ == "__main__":
    unittest.main()
