# Redrob Candidate Ranker

Deterministic CPU-only ranker for the Redrob Senior AI Engineer candidate-ranking challenge.

## Reproduce

```bash
python rank.py --candidates ./candidates.jsonl --jd ./job_description.docx --out ./submission.csv --validator ./validate_submission.py
```

Runtime on the 100,000-candidate JSONL was about 184 seconds on CPU.

## Approach

The ranker parses the JD using AI model built on feature engineering using Natural Language Processing it also computes corpus statistics and field-completeness rates in a pre-pass, then scores candidates with deterministic rule-based components:

- Skill depth for retrieval, vector search, ranking evaluation, Python production, learning-to-rank, and fine-tuning.
- Career evidence for production shipping, search/ranking/recommendation systems, product-company exposure, and startup/product ownership.
- JD-aware experience, location, notice-period, and education scoring.
- Behavioral activity/availability/engagement signals as a bounded multiplier.
- Hard filters and honeypot penalties for clear non-fit or impossible profiles.

The final CSV is sorted by score descending and candidate ID ascending for ties, with scores rounded to four decimals.

