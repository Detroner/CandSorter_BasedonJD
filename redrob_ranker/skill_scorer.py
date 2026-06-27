from __future__ import annotations

from typing import Any, Dict, List, Mapping

from utils import INTEREST_ONLY_PATTERNS, safe_get, safe_int


CANONICAL_SKILLS = {
    "python": {"python", "pyspark"},
    "machine learning": {"machine learning", "ml"},
    "nlp": {"nlp", "natural language processing"},
    "retrieval": {"retrieval", "semantic search", "dense retrieval"},
    "ranking": {"ranking", "learning to rank"},
    "search": {"search", "enterprise search", "vector search"},
    "recommendation": {"recommendation", "recommender system"},
    "embeddings": {"embeddings", "embedding"},
    "vector databases": {"qdrant", "weaviate", "pinecone", "milvus", "pgvector", "vector database", "vector db"},
    "llm tooling": {"langchain", "llamaindex", "huggingface", "hugging face", "transformers", "fine-tuning llms", "lora"},
    "deep learning": {"pytorch", "tensorflow", "keras"},
    "search infrastructure": {"faiss", "elasticsearch", "elastic search", "solr", "lucene", "opensearch"},
    "backend": {"backend", "flask", "fastapi", "django", "spring boot", "microservices", "rest apis", "rest api", "grpc"},
}

SUPPORTING_INFRA_SKILLS = {"airflow", "spark", "kafka", "beam", "apache beam", "dbt", "databricks", "hadoop"}
CORE_MATCH_SKILLS = {
    "python",
    "machine learning",
    "nlp",
    "retrieval",
    "ranking",
    "search",
    "recommendation",
    "embeddings",
    "vector databases",
    "llm tooling",
    "deep learning",
    "search infrastructure",
    "backend",
}


def _normalize(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _canonical_for_skill(name: str) -> str | None:
    norm = _normalize(name)
    for canonical, aliases in CANONICAL_SKILLS.items():
        if norm == canonical or norm in aliases:
            return canonical
    return None


def score_skills(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    profile = safe_get(candidate, "profile", default={}) or {}
    summary = str(profile.get("summary", "")).lower()
    listed_skills = safe_get(candidate, "skills", default=[]) or []

    matched: List[str] = []
    strong_evidence: List[str] = []
    adjacent: List[str] = []
    supporting_only: List[str] = []

    for skill in listed_skills:
        name = str(skill.get("name", ""))
        norm_name = _normalize(name)
        if norm_name in SUPPORTING_INFRA_SKILLS:
            if norm_name not in supporting_only:
                supporting_only.append(norm_name)
            continue

        canonical = _canonical_for_skill(name)
        if not canonical or canonical not in CORE_MATCH_SKILLS:
            continue
        if canonical not in matched:
            matched.append(canonical)

        duration = safe_int(skill.get("duration_months"), 0)
        proficiency = str(skill.get("proficiency", "")).lower()
        if duration >= 12 or proficiency in {"advanced", "expert"}:
            if canonical not in strong_evidence:
                strong_evidence.append(canonical)
        else:
            if canonical not in adjacent:
                adjacent.append(canonical)

    interest_only = any(pattern in summary for pattern in INTEREST_ONLY_PATTERNS)

    score = 0.0
    if strong_evidence:
        score += min(70.0, 18.0 * len(strong_evidence))
    if adjacent:
        score += min(10.0, 4.0 * len(adjacent))
    if "python" in matched:
        score += 8.0
    if "machine learning" in strong_evidence or "nlp" in strong_evidence or "deep learning" in strong_evidence:
        score += 8.0
    if "search infrastructure" in strong_evidence or "vector databases" in strong_evidence:
        score += 6.0
    if interest_only and not strong_evidence:
        score = min(score, 20.0)
    if not strong_evidence and supporting_only:
        score = min(score, 8.0)
    if len(strong_evidence) == 1 and not adjacent and not supporting_only:
        score = min(score, 26.0)

    score = max(0.0, min(100.0, score))
    return {
        "score": score,
        "matched_skills": matched[:6],
        "strong_evidence_skills": strong_evidence[:6],
        "adjacent_skills": adjacent[:6],
        "supporting_skills": supporting_only[:6],
        "interest_only": interest_only and not strong_evidence,
    }
