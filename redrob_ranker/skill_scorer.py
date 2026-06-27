from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .utils import safe_float, safe_get, safe_int, title_is_engineering


SKILL_CLUSTERS = {
    "retrieval": {
        "weight": 25,
        "terms": [
            "sentence-transformers",
            "sentence transformers",
            "bge",
            "e5",
            "embedding",
            "embeddings",
            "semantic search",
            "dense retrieval",
            "rag",
            "faiss",
            "hnsw",
            "ann",
            "approximate nearest",
            "retrieval",
        ],
    },
    "vector_search": {
        "weight": 20,
        "terms": [
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "elasticsearch",
            "opensearch",
            "solr",
            "vector database",
            "vector db",
            "hybrid search",
        ],
    },
    "ranking_eval": {
        "weight": 20,
        "terms": [
            "ndcg",
            "mrr",
            "map",
            "ranking eval",
            "offline eval",
            "a/b testing",
            "ab testing",
            "experiment design",
            "relevance",
            "learning to rank",
        ],
    },
    "python_production": {
        "weight": 15,
        "terms": [
            "python",
            "fastapi",
            "flask",
            "django",
            "rest",
            "api",
            "kubernetes",
            "docker",
            "deployment",
            "production",
            "mlops",
            "airflow",
            "spark",
        ],
    },
    "learning_to_rank": {
        "weight": 10,
        "terms": ["xgboost ranker", "lambdamart", "lambda mart", "ltr", "learning-to-rank", "ranknet"],
    },
    "llm_finetuning": {
        "weight": 10,
        "terms": ["lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning", "llm"],
    },
}

AI_SKILL_TERMS = {
    "ai",
    "ml",
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "rag",
    "computer vision",
    "speech recognition",
    "tensorflow",
    "pytorch",
    "transformers",
    "langchain",
    "openai",
    "hugging face",
    "fine-tuning",
    "embeddings",
}


def _match_terms(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _assessment_for(skill_name: str, assessments: Mapping[str, Any]) -> float | None:
    skill_lower = skill_name.lower()
    for key, value in assessments.items():
        key_lower = str(key).lower()
        if key_lower == skill_lower or key_lower in skill_lower or skill_lower in key_lower:
            score = safe_float(value, -1)
            if score >= 0:
                return max(0.0, min(1.0, score / 100.0))
    return None


def _skill_depth(skill: Mapping[str, Any], assessments: Mapping[str, Any]) -> float:
    proficiency = str(skill.get("proficiency", "")).lower()
    proficiency_weight = {
        "beginner": 0.40,
        "intermediate": 0.70,
        "advanced": 0.90,
        "expert": 1.00,
    }.get(proficiency, 0.55)

    assessment = _assessment_for(str(skill.get("name", "")), assessments)
    if assessment is not None:
        proficiency_weight = assessment

    duration = safe_int(skill.get("duration_months"), 0)
    if duration >= 36:
        duration_weight = 1.00
    elif duration >= 18:
        duration_weight = 0.85
    elif duration >= 6:
        duration_weight = 0.60
    else:
        duration_weight = 0.30

    endorsements = safe_int(skill.get("endorsements"), 0)
    if endorsements >= 20:
        endorsement_weight = 1.00
    elif endorsements >= 6:
        endorsement_weight = 0.85
    elif endorsements >= 1:
        endorsement_weight = 0.70
    else:
        endorsement_weight = 0.50

    return (proficiency_weight + duration_weight + endorsement_weight) / 3.0


def score_skills(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    skills = safe_get(candidate, "skills", default=[]) or []
    signals = safe_get(candidate, "redrob_signals", default={}) or {}
    assessments = signals.get("skill_assessment_scores") or {}
    profile = safe_get(candidate, "profile", default={}) or {}

    cluster_scores: Dict[str, float] = {}
    matched_skills: List[str] = []
    ai_skill_count = sum(
        1
        for skill in skills
        if any(term in str(skill.get("name", "")).lower() for term in AI_SKILL_TERMS)
    )

    for cluster_name, cluster in SKILL_CLUSTERS.items():
        best_depth = 0.0
        best_skill = ""
        for skill in skills:
            name = str(skill.get("name", ""))
            lowered = name.lower()
            if _match_terms(lowered, cluster["terms"]):
                depth = _skill_depth(skill, assessments)
                if depth > best_depth:
                    best_depth = depth
                    best_skill = name
        if best_depth > 0:
            cluster_scores[cluster_name] = best_depth * float(cluster["weight"])
            if best_skill and best_skill not in matched_skills:
                matched_skills.append(best_skill)
        else:
            cluster_scores[cluster_name] = 0.0

    raw_score = sum(cluster_scores.values())

    all_skill_text = " ".join(str(s.get("name", "")) for s in skills).lower()
    if "nlp" in all_skill_text or "natural language" in all_skill_text:
        raw_score += 4.0
        if "NLP" not in matched_skills:
            matched_skills.append("NLP")

    # AI keyword floods on non-technical profiles are a known challenge trap.
    if ai_skill_count >= 10 and not title_is_engineering(profile.get("current_title")):
        raw_score *= 0.50

    score = max(0.0, min(100.0, raw_score))
    return {
        "score": score,
        "cluster_scores": cluster_scores,
        "matched_skills": matched_skills[:5],
        "ai_skill_count": ai_skill_count,
    }
