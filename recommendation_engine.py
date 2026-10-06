"""Shared text-retrieval and evaluation helpers for the Streamlit pages."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "approved_dataset.csv"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

SEARCH_COLUMNS = (
    "pose_name",
    "sanskrit_name",
    "pain_area",
    "pain_keywords",
    "therapeutic_benefits",
    "description",
)

_EVALUATION_CASES = (
    ("Lower back pain after sitting", "lower back pain after sitting for long hours", "Back"),
    ("Neck stiffness", "neck stiffness after working at a desk", "Neck/Shoulder"),
    ("Shoulder discomfort", "shoulder discomfort and tight shoulders", "Neck/Shoulder"),
    ("Stress and relaxation", "stress, tension, and need for relaxation", "General mobility/relaxation"),
    ("Tight hips", "tight hips after sitting for a long time", "Hip"),
)


@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load the project CSV and let callers present file/format errors clearly."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    return pd.read_csv(path)


def prepare_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Build NLP documents and exclude rows that contain no searchable text."""
    if dataset.empty:
        raise ValueError("The dataset has no records.")

    available_columns = [
        column for column in SEARCH_COLUMNS if column in dataset.columns
    ]
    if not available_columns:
        raise ValueError(
            "No searchable columns were found. Add at least one of: "
            + ", ".join(SEARCH_COLUMNS)
        )

    prepared = dataset.copy()
    documents = (
        prepared[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )
    valid_rows = documents.ne("")
    invalid_count = int((~valid_rows).sum())
    prepared = prepared.loc[valid_rows].reset_index(drop=True)
    documents = documents.loc[valid_rows].tolist()
    if not documents:
        raise ValueError("No dataset rows contain searchable text.")
    return prepared, documents, invalid_count


@st.cache_resource(show_spinner=False)
def load_semantic_model() -> tuple[Any | None, str | None]:
    """Load and cache the optional embedding model, retaining load errors for UI."""
    try:
        SentenceTransformer = import_module(
            "sentence_transformers"
        ).SentenceTransformer
        return SentenceTransformer(SEMANTIC_MODEL_NAME), None
    except Exception as exc:  # Optional dependency/model download can fail in varied ways.
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner="Preparing semantic embeddings...")
def _encode_documents(documents: tuple[str, ...]) -> np.ndarray:
    model, error = load_semantic_model()
    if model is None:
        raise RuntimeError(error or "The semantic model could not be loaded.")
    return np.asarray(
        model.encode(
            list(documents),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )


def recommend_poses(
    query: str,
    dataset: pd.DataFrame,
    top_n: int = 5,
    use_semantic: bool = True,
) -> tuple[pd.DataFrame, str | None, int]:
    """Rank posture records with TF-IDF and optional semantic hybrid scoring."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a non-empty pain or discomfort description.")
    if top_n < 1:
        raise ValueError("The recommendation count must be at least one.")

    prepared, documents, invalid_count = prepare_dataset(dataset)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        document_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query.strip()])
    except ValueError as exc:
        raise ValueError(
            "The dataset does not contain enough valid text to create TF-IDF features."
        ) from exc

    tfidf_scores = cosine_similarity(query_vector, document_vectors).ravel()
    semantic_scores: np.ndarray | None = None
    semantic_error: str | None = None

    if use_semantic:
        model, semantic_error = load_semantic_model()
        if model is not None:
            try:
                document_embeddings = _encode_documents(tuple(documents))
                query_embedding = model.encode(
                    [query.strip()],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                semantic_scores = cosine_similarity(
                    np.asarray(query_embedding), document_embeddings
                ).ravel()
                semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
            except Exception as exc:  # Surface optional-model failures and keep TF-IDF usable.
                semantic_error = f"{type(exc).__name__}: {exc}"

    results = prepared.copy()
    results["tfidf_score"] = tfidf_scores
    results["semantic_score"] = (
        semantic_scores if semantic_scores is not None else np.nan
    )
    if semantic_scores is None:
        results["final_score"] = tfidf_scores
        results["scoring_method"] = "TF-IDF fallback"
    else:
        results["final_score"] = (
            TFIDF_WEIGHT * tfidf_scores + SEMANTIC_WEIGHT * semantic_scores
        )
        results["scoring_method"] = "Hybrid TF-IDF + semantic"

    results = results.sort_values(
        ["final_score", "tfidf_score"], ascending=False, kind="mergesort"
    )
    return results.head(min(top_n, len(results))).reset_index(drop=True), semantic_error, invalid_count


def pain_area_labels(value: Any) -> set[str]:
    """Return normalized pain-area labels from the dataset's semicolon format."""
    if pd.isna(value):
        return set()
    return {label.strip().casefold() for label in str(value).split(";") if label.strip()}


def available_evaluation_cases(dataset: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Select only predefined test cases whose labels exist in this dataset."""
    if "pain_area" not in dataset.columns:
        return []
    dataset_labels = set().union(*(pain_area_labels(value) for value in dataset["pain_area"]))
    return [
        case for case in _EVALUATION_CASES if case[2].casefold() in dataset_labels
    ]


def evaluate_recommendations(
    dataset: pd.DataFrame, top_k: int = 5, use_semantic: bool = True
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Evaluate the configured ranker against transparent pain_area labels."""
    if top_k < 1:
        raise ValueError("K must be at least one.")
    if "pain_area" not in dataset.columns:
        raise ValueError("The dataset needs a pain_area column for this evaluation.")

    cases = available_evaluation_cases(dataset)
    if not cases:
        raise ValueError("No predefined evaluation labels exist in this dataset.")

    rows: list[dict[str, Any]] = []
    semantic_errors: list[str] = []
    for title, query, expected_area in cases:
        recommendations, semantic_error, _ = recommend_poses(
            query, dataset, top_n=top_k, use_semantic=use_semantic
        )
        if semantic_error and semantic_error not in semantic_errors:
            semantic_errors.append(semantic_error)
        expected = expected_area.casefold()
        relevant = recommendations["pain_area"].map(
            lambda value: expected in pain_area_labels(value)
        )
        relevant_count = int(relevant.sum())
        total_relevant = sum(
            expected in pain_area_labels(value) for value in dataset["pain_area"]
        )
        precision = relevant_count / len(recommendations) if len(recommendations) else 0.0
        recall = relevant_count / total_relevant if total_relevant else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        relevant_ranks = np.flatnonzero(relevant.to_numpy())
        reciprocal_rank = (
            1.0 / (int(relevant_ranks[0]) + 1) if len(relevant_ranks) else 0.0
        )
        first_row = recommendations.iloc[0] if not recommendations.empty else None
        rows.append(
            {
                "Test Query": title,
                "Expected Category": expected_area,
                "Top Recommendation": (
                    first_row.get("pose_name", "Yoga posture")
                    if first_row is not None
                    else "No result"
                ),
                "Ranking Method": (
                    first_row.get("scoring_method", "Unavailable")
                    if first_row is not None
                    else "Unavailable"
                ),
                "Relevant?": bool(relevant.iloc[0]) if len(relevant) else False,
                f"Precision@{top_k}": precision,
                f"Recall@{top_k}": recall,
                f"F1@{top_k}": f1,
                f"Hit Rate@{top_k}": float(relevant_count > 0),
                "MRR": reciprocal_rank,
            }
        )

    table = pd.DataFrame(rows)
    metric_columns = [
        f"Precision@{top_k}",
        f"Recall@{top_k}",
        f"F1@{top_k}",
        f"Hit Rate@{top_k}",
        "MRR",
    ]
    metrics = {column: float(table[column].mean()) for column in metric_columns}
    return table, metrics, semantic_errors
