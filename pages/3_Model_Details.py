import importlib.util

import streamlit as st

from recommendation_engine import (
    SEMANTIC_MODEL_NAME,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
)

st.set_page_config(page_title="Model Details", page_icon="🤖", layout="wide")
st.title("🤖 Model Details")
st.write(
    "This page describes the retrieval pipeline implemented in the current project."
)

st.subheader("Implemented workflow")
st.code(
    """
User query
    ↓
Text preparation from available dataset fields
    ↓
TF-IDF (unigrams + bigrams)
    ↓
TF-IDF cosine similarity ─────────────────┐
    ↓                                      │
Optional Sentence Transformer embeddings   │
    ↓                                      │
Semantic cosine similarity ───────────────┤
    ↓                                      ↓
Hybrid score: 40% TF-IDF + 60% semantic (when available)
Otherwise: TF-IDF-only fallback
    ↓
Ranked Top-N results with pose details and cautions
""",
    language="text",
)

st.subheader("Current optional semantic dependency")
if importlib.util.find_spec("sentence_transformers") is None:
    st.info(
        "Sentence Transformers is not installed in the current environment. "
        "The query page remains functional using TF-IDF fallback. Install "
        "`requirements.txt` to enable semantic scoring."
    )
else:
    st.success(
        f"`sentence-transformers` is installed. The query page loads "
        f"`{SEMANTIC_MODEL_NAME}` on first semantic search and caches it."
    )

st.subheader("Key concepts")
concepts = {
    "NLP and text preparation": (
        "Available pose names, Sanskrit names, pain areas, keywords, benefit "
        "notes, and descriptions are combined into documents for retrieval."
    ),
    "TF-IDF": (
        "Term Frequency-Inverse Document Frequency gives higher weight to "
        "terms that distinguish documents. Scikit-learn uses English stop-word "
        "filtering and sublinear term frequency here."
    ),
    "N-grams": (
        "The vectorizer uses unigrams and bigrams (ngram_range=(1, 2)) so "
        "single words and adjacent word pairs can contribute."
    ),
    "Cosine similarity": (
        "Measures the angle-based similarity between the query vector and each "
        "posture document vector. This is text similarity, not clinical confidence."
    ),
    "Sentence embeddings": (
        f"When installed and loadable, `{SEMANTIC_MODEL_NAME}` maps text into "
        "dense vectors so semantically related wording may match even when "
        "the exact words differ."
    ),
    "Hybrid ranking": (
        f"The final score is {TFIDF_WEIGHT:.0%} TF-IDF similarity plus "
        f"{SEMANTIC_WEIGHT:.0%} semantic similarity. The weights are constants "
        "in recommendation_engine.py. Semantic load/encoding failures are "
        "reported and use TF-IDF-only ranking."
    ),
    "Model evaluation": (
        "The evaluation page measures retrieval agreement with predefined "
        "pain-area labels that exist in the CSV. It is an internal dataset "
        "check, not clinical or real-world validation."
    ),
}
for title, explanation in concepts.items():
    with st.expander(title):
        st.write(explanation)

st.subheader("Technology stack")
st.table(
    {
        "Component": [
            "Language",
            "Web UI",
            "Data",
            "Numerical operations",
            "Text retrieval",
            "Optional semantic model",
            "Charts",
        ],
        "Technology": [
            "Python",
            "Streamlit",
            "pandas",
            "NumPy",
            "scikit-learn",
            "Sentence Transformers",
            "Streamlit built-in charts",
        ],
    }
)
st.warning(
    "The system does not diagnose, prescribe treatment, or guarantee that any "
    "posture is safe or effective for a particular person."
)
