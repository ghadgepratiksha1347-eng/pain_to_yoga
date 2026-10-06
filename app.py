import pandas as pd
import streamlit as st

from recommendation_engine import DATA_PATH, load_dataset

st.set_page_config(
    page_title="Pain-to-Yoga Recommendation System",
    page_icon="🧘",
    layout="wide",
)

st.title("🧘 Automated Pain-to-Yoga Recommendation System")
st.subheader("An NLP-based posture retrieval prototype")
st.write(
    "Explore a dataset of yoga postures and retrieve records that textually "
    "match a pain or discomfort description. The system combines TF-IDF "
    "retrieval with optional sentence embeddings."
)

try:
    dataset = load_dataset()
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Add the approved CSV and reload.")
    st.stop()
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
    st.error(f"The dataset could not be read: {exc}")
    st.stop()

if dataset.empty:
    st.warning("The dataset file exists but contains no records.")
else:
    metric_columns = st.columns(3)
    metric_columns[0].metric("Dataset records", len(dataset))
    if "pose_name" in dataset.columns:
        metric_columns[1].metric(
            "Unique poses", dataset["pose_name"].nunique(dropna=True)
        )
    if "pain_area" in dataset.columns:
        metric_columns[2].metric(
            "Pain-area labels",
            len(
                {
                    label.strip()
                    for value in dataset["pain_area"].dropna()
                    for label in str(value).split(";")
                    if label.strip()
                }
            ),
        )

st.divider()
st.header("Project overview")
st.write(
    """
People may describe the same discomfort in different words. This prototype
provides a searchable, ranked view of posture records using the text included
in the project dataset. It does not infer a diagnosis or establish that a pose
will treat a condition.
"""
)

left, right = st.columns(2)
with left:
    st.subheader("Objective")
    st.write(
        "Turn free-text queries into ranked dataset results while keeping pose "
        "instructions and contraindications visible."
    )
    st.subheader("Core features")
    st.markdown(
        """
        - Natural-language query and adjustable Top-N results
        - TF-IDF with unigrams and bigrams, plus cosine similarity
        - Optional sentence-embedding scores and configurable hybrid ranking
        - Dataset search, filters, and category-based retrieval evaluation
        """
    )
with right:
    st.subheader("How it works")
    st.code(
        """
User query
    ↓
Text preparation
    ↓
TF-IDF (1- and 2-grams) ── cosine similarity
    ↓
Optional sentence embeddings ── cosine similarity
    ↓
Hybrid score (when available) or TF-IDF fallback
    ↓
Ranked Top-N posture records
""",
        language="text",
    )

st.subheader("Technology stack")
st.write("Python · Streamlit · pandas · NumPy · scikit-learn · Sentence Transformers (optional)")
st.info(
    "Use the sidebar to open **Pain Query**, **Posture Repository**, "
    "**Model Details**, and **Retrieval Evaluation**."
)
st.warning(
    "Educational information only—not a medical diagnosis, treatment plan, "
    "or guarantee of suitability. Consult a qualified healthcare professional "
    "when appropriate; stop if an activity increases pain."
)
