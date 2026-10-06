import pandas as pd
import streamlit as st

from recommendation_engine import (
    DATA_PATH,
    load_dataset,
    prepare_dataset,
    recommend_poses,
)

st.set_page_config(page_title="Pain Query", page_icon="🔎", layout="wide")
st.title("🔎 Pain Query & Yoga Recommendations")
st.write(
    "Describe your discomfort in your own words. The results are ranked by "
    "text similarity against the posture dataset."
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
    st.warning("The dataset is empty. Add valid posture records before searching.")
    st.stop()

try:
    _, _, invalid_rows = prepare_dataset(dataset)
except ValueError as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()
if invalid_rows:
    st.warning(f"{invalid_rows} row(s) without searchable text will be skipped.")

with st.form("pain_query_form"):
    query = st.text_area(
        "Pain or discomfort description",
        placeholder="Example: lower back feels stiff after sitting for a long time",
        height=120,
    )
    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=min(10, len(dataset)),
        value=min(5, len(dataset)),
    )
    submitted = st.form_submit_button(
        "Find recommendations", type="primary", use_container_width=True
    )

if submitted:
    if not query.strip():
        st.warning("Enter a pain or discomfort description before searching.")
        st.stop()

    try:
        recommendations, semantic_error, skipped_rows = recommend_poses(
            query, dataset, top_n=top_n, use_semantic=True
        )
    except ValueError as exc:
        st.error(f"Search could not be completed: {exc}")
        st.stop()
    except (OSError, RuntimeError) as exc:
        st.error(f"Search could not be completed: {exc}")
        st.stop()

    if semantic_error:
        st.warning(
            "Sentence Transformer is unavailable; showing TF-IDF-only results. "
            f"Reason: {semantic_error}"
        )
    elif recommendations["scoring_method"].eq("Hybrid TF-IDF + semantic").any():
        st.success("Hybrid ranking is active: 40% TF-IDF and 60% semantic similarity.")
    if skipped_rows:
        st.info(f"Skipped {skipped_rows} dataset row(s) with no searchable text.")

    st.divider()
    st.subheader("Ranked posture records")
    if recommendations.empty or recommendations["final_score"].max() <= 0:
        st.info(
            "No text match was found. Try including a body area or a keyword "
            "that appears in the dataset."
        )
    else:
        chart = recommendations.set_index(
            recommendations.get(
                "pose_name", pd.Series("Yoga posture", index=recommendations.index)
            ).astype(str)
        )
        st.caption("Final score is a text-similarity score, not a measure of medical confidence.")
        st.bar_chart(chart[["final_score"]], y="final_score")

        if recommendations["semantic_score"].notna().any():
            score_chart = chart[["tfidf_score", "semantic_score", "final_score"]]
            st.subheader("Score comparison for these results")
            st.bar_chart(score_chart)

        for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
            pose_name = row.get("pose_name", "Yoga posture")
            if pd.isna(pose_name) or not str(pose_name).strip():
                pose_name = "Yoga posture"
            with st.container(border=True):
                name_column, score_column = st.columns([3, 1])
                with name_column:
                    st.subheader(f"{rank}. {pose_name}")
                    sanskrit = row.get("sanskrit_name")
                    if pd.notna(sanskrit) and str(sanskrit).strip():
                        st.caption(f"Sanskrit: {sanskrit}")
                with score_column:
                    st.metric("Final match score", f"{row['final_score']:.1%}")

                score_columns = st.columns(3)
                score_columns[0].metric("TF-IDF", f"{row['tfidf_score']:.1%}")
                semantic = row.get("semantic_score")
                score_columns[1].metric(
                    "Semantic",
                    f"{semantic:.1%}" if pd.notna(semantic) else "Unavailable",
                )
                score_columns[2].metric(
                    "Ranking method", str(row.get("scoring_method", "TF-IDF"))
                )

                details = (
                    ("pain_area", "Pain area"),
                    ("therapeutic_benefits", "Dataset benefit note"),
                    ("instructions", "Instructions"),
                    ("difficulty", "Difficulty"),
                )
                for column, label in details:
                    value = row.get(column)
                    if pd.notna(value) and str(value).strip():
                        st.write(f"**{label}:** {value}")
                contraindications = row.get("contraindications")
                if pd.notna(contraindications) and str(contraindications).strip():
                    st.warning(f"**Contraindications / cautions:** {contraindications}")

st.divider()
st.info(
    "Match Score represents text/semantic similarity. It is NOT a medical "
    "probability, diagnosis, or guarantee that a pose is suitable."
)
st.warning(
    "Use recommendations for information only. Check each posture's cautions, "
    "avoid forcing movement, stop if pain increases, and consult a qualified "
    "healthcare professional when appropriate."
)
