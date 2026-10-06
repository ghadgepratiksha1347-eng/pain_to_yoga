import pandas as pd
import streamlit as st

from recommendation_engine import (
    DATA_PATH,
    available_evaluation_cases,
    evaluate_recommendations,
    load_dataset,
)

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
st.title("📈 Retrieval Evaluation")
st.write(
    "A small, transparent information-retrieval check using expected pain-area "
    "labels already present in the dataset."
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
    st.warning("The dataset is empty, so evaluation cannot run.")
    st.stop()

if "pain_area" not in dataset.columns:
    st.error("The dataset needs a `pain_area` column for category-based evaluation.")
    st.stop()

cases = available_evaluation_cases(dataset)
if not cases:
    st.info("None of the predefined test categories exist in this dataset.")
    st.stop()

max_k = min(10, len(dataset))
top_k = st.slider("Evaluate the top K recommendations", 1, max_k, min(5, max_k))

try:
    results, metrics, semantic_errors = evaluate_recommendations(
        dataset, top_k=top_k, use_semantic=True
    )
except (ValueError, KeyError) as exc:
    st.error(f"Evaluation could not be completed: {exc}")
    st.stop()

st.warning(
    "These are category-label retrieval checks, not clinical validation or "
    "real-world accuracy. `pain_area` is a coarse proxy label; the dataset has "
    "no independently reviewed relevance judgments."
)
st.info(
    "The evaluation uses the configured ranker: hybrid similarity when the "
    "semantic model loads, otherwise TF-IDF bigram ranking. Metrics describe "
    "agreement with the expected dataset category only."
)
if semantic_errors:
    st.warning(
        "Semantic model unavailable; evaluation ran with the TF-IDF fallback. "
        f"Reason: {semantic_errors[0]}"
    )

columns = st.columns(5)
for column, (name, value) in zip(columns, metrics.items()):
    column.metric(name, f"{value:.3f}")

st.subheader("Per-query results")
st.dataframe(
    results,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Aggregate metrics (mean across test queries)")
chart_data = pd.DataFrame(
    {"Metric": list(metrics), "Score": list(metrics.values())}
).set_index("Metric")
st.bar_chart(chart_data, y="Score")

with st.expander("What do these metrics mean?"):
    st.markdown(
        f"""
        - **Precision@{top_k}:** fraction of the first {top_k} results whose
          `pain_area` contains the expected category.
        - **Recall@{top_k}:** fraction of all rows with that category retrieved
          in the first {top_k}.
        - **F1@{top_k}:** harmonic mean of precision and recall for that query.
        - **Hit Rate@{top_k}:** whether at least one relevant row appears in the
          first {top_k}.
        - **MRR:** mean reciprocal rank of the first relevant result across
          the test queries.
        """
    )
