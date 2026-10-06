import pandas as pd
import streamlit as st

from recommendation_engine import DATA_PATH, load_dataset

st.set_page_config(page_title="Posture Repository", page_icon="🧘", layout="wide")
st.title("🧘 Posture Repository")
st.write("Search and explore the posture records included in the project dataset.")

try:
    dataset = load_dataset()
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Add the approved CSV and reload.")
    st.stop()
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
    st.error(f"The dataset could not be read: {exc}")
    st.stop()

if dataset.empty:
    st.warning("The dataset is empty.")
    st.stop()

metrics = st.columns(3)
metrics[0].metric("Total records", len(dataset))
if "difficulty" in dataset.columns:
    metrics[1].metric(
        "Difficulty levels", dataset["difficulty"].nunique(dropna=True)
    )
if "pain_area" in dataset.columns:
    pain_areas = {
        label.strip()
        for value in dataset["pain_area"].dropna()
        for label in str(value).split(";")
        if label.strip()
    }
    metrics[2].metric("Pain-area labels", len(pain_areas))

st.divider()
filtered = dataset.copy()
search_columns = [
    column
    for column in (
        "pose_name",
        "sanskrit_name",
        "pain_area",
        "pain_keywords",
        "therapeutic_benefits",
    )
    if column in filtered.columns
]
search = st.text_input(
    "Search by pose, pain area, benefit, or keyword",
    placeholder="Example: back pain, Balasana",
)
if search.strip() and search_columns:
    matches = pd.Series(False, index=filtered.index)
    for column in search_columns:
        matches |= filtered[column].fillna("").astype(str).str.contains(
            search.strip(), case=False, regex=False
        )
    filtered = filtered[matches]

filter_columns = st.columns(2)
if "pain_area" in filtered.columns:
    area_options = sorted(
        {
            label.strip()
            for value in filtered["pain_area"].dropna()
            for label in str(value).split(";")
            if label.strip()
        }
    )
    selected_area = filter_columns[0].selectbox(
        "Filter by pain area", ["All"] + area_options
    )
    if selected_area != "All":
        filtered = filtered[
            filtered["pain_area"].fillna("").astype(str).map(
                lambda value: selected_area in [part.strip() for part in value.split(";")]
            )
        ]
if "difficulty" in filtered.columns:
    difficulty_options = sorted(
        filtered["difficulty"].dropna().astype(str).unique().tolist()
    )
    selected_difficulty = filter_columns[1].selectbox(
        "Filter by difficulty", ["All"] + difficulty_options
    )
    if selected_difficulty != "All":
        filtered = filtered[
            filtered["difficulty"].astype(str).eq(selected_difficulty)
        ]

st.write(f"Showing **{len(filtered)}** of **{len(dataset)}** records.")
st.dataframe(filtered, use_container_width=True, hide_index=True)

if not filtered.empty and "pose_name" in filtered.columns:
    pose_names = filtered["pose_name"].dropna().astype(str).unique().tolist()
    if pose_names:
        st.subheader("Pose details")
        selected_pose = st.selectbox("Select a posture", pose_names)
        pose = filtered[filtered["pose_name"].astype(str).eq(selected_pose)].iloc[0]
        st.markdown(f"### {selected_pose}")
        details = (
            ("sanskrit_name", "Sanskrit name"),
            ("pose_type", "Pose type"),
            ("pain_area", "Pain area"),
            ("pain_keywords", "Pain keywords"),
            ("therapeutic_benefits", "Dataset benefit note"),
            ("instructions", "Instructions"),
            ("difficulty", "Difficulty"),
            ("contraindications", "Contraindications / cautions"),
            ("description", "Description"),
            ("source_reference", "Source reference"),
        )
        for column, label in details:
            value = pose.get(column)
            if pd.notna(value) and str(value).strip():
                if column == "contraindications":
                    st.warning(f"**{label}:** {value}")
                else:
                    st.write(f"**{label}:** {value}")

st.divider()
if "difficulty" in dataset.columns:
    st.subheader("Records by difficulty")
    difficulty_counts = dataset["difficulty"].fillna("Unspecified").value_counts()
    st.bar_chart(difficulty_counts)
if "pain_area" in dataset.columns:
    st.subheader("Records by pain-area label")
    category_counts = (
        dataset["pain_area"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    category_counts = category_counts[category_counts.ne("")].value_counts()
    st.bar_chart(category_counts)

st.info(
    "This page displays dataset annotations as supplied. The project does not "
    "independently verify medical claims or clinical suitability."
)
