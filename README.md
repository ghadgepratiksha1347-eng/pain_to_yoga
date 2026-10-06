# Automated Pain-to-Yoga Recommendation System

An educational Streamlit prototype that ranks yoga-pose dataset records against
a free-text pain or discomfort query. It uses TF-IDF with unigrams and bigrams,
and optionally combines those scores with Sentence Transformer embeddings.

> **Safety:** This project is not a medical diagnosis or treatment system.
> Recommendations are informational text matches, not proof that a pose is
> safe, effective, or suitable. Review all cautions and consult a qualified
> healthcare professional when appropriate.

## 1. Project overview

People can describe a similar discomfort using different terms. This project
offers an interactive interface to search the included posture annotations and
see a ranked list with instructions and contraindications.

## 2. Problem statement

Manually finding relevant posture information in a list can be difficult for
users who do not know pose names. The prototype demonstrates how a natural
language query can retrieve textually similar records.

## 3. Objective

- Accept a free-text pain or discomfort description.
- Rank dataset records using text similarity.
- Present posture details, dataset benefit notes, instructions, difficulty,
  and contraindications.
- Provide transparent category-based retrieval checks without claiming clinical
  or real-world accuracy.

## 4. Key features

- Multi-page Streamlit UI: home, query, repository, model details, evaluation.
- TF-IDF vectorization, unigrams/bigrams, cosine similarity, ranking, and Top-N.
- Optional `all-MiniLM-L6-v2` semantic embeddings with caching.
- TF-IDF-only fallback if Sentence Transformers or model loading is unavailable.
- Hybrid ranking, defaulting to 40% TF-IDF and 60% semantic similarity.
- Dataset filters, pose details, dataset charts, score charts, and evaluation.
- Visible safety information and contraindications.

## 5. AI/NLP approach

### TF-IDF

Scikit-learn converts the query and posture text into sparse weighted vectors.
English stop words are excluded, sublinear term frequency is enabled, and the
vocabulary uses single words and adjacent pairs.

### N-grams

`ngram_range=(1, 2)` includes unigrams and bigrams. This helps retain short
phrases such as adjacent pain-area terms; it does not provide deep language
understanding by itself.

### Cosine similarity

Cosine similarity compares the direction of query and record vectors. The
result is a text-similarity score, not a probability or medical confidence.

### Semantic embeddings and Sentence Transformers

When the optional dependency and model are available, the application loads
`sentence-transformers/all-MiniLM-L6-v2` once and caches the model and the
dataset embeddings. The first use may download model files and therefore
requires network access. Model-loading/encoding errors are shown in the UI;
the search remains usable with TF-IDF.

### Hybrid recommendation and ranking

The final score is:

```text
0.40 * TF-IDF cosine similarity + 0.60 * semantic cosine similarity
```

The weights are constants in `recommendation_engine.py`. If semantic scoring
is unavailable, the final score is the TF-IDF score. Results are sorted by
final score and then TF-IDF score, descending.

## 6. Model evaluation

The **Retrieval Evaluation** page evaluates five fixed example queries only
when their expected categories are present in the CSV:

- lower back pain → `Back`
- neck stiffness → `Neck/Shoulder`
- shoulder discomfort → `Neck/Shoulder`
- stress/relaxation → `General mobility/relaxation`
- tight hips → `Hip`

Evaluation uses the configured production ranker (hybrid if the embedding
model loads, otherwise TF-IDF). Relevance is defined transparently as whether
the returned record's `pain_area` includes the expected category.

The CSV does not contain independently judged relevance labels or clinical
outcomes. Consequently, the reported values are small internal retrieval
checks—not real-world accuracy, clinical validation, or evidence of treatment
effectiveness. Dataset categories are coarse proxy labels.

### Precision@K

Fraction of the first K retrieved records whose `pain_area` contains the
expected category.

### Recall@K

Fraction of all dataset records carrying the expected category that appear in
the first K retrieved records.

### F1@K

Harmonic mean of precision and recall for one query.

### Hit Rate@K

Whether at least one relevant category-labelled record appears in the first K.

### Mean Reciprocal Rank (MRR)

Mean of the reciprocal rank of the first relevant result for each test query.

## 7. Charts

Charts are derived from the current CSV or actual query/evaluation output:
recommendation scores, TF-IDF/semantic/final score comparison (when semantic
scores exist), posture counts by difficulty and pain-area label, and aggregate
retrieval metrics. They are not fabricated benchmark data.

## 8. Dataset and limitations

The app reads `data/approved_dataset.csv`; the existing column names are
preserved. The project CSV has 326 records and the following fields:

`pose_id`, `pose_name`, `sanskrit_name`, `pose_type`, `pain_area`,
`pain_keywords`, `therapeutic_benefits`, `contraindications`, `instructions`,
`difficulty`, `description`, `source_reference`.

The dataset is a prototype: its source field describes a 215-pose reference
combined with curated prototype variants, and the benefit/instruction text is
not independently clinically validated. A row or similarity score must not be
interpreted as authoritative health guidance.

## 9. Project structure

```text
Pain_to_Yoga_Project/
├── app.py
├── recommendation_engine.py
├── requirements.txt
├── README.md
├── data/
│   └── approved_dataset.csv
├── pages/
│   ├── 1_Query_Input.py
│   ├── 2_Posture_Repository.py
│   ├── 3_Model_Details.py
│   └── 4_Model_Evaluation.py
└── tests/
    └── test_recommendation_engine.py
```

## 10. Technologies

- Python 3.10+
- Streamlit
- pandas and NumPy
- scikit-learn
- Sentence Transformers (optional at runtime; declared for semantic search)
- Streamlit built-in charts

## 11. Installation

Create and activate a virtual environment from the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sentence Transformers may install a large machine-learning runtime. On the
first semantic search, the selected model may also need to be downloaded.
If it cannot load, TF-IDF search still works.

## 12. Run the application

```bash
streamlit run app.py
```

Example query: `lower back feels stiff after sitting for a long time`.
The app displays ranked records with TF-IDF, semantic (when available), and
final match scores, followed by available pose details and cautions. Scores
are similarity values, not medical probabilities.

## 13. Run tests

```bash
python -m unittest discover -s tests -v
```

## 14. GitHub usage

Initialize or clone the repository, review the dataset's licensing/source
requirements before publishing, then add and push the project:

```bash
git add app.py recommendation_engine.py pages data/approved_dataset.csv tests requirements.txt README.md .gitignore
git commit -m "Prepare pain-to-yoga recommendation project"
git push
```

`.gitignore` excludes local virtual environments, Python cache files, editor
settings, and Streamlit secrets. The required CSV remains included.

## 15. Streamlit deployment

1. Push the project to GitHub with `app.py`, `pages/`, `data/`, and
   `requirements.txt`.
2. In Streamlit Community Cloud, select the repository and branch.
3. Set the main file to `app.py` and deploy.
4. Allow time/network access for the first Sentence Transformer model download,
   or use the working TF-IDF fallback when the optional model cannot load.

## 16. Future improvements

- Replace prototype annotations with reviewed, properly licensed and
  clinically vetted sources.
- Collect independent relevance judgments and evaluate on a held-out set.
- Add controlled feedback from qualified yoga/health professionals.
- Compare retrieval settings using reproducible offline benchmarks.