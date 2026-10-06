import unittest

import pandas as pd

from recommendation_engine import (
    available_evaluation_cases,
    evaluate_recommendations,
    pain_area_labels,
    prepare_dataset,
    recommend_poses,
)


def sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pose_name": "Back Stretch",
                "pain_area": "Back",
                "pain_keywords": "lower back, back pain",
                "description": "Gentle back mobility",
            },
            {
                "pose_name": "Neck Release",
                "pain_area": "Neck/Shoulder",
                "pain_keywords": "neck stiffness, shoulder",
                "description": "Relax the neck and shoulders",
            },
            {
                "pose_name": "Hip Opener",
                "pain_area": "Hip",
                "pain_keywords": "tight hips",
                "description": "Gentle hip mobility",
            },
        ]
    )


class RecommendationEngineTests(unittest.TestCase):
    def test_tfidf_search_ranks_matching_pose_first(self) -> None:
        results, warning, skipped = recommend_poses(
            "lower back pain", sample_dataset(), top_n=2, use_semantic=False
        )
        self.assertEqual(results.iloc[0]["pose_name"], "Back Stretch")
        self.assertEqual(results.iloc[0]["scoring_method"], "TF-IDF fallback")
        self.assertIsNone(warning)
        self.assertEqual(skipped, 0)

    def test_empty_searchable_rows_are_skipped(self) -> None:
        dataset = sample_dataset()
        dataset.loc[0, ["pose_name", "pain_area", "pain_keywords", "description"]] = ""
        prepared, documents, skipped = prepare_dataset(dataset)
        self.assertEqual(len(prepared), 2)
        self.assertEqual(len(documents), 2)
        self.assertEqual(skipped, 1)

    def test_pain_area_labels_split_multiple_categories(self) -> None:
        self.assertEqual(
            pain_area_labels("Back; Neck/Shoulder"),
            {"back", "neck/shoulder"},
        )

    def test_evaluation_only_uses_categories_present_in_dataset(self) -> None:
        cases = available_evaluation_cases(sample_dataset())
        self.assertEqual(
            {expected for _, _, expected in cases},
            {"Back", "Neck/Shoulder", "Hip"},
        )
        table, metrics, errors = evaluate_recommendations(
            sample_dataset(), top_k=2, use_semantic=False
        )
        self.assertEqual(len(table), 4)
        self.assertIn("Precision@2", metrics)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.values()))
        self.assertEqual(errors, [])

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            recommend_poses("  ", sample_dataset(), use_semantic=False)


if __name__ == "__main__":
    unittest.main()
