from __future__ import annotations

import unittest

from training.build_page_level_split import build_page_level_split


class BuildPageLevelSplitTests(unittest.TestCase):
    def test_build_page_level_split_can_split_single_project_dataset_by_page(self) -> None:
        rows = [
            {"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": f"IMG_{index:04d}", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            for index in range(10)
        ]

        split = build_page_level_split(
            rows,
            dataset_slug="demo_dataset",
            val_ratio=0.2,
            holdout_ratio=0.2,
        )

        self.assertEqual(split["dataset_slug"], "demo_dataset")
        self.assertEqual(len(split["train_page_ids"]), 6)
        self.assertEqual(len(split["val_page_ids"]), 2)
        self.assertEqual(len(split["holdout_page_ids"]), 2)
        self.assertEqual(len(set(split["train_page_ids"]) & set(split["val_page_ids"])), 0)
        self.assertEqual(len(set(split["train_page_ids"]) & set(split["holdout_page_ids"])), 0)

    def test_build_page_level_split_can_use_train_val_only_mode_for_tiny_dataset(self) -> None:
        rows = [
            {"dataset_slug": "tiny_dataset", "project_slug": "project_001", "page_id": f"IMG_{index:04d}", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            for index in range(7)
        ]

        split = build_page_level_split(
            rows,
            dataset_slug="tiny_dataset",
            val_ratio=0.28,
            holdout_ratio=0.14,
            train_val_only=True,
        )

        self.assertEqual(len(split["holdout_page_ids"]), 0)
        self.assertEqual(len(split["val_page_ids"]), 2)
        self.assertEqual(len(split["train_page_ids"]), 5)
        self.assertEqual(split["metadata"]["mode"], "train_val_only")

    def test_build_page_level_split_ignores_rows_without_manual_quad(self) -> None:
        rows = [
            {"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": "IMG_0001", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            {"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": "IMG_0002", "manual_quad": None},
            {"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": "IMG_0003", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        ]

        split = build_page_level_split(
            rows,
            dataset_slug="demo_dataset",
            val_ratio=0.5,
            holdout_ratio=0.0,
            train_val_only=True,
        )

        all_page_ids = split["train_page_ids"] + split["val_page_ids"] + split["holdout_page_ids"]
        self.assertNotIn("demo_dataset:project_001:IMG_0002", all_page_ids)
        self.assertEqual(split["metadata"]["manual_pages"], 2)

    def test_build_page_level_split_can_preserve_label_distribution_with_explicit_counts(self) -> None:
        rows = []
        stratify_labels = {}
        for index in range(3):
            page_id = f"OUT_{index:04d}"
            rows.append({"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": page_id, "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
            stratify_labels[f"demo_dataset:project_001:{page_id}"] = "corner_out_of_frame"
        for index in range(5):
            page_id = f"PER_{index:04d}"
            rows.append({"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": page_id, "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
            stratify_labels[f"demo_dataset:project_001:{page_id}"] = "strong_perspective"
        for index in range(12):
            page_id = f"NOR_{index:04d}"
            rows.append({"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": page_id, "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
            stratify_labels[f"demo_dataset:project_001:{page_id}"] = "normal"

        split = build_page_level_split(
            rows,
            dataset_slug="demo_dataset",
            target_counts={"train": 12, "val": 4, "holdout": 4},
            stratify_labels=stratify_labels,
        )

        self.assertEqual(len(split["train_page_ids"]), 12)
        self.assertEqual(len(split["val_page_ids"]), 4)
        self.assertEqual(len(split["holdout_page_ids"]), 4)

        train_labels = [stratify_labels[page_id] for page_id in split["train_page_ids"]]
        val_labels = [stratify_labels[page_id] for page_id in split["val_page_ids"]]
        holdout_labels = [stratify_labels[page_id] for page_id in split["holdout_page_ids"]]

        self.assertEqual(train_labels.count("corner_out_of_frame"), 1)
        self.assertEqual(val_labels.count("corner_out_of_frame"), 1)
        self.assertEqual(holdout_labels.count("corner_out_of_frame"), 1)
        self.assertEqual(train_labels.count("strong_perspective"), 3)
        self.assertEqual(val_labels.count("strong_perspective"), 1)
        self.assertEqual(holdout_labels.count("strong_perspective"), 1)
        self.assertEqual(train_labels.count("normal"), 8)
        self.assertEqual(val_labels.count("normal"), 2)
        self.assertEqual(holdout_labels.count("normal"), 2)

    def test_build_page_level_split_rejects_invalid_explicit_counts(self) -> None:
        rows = [
            {"dataset_slug": "demo_dataset", "project_slug": "project_001", "page_id": f"IMG_{index:04d}", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            for index in range(5)
        ]

        with self.assertRaises(ValueError):
            build_page_level_split(
                rows,
                dataset_slug="demo_dataset",
                target_counts={"train": 3, "val": 2, "holdout": 2},
            )


if __name__ == "__main__":
    unittest.main()
