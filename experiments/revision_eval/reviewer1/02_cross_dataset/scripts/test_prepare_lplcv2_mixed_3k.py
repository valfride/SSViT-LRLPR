#!/usr/bin/env python3
"""Pure-selection regression tests (no restricted images or LMDB files needed).

Run:
    python3 -m unittest discover -s experiments/revision_eval/reviewer1/02_cross_dataset/scripts \
        -p 'test_prepare_lplcv2_mixed_3k.py' -v
"""
import unittest

import prepare_lplcv2_mixed_3k as builder


def make_row(group, index, width=30, height=10):
    return {
        "track_id": f"track_{index:06d}",
        "source_image": f"{group}_{index:06d}.jpg",
        "annotation_index": 0,
        "ocr": f"{group}{index:06d}",
        "crop_width": width,
        "crop_height": height,
    }


class AspectControlledSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.illegible = [make_row("I", i) for i in range(905)]
        cls.poor = [make_row("P", i) for i in range(3000)]
        # A near-square and an overly-wide crop in EACH original subset.
        cls.illegible[0] = make_row("I", 0, 10, 10)
        cls.illegible[1] = make_row("I", 1, 50, 10)
        cls.poor[0] = make_row("P", 0, 10, 10)
        cls.poor[1] = make_row("P", 1, 50, 10)
        # Boundaries are inclusive: exactly 1.7 and exactly 4.0 stay.
        cls.illegible[2] = make_row("I", 2, 17, 10)
        cls.poor[2] = make_row("P", 2, 40, 10)
        cls.base = (
            [("Illegible", r) for r in cls.illegible] +
            [("Poor", r) for r in cls.poor[:2095]]
        )

    def select(self, maximum, poor=None):
        return builder.aspect_filter_and_refill(
            self.base, self.illegible, self.poor if poor is None else poor,
            seed=2026, threshold=1.7, max_threshold=maximum,
            refill_min=2.5, refill_max=3.5)

    def test_two_sided_removals_preserve_total_and_category_accounting(self):
        selected, audit, replacement_ids = self.select(4.0)
        self.assertEqual(3000, len(selected))
        self.assertEqual({"Illegible": 2, "Poor": 2}, audit["removed_counts"])
        self.assertEqual(audit["removed_counts"], {
            "Illegible": 2, "Poor": 2})
        self.assertEqual({"Illegible": 1, "Poor": 1}, audit["removed_below_min"])
        self.assertEqual({"Illegible": 1, "Poor": 1}, audit["removed_above_max"])
        self.assertEqual({"Illegible": 903, "Poor": 2097}, audit["final_counts"])
        self.assertEqual(4, audit["removed_total"])
        self.assertEqual(4, len(replacement_ids))
        self.assertTrue(all(
            1.7 <= builder.original_aspect(r) <= 4.0 for _, r in selected))
        self.assertEqual(len(selected), len({
            (r["source_image"], r["annotation_index"]) for _, r in selected}))
        selected_ids = {r["source_image"] for _, r in selected}
        self.assertIn("I_000002.jpg", selected_ids)
        self.assertIn("P_000002.jpg", selected_ids)
        self.assertNotIn("I_000000.jpg", selected_ids)
        self.assertNotIn("I_000001.jpg", selected_ids)

    def test_deterministic_refill(self):
        a, report_a, ids_a = self.select(4.0)
        b, report_b, ids_b = self.select(4.0)
        self.assertEqual([(grp, r["source_image"]) for grp, r in a],
                         [(grp, r["source_image"]) for grp, r in b])
        self.assertEqual(report_a, report_b)
        self.assertEqual(ids_a, ids_b)

    def test_backward_compatible_lower_bound_only(self):
        selected, audit, ids = self.select(None)
        self.assertEqual(3000, len(selected))
        self.assertEqual(2, audit["removed_total"])
        self.assertEqual({}, audit["removed_above_max"])
        self.assertEqual({"Illegible": 904, "Poor": 2096}, audit["final_counts"])
        self.assertEqual(2, len(ids))
        self.assertTrue(any(builder.original_aspect(r) == 5 for _, r in selected))

    def test_without_filter_keeps_original_selection(self):
        selected, audit, ids = builder.aspect_filter_and_refill(
            self.base, self.illegible, self.poor, 2026,
            None, None, 2.5, 3.5)
        self.assertIs(self.base, selected)
        self.assertIsNone(audit)
        self.assertEqual(set(), ids)

    def test_never_relaxes_if_replacements_are_insufficient(self):
        with self.assertRaisesRegex(RuntimeError, "only 3 unused Poor"):
            self.select(4.0, poor=self.poor[:2098])

    def test_invalid_upper_bound_and_unpaired_maximum_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.select(3.0)
        with self.assertRaisesRegex(ValueError, "requires a minimum"):
            builder.aspect_filter_and_refill(
                self.base, self.illegible, self.poor, 2026,
                None, 4.0, 2.5, 3.5)


if __name__ == "__main__":
    unittest.main()
