"""Test the targets shared by multimodal data validation and training."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_multimodal.models.multimodal import MultimodalAgent  # noqa: E402
from laya_multimodal.train import decision_target, target_from_expected, target_from_gold  # noqa: E402


DEPARTMENT = {"type": "choice", "instructions": "which team ?",
              "criteria": {"billing": "", "technical": "", "other": ""}}
URGENT = {"type": "noul", "instructions": "is it urgent ?"}
LEVEL = {"type": "score", "instructions": "which level ?", "criteria": ["low", "high"]}


class TargetTests(unittest.TestCase):
    def test_target_from_gold_orders_and_normalises(self):
        def target(question, probs):
            return target_from_gold(MultimodalAgent._to_internal(question), {"probabilities": probs})

        self.assertEqual(target(DEPARTMENT, {"technical": 2.0, "billing": 2.0}), [0.5, 0.5, 0.0])
        self.assertEqual(target(DEPARTMENT, {"technical": 1e308, "billing": 1e308}), [0.5, 0.5, 0.0])
        self.assertEqual(target(URGENT, {"true": 0.75, "false": 0.25}), [0.25, 0.75])
        self.assertEqual(target(LEVEL, {"1": 1.0}), [0.0, 1.0])
        self.assertEqual(target(DEPARTMENT, {}), [1 / 3] * 3)
        # list-form choice and non-string labels, which inference accepts
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": ["app", "screen"]},
                                {"screen": 1.0}), [0.0, 1.0])
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "", 2: ""}},
                                {"2": 3.0, "1": 1.0}), [0.25, 0.75])
        for bad in ({"billing": -1.0}, {"billing": float("nan")}, {"billing": "high"}, [0.5, 0.5]):
            with self.assertRaises(ValueError):
                target(DEPARTMENT, bad)

    def test_target_from_expected_choice(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(MultimodalAgent._to_internal(question), exp, label_smoothing=smoothing)

        self.assertEqual(target(DEPARTMENT, "billing"), [1.0, 0.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "technical"), [0.0, 1.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "other"), [0.0, 0.0, 1.0])
        # Non-string labels matching keys
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "a", 2: "b"}}, 1),
                         [1.0, 0.0])

    def test_target_from_expected_noul(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(MultimodalAgent._to_internal(question), exp, label_smoothing=smoothing)

        self.assertEqual(target(URGENT, True), [0.0, 1.0])
        self.assertEqual(target(URGENT, False), [1.0, 0.0])
        self.assertEqual(target(URGENT, "true"), [0.0, 1.0])
        self.assertEqual(target(URGENT, "false"), [1.0, 0.0])

    def test_target_from_expected_score(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(MultimodalAgent._to_internal(question), exp, label_smoothing=smoothing)

        self.assertEqual(target(LEVEL, 0), [1.0, 0.0])
        self.assertEqual(target(LEVEL, 1), [0.0, 1.0])
        self.assertEqual(target(LEVEL, "1"), [0.0, 1.0])

        # Fractional score answers (e.g. from laya-multimodal eval ScoreMAE / ScoreWithin)
        three_level = {"type": "score", "instructions": "which level ?", "criteria": ["low", "mid", "high"]}
        self.assertEqual(target(three_level, 1.5), [0.0, 0.5, 0.5])
        res_frac2 = target(three_level, 0.2)
        self.assertAlmostEqual(res_frac2[0], 0.8)
        self.assertAlmostEqual(res_frac2[1], 0.2)
        self.assertAlmostEqual(res_frac2[2], 0.0)

    def test_target_from_expected_label_smoothing(self):
        res = target_from_expected(MultimodalAgent._to_internal(DEPARTMENT), "billing", label_smoothing=0.1)
        self.assertAlmostEqual(res[0], 0.9 + 0.1 / 3)
        self.assertAlmostEqual(res[1], 0.1 / 3)
        self.assertAlmostEqual(res[2], 0.1 / 3)
        self.assertAlmostEqual(sum(res), 1.0)

        res_noul = target_from_expected(MultimodalAgent._to_internal(URGENT), True, label_smoothing=0.2)
        self.assertAlmostEqual(res_noul[0], 0.1)
        self.assertAlmostEqual(res_noul[1], 0.9)
        self.assertAlmostEqual(sum(res_noul), 1.0)

    def test_target_from_expected_rejects_invalid(self):
        with self.assertRaises(ValueError):
            target_from_expected(MultimodalAgent._to_internal(DEPARTMENT), "unknown_choice")
        with self.assertRaises(ValueError):
            target_from_expected(MultimodalAgent._to_internal(URGENT), "not_a_bool")
        for bad_num in (-3, 7, 0.5, -0.1, 2):
            with self.assertRaises(ValueError):
                target_from_expected(MultimodalAgent._to_internal(URGENT), bad_num)
        with self.assertRaises(ValueError):
            target_from_expected(MultimodalAgent._to_internal(LEVEL), 99)
        with self.assertRaises(ValueError):
            target_from_expected(MultimodalAgent._to_internal(LEVEL), -0.5)
        for bad_smooth in (-0.1, 1.0, 1.5, True):
            with self.assertRaises(ValueError):
                target_from_expected(MultimodalAgent._to_internal(DEPARTMENT), "billing", label_smoothing=bad_smooth)

    def test_decision_target_prefers_gold_when_both_labels_are_present(self):
        question = MultimodalAgent._to_internal(URGENT)
        row = {"expected": {"urgent": True},
               "gold": {"urgent": {"probabilities": {"false": .8, "true": .2}}}}
        self.assertEqual(decision_target(row, "urgent", question), [.8, .2])
        self.assertEqual(decision_target({"expected": row["expected"]}, "urgent", question), [0., 1.])


if __name__ == "__main__":
    unittest.main()
