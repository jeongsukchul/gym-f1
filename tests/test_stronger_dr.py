import unittest
from reevaluate_stronger_dr import stronger_ranges


class TestStrongerDR(unittest.TestCase):
    def test_widen_around_nominal_without_mutating_original(self):
        original = {"m": [.85, 1.15], "tire": [.75, 1.25]}
        widened = stronger_ranges(original, 4./3.)
        self.assertAlmostEqual(widened["m"][0], .8)
        self.assertAlmostEqual(widened["m"][1], 1.2)
        self.assertAlmostEqual(widened["tire"][0], 2./3.)
        self.assertAlmostEqual(widened["tire"][1], 4./3.)
        self.assertEqual(original["m"], [.85, 1.15])

    def test_reject_invalid_or_nonphysical_strength(self):
        for strength in (1., .5, float("nan"), float("inf"), 10.):
            with self.assertRaises(ValueError):
                stronger_ranges({"m": [.75, 1.25]}, strength)


if __name__ == "__main__":
    unittest.main()
