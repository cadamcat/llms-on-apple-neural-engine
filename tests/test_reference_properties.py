"""Small known-value properties for the frozen reference rounding rules.

These tests intentionally avoid chain generation, matrix multiplication, and
any device/model work.  They use the public helpers from references.prepare.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ane_scope.references.prepare import fp16_rza, q8_rne, q8_rza


class ReferenceRoundingProperties(unittest.TestCase):
    def test_q8_rza_positive_and_negative_ties_away(self):
        values = np.array([-0.1875, -0.0625, 0.0625, 0.1875], dtype=np.float64)
        self.assertEqual(q8_rza(values).tolist(), [-2, -1, 1, 2])

    def test_q8_rne_positive_and_negative_ties_even(self):
        values = np.array([-0.1875, -0.0625, 0.0625, 0.1875], dtype=np.float64)
        self.assertEqual(q8_rne(values).tolist(), [-2, 0, 0, 2])

    def test_q8_saturates_both_i8_endpoints(self):
        self.assertEqual(q8_rza(np.array([-1000.0, 1000.0])).tolist(), [-128, 127])

    def test_fp16_rza_tie_and_signed_zero(self):
        half_ulp = 2.0 ** -11
        result = fp16_rza(np.array([1.0 + half_ulp, -(1.0 + half_ulp), 0.0, -0.0]))
        self.assertEqual(result[:2].tolist(), [1.0009765625, -1.0009765625])
        self.assertEqual(result[2:].tolist(), [0.0, 0.0])

    def test_invalid_scales_raise(self):
        for scale in (0.0, -0.125, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    q8_rza(np.array([0.0]), scale)
                with self.assertRaises(ValueError):
                    q8_rne(np.array([0.0]), scale)


if __name__ == "__main__":
    unittest.main()
