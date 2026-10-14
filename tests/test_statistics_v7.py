import unittest
from unittest.mock import patch

from mathforge import Rational as Q
from mathforge.errors import InvalidInput
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Sqrt
import mathforge.statistics as s


class ExactStatisticsTests(unittest.TestCase):
    def checked(self, op, **args):
        value, evidence = s.produce(op, args)
        self.assertTrue(s.check_operation(op, args, value, evidence))
        self.assertFalse(s.check_operation(op, args, value, ("forged",)))
        return value

    def test_statistics_and_ddof(self):
        data = (1, 2, 4)
        self.assertEqual(self.checked("mean", data=data), Q(7, 3))
        self.assertEqual(self.checked("median", data=data), Q(2))
        self.assertEqual(self.checked("variance", data=data, ddof=0), Q(14, 9))
        self.assertEqual(self.checked("variance", data=data, ddof=1), Q(7, 3))
        self.assertEqual(self.checked("covariance", x=data, y=(2, 4, 8), ddof=1), Q(14, 3))
        self.assertEqual(self.checked("stddev", data=(0, 2), ddof=0), Q(1))
        self.assertEqual(self.checked("stddev", data=(0, 2), ddof=1), Sqrt(Q(2)))

    def test_order_statistics_ties_and_rationals(self):
        for data, expected in (((9, 1, 3, 2), Q(5, 2)), ((2, 2, 2), Q(2)),
                               ((Q(-1, 3), Q(1, 2)), Q(1, 12)), ((7,), Q(7))):
            self.assertEqual(self.checked("median", data=data), expected)
        self.assertEqual(self.checked("variance", data=(7,), ddof=0), Q(0))

    def test_independent_and_corrupt_checks(self):
        for op, args in (("mean", {"data": (1, 3)}), ("median", {"data": (1, 3)}),
                         ("variance", {"data": (1, 3), "ddof": 0}),
                         ("stddev", {"data": (1, 3), "ddof": 0}),
                         ("covariance", {"x": (1, 3), "y": (2, 4), "ddof": 0})):
            value, evidence = s.produce(op, args)
            with patch.object(s, op, side_effect=AssertionError("producer used")):
                self.assertTrue(s.check_operation(op, args, value, evidence))
                self.assertFalse(s.check_operation(op, args, Q(99), evidence))
        self.assertFalse(s.check_operation("stddev", {"data": (0, 2), "ddof": 0}, Q(-1), ()))

    def test_invalid_and_budget(self):
        for call in (lambda: s.mean(()), lambda: s.mean((1.5,)), lambda: s.variance((1,), 1),
                     lambda: s.variance((1, 2), True), lambda: s.covariance((1,), (1, 2))):
            with self.assertRaises(InvalidInput):
                call()
        with computation(ComputationLimits(max_nodes=3)):
            with self.assertRaises(ResourceLimitError):
                s.mean(range(20))


if __name__ == "__main__":
    unittest.main()
