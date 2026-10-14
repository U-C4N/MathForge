"""Cartesian algebraic arithmetic, principal square roots, and quadratic roots."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from mathforge import complex_numbers as cx
from mathforge import algebraic_arithmetic as aa
from mathforge import Rational, sqrt, ComputationLimits
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ResourceLimitError


class ComplexTests(unittest.TestCase):
    def checked(self, operation, **args):
        value, evidence = cx.produce(operation, args)
        self.assertTrue(cx.check_operation(operation, args, value, evidence), (operation, value))
        return value, evidence

    def test_rational_cartesian_operations(self):
        a, b = cx.ComplexAlgebraic(2, 3), cx.ComplexAlgebraic(4, -5)
        self.assertEqual(self.checked("complex_add", left=a, right=b)[0], cx.ComplexAlgebraic(6, -2))
        self.assertEqual(self.checked("complex_sub", left=a, right=b)[0], cx.ComplexAlgebraic(-2, 8))
        self.assertEqual(self.checked("complex_mul", left=a, right=b)[0], cx.ComplexAlgebraic(23, 2))
        self.assertEqual(self.checked("complex_div", left=a, right=b)[0], cx.ComplexAlgebraic(Rational(-7, 41), Rational(22, 41)))
        self.assertEqual(self.checked("complex_conjugate", value=a)[0], cx.ComplexAlgebraic(2, -3))
        self.assertEqual(self.checked("complex_norm_squared", value=a)[0], Rational(13))
        self.assertEqual(self.checked("complex_real", value=a)[0], Rational(2))
        self.assertEqual(self.checked("complex_imag", value=a)[0], Rational(3))

    def test_independent_algebraic_components(self):
        a, _ = self.checked("complex_algebraic", real=sqrt(2), imag=sqrt(3))
        result, _ = self.checked("complex_norm_squared", value=a)
        self.assertEqual(result, Rational(5))
        product, _ = self.checked("complex_mul", left=a, right=a.conjugate())
        self.assertEqual(product, cx.ComplexAlgebraic(5))

    def test_signed_powers_and_principal_root(self):
        imaginary = cx.ComplexAlgebraic(0, 1)
        for n, expected in ((0, cx.ComplexAlgebraic(1)), (4, cx.ComplexAlgebraic(1)),
                            (-1, cx.ComplexAlgebraic(0, -1)), (-3, imaginary)):
            self.assertEqual(self.checked("complex_pow", base=imaginary, n=n)[0], expected)
        for source, expected in ((cx.ComplexAlgebraic(3, 4), cx.ComplexAlgebraic(2, 1)),
                                 (cx.ComplexAlgebraic(-3, -4), cx.ComplexAlgebraic(1, -2)),
                                 (-9, cx.ComplexAlgebraic(0, 3)), (0, cx.ComplexAlgebraic(0))):
            self.assertEqual(self.checked("complex_sqrt", value=source)[0], expected)
        with self.assertRaises(InvalidInput):
            cx.produce("complex_pow", {"base": 0, "n": -1})
        with self.assertRaises(InvalidInput):
            cx.produce("complex_div", {"left": imaginary, "right": 0})
        with self.assertRaises(UnsupportedOperation):
            sqrt(-1)

    def test_quadratic_multiplicities_and_order(self):
        repeated, _ = self.checked("quadratic_roots", a=1, b=-2, c=1)
        self.assertEqual(repeated, (cx.ComplexRoot(cx.ComplexAlgebraic(1), 2),))
        imaginary, _ = self.checked("quadratic_roots", a=1, b=0, c=1)
        self.assertEqual(imaginary, (cx.ComplexRoot(cx.ComplexAlgebraic(0, -1), 1), cx.ComplexRoot(cx.ComplexAlgebraic(0, 1), 1)))
        real, _ = self.checked("quadratic_roots", a=-1, b=0, c=2)
        self.assertEqual(real[0].root, cx.ComplexAlgebraic(-sqrt(2)))
        self.assertEqual(real[1].root, cx.ComplexAlgebraic(sqrt(2)))
        with self.assertRaises(InvalidInput):
            cx.produce("quadratic_roots", {"a": 0, "b": 1, "c": 0})

    def test_square_root_of_nonrational_algebraic_components(self):
        source = cx.ComplexAlgebraic(1, 1)
        root, _ = self.checked("complex_sqrt", value=source)
        self.assertEqual(root*root, source)
        fourth_root, _ = self.checked("complex_sqrt", value=cx.ComplexAlgebraic(sqrt(2)))
        self.assertEqual(fourth_root**4, cx.ComplexAlgebraic(2))

    def test_corruption_and_all_producers_disabled(self):
        z = cx.ComplexAlgebraic(3, 4)
        cases = [("complex_algebraic", {"real": sqrt(2), "imag": 1}),
                 ("complex_mul", {"left": z, "right": z}),
                 ("complex_div", {"left": z, "right": 2}),
                 ("complex_pow", {"base": z, "n": -3}),
                 ("complex_sqrt", {"value": z}), ("quadratic_roots", {"a": 1, "b": 0, "c": 2})]
        records = [(op, args, *cx.produce(op, args)) for op, args in cases]
        with patch.object(cx, "produce", side_effect=AssertionError("complex producer")), \
             patch.object(cx, "_binary_formula", side_effect=AssertionError("formula producer")), \
             patch.object(aa, "produce", side_effect=AssertionError("real algebraic producer")):
            for op, args, value, evidence in records:
                self.assertTrue(cx.check_operation(op, args, value, evidence), op)
                self.assertFalse(cx.check_operation(op, args, value, evidence+("forged",)), op)
        value, evidence = cx.produce("complex_sqrt", {"value": z})
        self.assertFalse(cx.check_operation("complex_sqrt", {"value": z}, replace(value, imag=-value.imag), evidence))
        roots, evidence = cx.produce("quadratic_roots", {"a": 1, "b": 0, "c": 1})
        self.assertFalse(cx.check_operation("quadratic_roots", {"a": 1, "b": 0, "c": 1}, roots[::-1], evidence))

    def test_budget_and_inexact_inputs(self):
        with self.assertRaises(ResourceLimitError):
            cx.produce("complex_sqrt", {"value": 2}, limits=ComputationLimits(max_work=1))
        with self.assertRaises(TypeError):
            cx.ComplexAlgebraic(1.25)


if __name__ == "__main__":
    unittest.main()
