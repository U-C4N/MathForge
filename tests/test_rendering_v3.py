"""Exact presentation is additive, bounded and never a serialization shortcut."""
from dataclasses import replace
import unittest

import mathforge as m


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.x = m.Symbol("x", uid="00000000-0000-4000-8000-000000000001")

    def test_exact_fraction_and_radical_snapshots(self):
        self.assertEqual(m.render_text(m.Rational(-3, 4)), "-3/4")
        self.assertEqual(m.to_latex(m.Rational(-3, 4)), r"-\frac{3}{4}")
        self.assertEqual(m.to_latex(m.sqrt(2)), r"\sqrt{2}")
        self.assertEqual(m.render_text(m.Rational(1, 10)), "1/10")

    def test_precedence(self):
        x = self.x
        self.assertEqual(m.render_text(m.Pow(m.Add((x, m.Rational(1))), 2)), "(x + 1)^2")
        self.assertEqual(m.to_latex(m.Pow(m.Rational(-1, 2), 2)), r"\left(-\frac{1}{2}\right)^{2}")
        self.assertEqual(m.render_text(m.Pow(m.Pow(x, 2), 3)), "(x^2)^3")

    def test_polynomial(self):
        p = m.Polynomial((-2, 0, 1), self.x)
        self.assertEqual(m.render_text(p), "x^2 - 2")
        self.assertEqual(m.to_latex(p), "x^{2} - 2")

    def test_noncanonical_multiplication_is_unambiguous(self):
        self.assertEqual(m.to_latex(m.Mul((m.Rational(-2), m.Rational(-3)))),
                         r"\left(-2\right) \cdot \left(-3\right)")
        self.assertEqual(m.to_latex(m.Mul((m.Rational(2), m.Rational(3)))), r"2 \cdot 3")
        self.assertEqual(m.render_text(m.Mul((m.Rational(-1), m.Rational(-2)))), "-(-2)")

    def test_holes_are_visible_and_objects_unchanged(self):
        x = self.x
        rf = m.rational_function(x, x, variable=x)
        before, original_str = m.dumps(rf), str(rf)
        self.assertEqual(m.render_text(rf), "(1)/(1) where x != 0")
        self.assertEqual(m.to_latex(rf), r"\frac{1}{1}\quad\text{where } x \ne 0")
        self.assertEqual(m.dumps(rf), before)
        self.assertEqual(str(rf), original_str)

    def test_root_descriptor_and_multiplicity(self):
        root = m.RealAlgebraicRoot((-2, 0, 0, 1), 0)
        self.assertEqual(m.render_text(root), "RootOf(t^3 - 2; real_index=0)")
        self.assertEqual(m.to_latex(root), r"\operatorname{RootOf}_{\mathbb{R}}\left(t^{3} - 2; 0\right)")
        self.assertIn("multiplicity=3", m.render_text(m.RootRecord(root, 3)))

    def test_sets(self):
        s = m.IntervalSet((m.Interval(None, -1), m.Interval(1, None, True)))
        self.assertEqual(m.render_text(s), "(-inf, -1) union [1, inf)")
        self.assertEqual(m.to_latex(s), r"(-\infty, -1) \cup [1, \infty)")
        self.assertEqual(m.to_latex(m.EmptySet()), r"\varnothing")
        self.assertEqual(m.to_latex(m.UniversalSet()), r"\mathbb{R}")
        self.assertEqual(m.render_text(m.FiniteSet((1, 2))), "{1, 2}")

    def test_matrices_and_empty_dimensions(self):
        matrix = m.Matrix(((1, 2), (3, 4)))
        self.assertEqual(m.render_text(matrix), "[[1, 2], [3, 4]]")
        self.assertEqual(m.to_latex(matrix), r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}")
        self.assertEqual(m.render_text(m.Matrix((), ncols=3)), "Matrix(0x3)")
        self.assertEqual(m.render_text(m.Matrix(((), ()))), "Matrix(2x0)")
        self.assertIn("pivot columns (0-based)", m.render_text(m.rref(matrix)))

    def test_affine_system_and_factorization(self):
        x, y = self.x, m.symbol("y")
        system = m.LinearSystem(m.Matrix(((1, 1),)), (2,), (x, y))
        result = m.solve_system(system)
        self.assertIn("parameters are independent real scalars", m.render_text(result.unwrap()))
        self.assertIn(" = ", m.to_latex(system))
        f = m.Polynomial.from_expression((x - 1)**2, x)
        self.assertIn("^2", m.render_text(m.square_free(f)))

    def test_no_implicit_verification_claim(self):
        result = m.solve(m.Eq(self.x**2, 2), for_=self.x)
        before = m.dumps(result)
        self.assertIn("completed / solutions", m.render_text(result))
        self.assertNotIn("verified", m.render_text(result))
        self.assertIn("verified", m.render_text(m.verify(result)))
        self.assertEqual(m.dumps(result), before)
        failure = m.operations.integrate(m.sqrt(2), self.x, 0, 1)
        self.assertIn("unsupported", m.render_text(failure))

    def test_symbol_ambiguity_requires_aliases(self):
        other = m.symbol("x")
        value = m.Add((self.x, other))
        with self.assertRaises(m.InvalidInput):
            m.render_text(value)
        aliases = {"left": self.x, "right": other}
        self.assertEqual(m.render_text(value, symbols=aliases), "left + right")
        with self.assertRaises(m.InvalidInput):
            m.to_latex(value, symbols={"a": self.x, "b": self.x})
        with self.assertRaises(m.InvalidInput):
            m.render_text(self.x, symbols={"a": replace(self.x, name="changed")})

    def test_latex_escape_and_text_quoting(self):
        symbol = m.symbol(r"\{}_ ^%&#$~")
        latex = m.to_latex(symbol)
        for escape in (r"\textbackslash{}", r"\{", r"\}", r"\_", r"\textasciicircum{}", r"\%", r"\&", r"\#", r"\$", r"\textasciitilde{}"):
            self.assertIn(escape, latex)
        self.assertTrue(m.render_text(symbol).startswith('"'))

    def test_character_boundaries(self):
        self.assertEqual(m.render_text(123, limits=m.RenderLimits(max_chars=3)), "123")
        with self.assertRaises(m.ResourceLimitError):
            m.render_text(123, limits=m.RenderLimits(max_chars=2))
        with self.assertRaises(m.ResourceLimitError):
            m.to_latex(m.symbol("a_b"), limits=m.RenderLimits(max_chars=4))

    def test_node_depth_boundaries_and_cycle(self):
        self.assertEqual(m.render_text((1,), limits=m.RenderLimits(max_nodes=2, max_depth=1)), "(1)")
        with self.assertRaises(m.ResourceLimitError):
            m.render_text((1,), limits=m.RenderLimits(max_nodes=1))
        with self.assertRaises(m.ResourceLimitError):
            m.render_text(((1,),), limits=m.RenderLimits(max_depth=1))
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(m.ResourceLimitError):
            m.render_text(cycle)

    def test_large_integer_without_global_int_setting(self):
        import sys
        before = sys.get_int_max_str_digits()
        value = 10**4500
        self.assertEqual(m.render_text(value), "1" + "0" * 4500)
        self.assertEqual(sys.get_int_max_str_digits(), before)
        with self.assertRaises(m.ResourceLimitError):
            m.render_text(value, limits=m.RenderLimits(max_chars=100))

    def test_raised_depth_reports_host_resource_limit(self):
        value = 1
        for _ in range(400):
            value = (value,)
        for renderer in (m.render_text, m.to_latex):
            with self.assertRaises(m.ResourceLimitError):
                renderer(value, limits=m.RenderLimits(max_depth=1000))

    def test_wide_input_and_long_polynomial_fail_without_full_render(self):
        from unittest.mock import patch
        import mathforge.rendering as rendering
        value = [1] * 200_000
        with self.assertRaises(m.ResourceLimitError):
            m.render_text(value, limits=m.RenderLimits(max_nodes=1))
        p = m.Polynomial((10**500,) * 1000, self.x)
        original = rendering._Renderer.integer
        calls = []
        def counted(renderer, number):
            calls.append(number)
            return original(renderer, number)
        with patch.object(rendering._Renderer, "integer", counted):
            with self.assertRaises(m.ResourceLimitError):
                m.render_text(p, limits=m.RenderLimits(max_chars=1024))
        self.assertLess(len(calls), 20)

    def test_invalid_and_unsupported_inputs(self):
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(m.InvalidInput):
                m.RenderLimits(max_chars=limit)
        self.assertEqual(m.render_text(True), "true")
        for value in (1.2, object(), {"x": 1}):
            with self.assertRaises(m.UnsupportedOperation):
                m.render_text(value)
        with self.assertRaises(m.InvalidInput):
            m.render_text(1, limits=m.ComputationLimits())


if __name__ == "__main__":
    unittest.main()
