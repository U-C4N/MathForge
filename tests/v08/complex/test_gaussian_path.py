"""Complex scalars over the Gaussian rationals: the lean path, the structured path and v0.7 agree.

An operand is a Gaussian rational when it is an int, a Rational or a ComplexAlgebraic with Rational parts. For such
operands ``mathforge.lean.functions.complex_*`` computes with integers and checks its answer another way
(src/mathforge/lean/algebra_complex.py); everything else goes to the generic runner or the structured path.

The seeded inputs below are deterministic, built from an integer hash and no ``random``. For each operation, the
sha256 of the first COUNT outcomes (the serialized value, or the error class and code) was computed on a v0.7 tree
and is pinned in V07_DIGESTS: ``digest(operation)`` must give it on the current code. The module imports nothing
from mathforge.lean outside the tests, so ``digest`` runs on a v0.7 tree as well:

    PYTHONPATH=<v0.7 tree>/src python -c "from tests.v08.complex.test_gaussian_path import digests; print(digests())"
"""

from __future__ import annotations

import hashlib
import json
import unittest
from fractions import Fraction
from unittest.mock import patch

import mathforge as mf
from mathforge.errors import OperationError
from mathforge.serialization import to_data

COUNT = 2000
BINARY = ("complex_add", "complex_sub", "complex_mul", "complex_div")
UNARY = ("complex_conjugate", "complex_real", "complex_imag", "complex_norm_squared")
OPERATIONS = ("complex_algebraic", *BINARY, *UNARY, "complex_sqrt", "complex_pow", "quadratic_roots")
TYPE_EDGES = (True, 1.5, "1", None)

# The sha256 of the outcomes of the first COUNT seeded inputs of each operation, from the v0.7 tree (commit 007a4b3).
V07_DIGESTS = {
    "complex_algebraic": "40e653b5522a084ad5a808a6d894417cdd82d92a511853ecfddca31677a88dc0",
    "complex_add": "73da2bf696d689e0fa6e3f81223810a9865826288593d410af9deefd8204f622",
    "complex_sub": "f3899b31b207f96680135ef9d95963cd7038434384838a855ba48ab62b6e3e32",
    "complex_mul": "d3d1cf80aa23b73ee5ea86aa64235494ed388228b7f756ef9d29f5655d4ae791",
    "complex_div": "2bb026960c5297f3cd34e5d9a7ec572a18d9891940ad973572393c13cf279f53",
    "complex_conjugate": "dc5fc3617f932ef547cbbc6a3a417b48c5a204db45b4bbf459bd50d650e1fce1",
    "complex_real": "9e63f671e4e82ff8620525e35090cac7af3e68694904fef31cc3e726cc3f1c7a",
    "complex_imag": "e810560e6939d1901557b3ac48621969076422568e577754cec5d978bfca9bad",
    "complex_norm_squared": "1abe70aedec26cca9d7749a88fbeed9bdfc3ae06828aa3ba02bbf575ed23b3ac",
    "complex_sqrt": "25758b994e17698b8969a819318c4bdc9f5ce7634a98857f220b0d8f2cb681e7",
    "complex_pow": "a54e3807938952d43d8770167db57237ed7909e259d73f238419f3fcabda5801",
    "quadratic_roots": "55ce769efd987596539c5c240b07bd20a136599117d8ad11b51abb976337de2f",
}

_MASK = (1 << 64) - 1


# ---------------------------------------------------------------------------------------------------------------
# Seeded inputs

def _mix(*values):
    h = 0x9E3779B97F4A7C15
    for value in values:
        h = ((h ^ (value & _MASK)) * 0xBF58476D1CE4E5B9) & _MASK
        h ^= h >> 31
    return h


def _draw(low, high, *key):
    return low + _mix(*key) % (high - low + 1)


def _fraction(tag, i, k, big):
    """A Fraction: an integer, or n/d with |n| <= 999 and d <= 99 (big) or |n| <= 9 and d <= 4."""
    n = _draw(-999 if big else -9, 999 if big else 9, tag, i, k, 0)
    if big or _draw(0, 2, tag, i, k, 2) == 0:
        return Fraction(n, _draw(1, 99 if big else 4, tag, i, k, 1))
    return Fraction(n)


def _rational(value):
    """The int or Rational that stands for a Fraction, by turns: both input types are covered."""
    return int(value) if value.denominator == 1 else mf.Rational(value.numerator, value.denominator)


def _operand(tag, i, k):
    """An int, a Rational, or a ComplexAlgebraic with Rational parts, some of them zero, one or a unit."""
    kind = _draw(0, 9, tag, i, k, 9)
    if kind == 0:
        return _draw(-30, 30, tag, i, k, 8)
    if kind == 1:
        return _rational(_fraction(tag, i, k, True))
    if kind == 2:
        return mf.ComplexAlgebraic(_draw(-1, 1, tag, i, k, 7), _draw(-1, 1, tag, i, k, 6))
    big = kind in (3, 4, 5)
    return mf.ComplexAlgebraic(_rational(_fraction(tag, i, k, big)), _rational(_fraction(tag, i, k + 50, big)))


def _parts(value):
    """(real, imaginary) of a Gaussian operand as Fractions."""
    if type(value) is mf.ComplexAlgebraic:
        return (Fraction(value.real.numerator, value.real.denominator),
                Fraction(value.imag.numerator, value.imag.denominator))
    if type(value) is mf.Rational:
        return Fraction(value.numerator, value.denominator), Fraction(0)
    return Fraction(value), Fraction(0)


def _square_of(i):
    """A Gaussian square (p + q i)**2 with its principal root, for p and q rationals of a few digits."""
    p, q = _fraction(30, i, 0, False), _fraction(30, i, 1, False)
    if p < 0 or (p == 0 and q < 0):
        root = (-p, -q)
    else:
        root = (p, q)
    return mf.ComplexAlgebraic(_rational(p * p - q * q), _rational(2 * p * q)), root


def _quadratic(i):
    """(a, b, c) of a polynomial with two distinct rational roots, a double root or a conjugate pair p +- q i,
    and the roots as Fraction pairs in the order that quadratic_roots returns them (increasing)."""
    a = _fraction(40, i, 0, False)
    if a == 0:
        a = Fraction(1)
    kind = _draw(0, 3, 40, i, 1)
    p, q = _fraction(40, i, 2, False), _fraction(40, i, 3, False)
    if kind == 0:  # two real roots p and p + |q| (or a double root if q is zero)
        roots = [(p, Fraction(0)), (p + abs(q), Fraction(0))]
    elif kind == 1:
        roots = [(p, Fraction(0))] * 2
    else:
        q = abs(q) or Fraction(1)
        roots = [(p, -q), (p, q)]
    r1, r2 = roots[0], roots[1]
    total, product = (r1[0] + r2[0], r1[1] + r2[1]), (r1[0] * r2[0] - r1[1] * r2[1], r1[0] * r2[1] + r1[1] * r2[0])
    # The sum and the product of conjugate or real roots are real.
    assert total[1] == 0 and product[1] == 0
    # Increasing by real part, then imaginary part, whatever the sign of a.
    roots = [roots[0]] if roots[0] == roots[1] else sorted(roots)
    return (_rational(a), _rational(-a * total[0]), _rational(a * product[0])), roots


def arguments(operation, i):
    """The arguments of seeded input i of an operation."""
    tag = OPERATIONS.index(operation) + 1
    if operation == "complex_algebraic":
        return _rational(_fraction(tag, i, 0, i % 2 == 0)), _rational(_fraction(tag, i, 1, i % 3 == 0))
    if operation in BINARY:
        return _operand(tag, i, 0), _operand(tag, i, 1)
    if operation in UNARY:
        return (_operand(tag, i, 0),)
    if operation == "complex_sqrt":
        return (_square_of(i)[0],)
    if operation == "complex_pow":
        return _operand(tag, i, 0), _draw(-8, 24, tag, i, 5)
    return _quadratic(i)[0]


def expected(operation, args, i):
    """What the operation gives for seeded input i, from Fraction arithmetic alone: a pair of Fractions (or a tuple of
    them, for quadratic_roots), a Fraction, or None where the structured path raises."""
    if operation == "complex_algebraic":
        return _parts(args[0])[0], _parts(args[1])[0]
    if operation in BINARY:
        (a, b), (c, d) = _parts(args[0]), _parts(args[1])
        if operation == "complex_add":
            return a + c, b + d
        if operation == "complex_sub":
            return a - c, b - d
        if operation == "complex_mul":
            return a * c - b * d, a * d + b * c
        if c == 0 and d == 0:
            return None
        norm = c * c + d * d
        return (a * c + b * d) / norm, (b * c - a * d) / norm
    if operation in UNARY:
        a, b = _parts(args[0])
        return {"complex_conjugate": (a, -b), "complex_real": a, "complex_imag": b,
                "complex_norm_squared": a * a + b * b}[operation]
    if operation == "complex_sqrt":
        return _square_of(i)[1]
    if operation == "complex_pow":
        (a, b), n = _parts(args[0]), args[1]
        if n < 0:
            if a == 0 and b == 0:
                return None
            norm = a * a + b * b
            a, b, n = a / norm, -b / norm, -n  # The reciprocal's conjugate.
        x, y = Fraction(1), Fraction(0)
        for _ in range(n):
            x, y = x * a - y * b, x * b + y * a
        return x, y
    return tuple(_quadratic(i)[1])


def _same(value, want, operation):
    """Whether a returned value is the one that expected() computed."""
    if want is None:
        return False
    if operation == "quadratic_roots":
        return (type(value) is tuple and len(value) == len(want)
                and all(_same(root.root, pair, "complex_algebraic")
                        and root.multiplicity == (2 if len(want) == 1 else 1) for root, pair in zip(value, want)))
    if operation in ("complex_real", "complex_imag", "complex_norm_squared"):
        return type(value) is mf.Rational and _fraction_of(value) == want
    return (type(value) is mf.ComplexAlgebraic and type(value.real) is mf.Rational
            and type(value.imag) is mf.Rational and (_fraction_of(value.real), _fraction_of(value.imag)) == want)


def _fraction_of(rational):
    return Fraction(rational.numerator, rational.denominator)


def outcome(call):
    """("value", value), or ("error", class, code) for a failure; a Result stands for its unwrapped value."""
    try:
        value = call()
        if isinstance(value, mf.Result):
            value = value.unwrap()
        return ("value", value)
    except Exception as error:
        return ("error", type(error), getattr(error, "code", None))


def digest(operation, count=COUNT):
    """sha256 of the serialized outcomes of the first ``count`` seeded inputs, through the public value function."""
    sha = hashlib.sha256()
    for i in range(count):
        try:
            record = ["value", to_data(getattr(mf, operation)(*arguments(operation, i)))]
        except OperationError as error:
            record = ["error", type(error).__name__, error.code]
        sha.update(json.dumps(record, sort_keys=True).encode("ascii"))
    return sha.hexdigest()


def digests():
    return {operation: digest(operation) for operation in OPERATIONS}


def _data(value):
    try:
        return to_data(value)
    except Exception as error:
        return ("not serializable", type(error).__name__)


class GaussianPathTest(unittest.TestCase):
    maxDiff = None

    def assertSameValue(self, got, want, context=None):
        self.assertIs(type(got), type(want), context)
        self.assertEqual(got, want, context)
        self.assertEqual(_data(got), _data(want), context)

    def test_values_equal_v07(self):
        """2,000 seeded Gaussian-rational inputs per operation: the public value function (the lean path) gives the
        value that Fraction arithmetic gives, and the outcomes hash to what v0.7 gave."""
        for operation in OPERATIONS:
            function = getattr(mf, operation)
            with self.subTest(operation=operation):
                for i in range(COUNT):
                    args = arguments(operation, i)
                    want = expected(operation, args, i)
                    if want is None:
                        with self.assertRaises(OperationError, msg=(operation, args)):
                            function(*args)
                    else:
                        self.assertTrue(_same(function(*args), want, operation), (operation, i, args))
                self.assertEqual(digest(operation), V07_DIGESTS[operation], "the outcomes differ from v0.7's")

    def test_sqrt_of_three_plus_four_i(self):
        root = mf.complex_sqrt(mf.ComplexAlgebraic(3, 4))
        self.assertEqual(root, mf.ComplexAlgebraic(2, 1))
        for source, want in ((mf.ComplexAlgebraic(-4, 0), (0, 2)), (mf.ComplexAlgebraic(3, -4), (2, -1)),
                             (mf.ComplexAlgebraic(-3, -4), (1, -2)), (mf.ComplexAlgebraic(-3, 4), (1, 2)),
                             (mf.ComplexAlgebraic(0, 2), (1, 1)), (mf.ComplexAlgebraic(0, -2), (1, -1)),
                             (mf.ComplexAlgebraic(0, 0), (0, 0)), (mf.ComplexAlgebraic(9, 0), (3, 0)),
                             (mf.ComplexAlgebraic(mf.Rational(-9, 4), 0), (0, mf.Rational(3, 2)))):
            with self.subTest(source=source):
                self.assertEqual(mf.complex_sqrt(source), mf.ComplexAlgebraic(*want))

    def test_lean_equals_structured(self):
        from mathforge import lean, operations
        for operation in OPERATIONS:
            function = getattr(lean.functions, operation)
            structured = getattr(operations, operation)
            with self.subTest(operation=operation):
                for i in range(0, COUNT, 20):
                    args = arguments(operation, i)
                    want = outcome(lambda: structured(*args))
                    got = function(*args)
                    if want[0] == "value":
                        self.assertIsNot(got, lean.FALLBACK, (operation, i, args))
                        self.assertSameValue(got, want[1], (operation, i, args))
                    else:
                        self.assertIs(got, lean.FALLBACK, (operation, i, args))
                # The type edges, in each argument in turn: the lean path answers FALLBACK where the structured
                # path fails, and never differs where it succeeds.
                args = arguments(operation, 3)
                for position in range(len(args)):
                    for edge in TYPE_EDGES:
                        edged = (*args[:position], edge, *args[position + 1:])
                        want = outcome(lambda: structured(*edged))
                        got = function(*edged)
                        self.assertEqual(want[0], "error", (operation, position, edge))
                        self.assertIs(got, lean.FALLBACK, (operation, position, edge))
                        self.assertEqual(outcome(lambda: getattr(mf, operation)(*edged)), want)

    def test_non_gaussian_operands_take_the_generic_runner(self):
        """An operand with an irrational part is no Gaussian one: the answer is still the structured path's."""
        from mathforge import lean, operations
        irrational = mf.ComplexAlgebraic(mf.sqrt(2), 1)
        gauss = mf.ComplexAlgebraic(mf.Rational(1, 2), 3)
        cases = [("complex_add", (irrational, gauss)), ("complex_sub", (gauss, irrational)),
                 ("complex_mul", (irrational, gauss)), ("complex_div", (gauss, irrational)),
                 ("complex_conjugate", (irrational,)), ("complex_real", (irrational,)),
                 ("complex_imag", (irrational,)), ("complex_norm_squared", (irrational,)),
                 ("complex_sqrt", (mf.ComplexAlgebraic(1, 1),)), ("complex_sqrt", (mf.ComplexAlgebraic(3, 0),)),
                 ("complex_pow", (irrational, 3)), ("complex_algebraic", (mf.sqrt(2), 1)),
                 ("complex_add", (mf.sqrt(2), 1)), ("quadratic_roots", (1, 0, 2)), ("quadratic_roots", (1, -4, 1)),
                 ("quadratic_roots", (2, 3, 7))]
        for operation, args in cases:
            with self.subTest(operation=operation, args=args):
                want = outcome(lambda: getattr(operations, operation)(*args))
                self.assertEqual(want[0], "value")
                got = getattr(lean.functions, operation)(*args)
                self.assertIsNot(got, lean.FALLBACK)
                self.assertSameValue(got, want[1])

    def test_errors_stay_the_structured_paths(self):
        from mathforge import lean
        z = mf.ComplexAlgebraic(1, 2)
        zero = mf.ComplexAlgebraic(0, 0)
        cases = [("complex_div", (z, 0)), ("complex_div", (z, zero)), ("complex_div", (z, mf.Rational(0))),
                 ("complex_pow", (zero, -1)), ("complex_pow", (0, -3)), ("quadratic_roots", (0, 2, 3)),
                 ("quadratic_roots", (mf.Rational(0), 2, 3)), ("quadratic_roots", (1, 2, 3.5)),
                 ("complex_pow", (z, 1.5)), ("complex_pow", (z, True)), ("complex_mul", (z, 2.5)),
                 ("complex_add", (mf.Symbol("x"), z))]
        for operation, args in cases:
            with self.subTest(operation=operation, args=args):
                self.assertIs(getattr(lean.functions, operation)(*args), lean.FALLBACK)
                with self.assertRaises(OperationError):
                    getattr(mf, operation)(*args)

    def test_limits_fall_back(self):
        """An operand past max_integer_bits, a result past it, and a power whose answer would be far past it all go
        to the structured path, which raises v0.7's error."""
        from mathforge import lean
        huge, edge = 1 << 140_000, 1 << 65_535  # 140,001 bits, and 65,536 bits: exactly the limit.
        z = mf.ComplexAlgebraic(3, 4)
        cases = [("complex_algebraic", (huge, 0)), ("complex_add", (huge, z)), ("complex_sub", (z, huge)),
                 ("complex_mul", (huge, z)), ("complex_div", (z, huge)), ("complex_conjugate", (huge,)),
                 ("complex_real", (huge,)), ("complex_imag", (huge,)), ("complex_norm_squared", (huge,)),
                 ("complex_sqrt", (huge,)), ("complex_pow", (huge, 2)), ("complex_pow", (z, huge)),
                 ("complex_pow", (z, 10 ** 6)), ("complex_pow", (mf.Rational(3, 5), 100000)),
                 ("quadratic_roots", (huge, 2, 3)), ("complex_add", (edge, edge)), ("complex_mul", (edge, 2))]
        for operation, args in cases:
            with self.subTest(operation=operation):
                self.assertIs(getattr(lean.functions, operation)(*args), lean.FALLBACK)
                with self.assertRaises(OperationError) as raised:
                    getattr(mf, operation)(*args)
                self.assertEqual(raised.exception.code, "resource_limit_exceeded")
        # Exactly the limit is no error: the lean path answers, as the structured path does.
        for operation, args in (("complex_add", (edge, 0)), ("complex_algebraic", (edge, edge)),
                                ("complex_conjugate", (edge,)), ("complex_real", (edge,))):
            with self.subTest(operation=operation, limit="exact"):
                got = getattr(lean.functions, operation)(*args)
                self.assertIsNot(got, lean.FALLBACK)
                self.assertEqual(got, getattr(mf.operations, operation)(*args).unwrap())

    def test_active_budget_falls_back(self):
        from mathforge import lean
        from mathforge.limits import computation
        z = mf.ComplexAlgebraic(3, 4)
        calls = {"complex_algebraic": (3, 4), "complex_add": (z, z), "complex_sub": (z, z), "complex_mul": (z, z),
                 "complex_div": (z, z), "complex_conjugate": (z,), "complex_real": (z,), "complex_imag": (z,),
                 "complex_norm_squared": (z,), "complex_sqrt": (z,), "complex_pow": (z, 3),
                 "quadratic_roots": (1, 2, 5)}
        for operation, args in calls.items():
            with self.subTest(operation=operation):
                self.assertIsNot(getattr(lean.functions, operation)(*args), lean.FALLBACK)
                with computation():
                    self.assertIs(getattr(lean.functions, operation)(*args), lean.FALLBACK)

    def test_special_values(self):
        z = mf.ComplexAlgebraic(mf.Rational(1, 2), mf.Rational(-3, 4))
        self.assertEqual(mf.complex_pow(mf.ComplexAlgebraic(0, 0), 0), mf.ComplexAlgebraic(1, 0))
        self.assertEqual(mf.complex_pow(mf.ComplexAlgebraic(0, 0), 5), mf.ComplexAlgebraic(0, 0))
        self.assertEqual(mf.complex_pow(z, 0), mf.ComplexAlgebraic(1, 0))
        self.assertEqual(mf.complex_pow(z, 1), z)
        self.assertEqual(mf.complex_pow(mf.ComplexAlgebraic(0, 1), 10 ** 30 + 3), mf.ComplexAlgebraic(0, -1))
        self.assertEqual(mf.complex_pow(z, -2) * mf.complex_pow(z, 2), mf.ComplexAlgebraic(1, 0))
        self.assertEqual(mf.complex_div(z, z), mf.ComplexAlgebraic(1, 0))
        self.assertEqual(mf.complex_mul(mf.ComplexAlgebraic(1, 1), mf.ComplexAlgebraic(1, -1)),
                         mf.ComplexAlgebraic(2, 0))
        self.assertEqual(mf.complex_real(7), mf.Rational(7))
        self.assertEqual(mf.complex_imag(mf.Rational(1, 3)), mf.Rational(0))
        self.assertEqual(mf.complex_norm_squared(mf.ComplexAlgebraic(3, 4)), mf.Rational(25))
        roots = mf.quadratic_roots(1, 2, 1)
        self.assertEqual(roots, (mf.ComplexRoot(mf.ComplexAlgebraic(-1, 0), 2),))
        self.assertEqual([root.root for root in mf.quadratic_roots(-1, 0, 1)],
                         [mf.ComplexAlgebraic(-1, 0), mf.ComplexAlgebraic(1, 0)])
        self.assertEqual([root.root for root in mf.quadratic_roots(-2, 0, -8)],
                         [mf.ComplexAlgebraic(0, -2), mf.ComplexAlgebraic(0, 2)])

    def test_wrong_answers_are_refused(self):
        """Verification is never lazy: a wrong answer from the producing half is not returned."""
        from mathforge import lean
        from mathforge.lean import algebra_complex as module
        z, w = mf.ComplexAlgebraic(mf.Rational(1, 2), 3), mf.ComplexAlgebraic(2, mf.Rational(-5, 3))
        sound = {"complex_add": (z, w), "complex_sub": (z, w), "complex_mul": (z, w), "complex_div": (z, w),
                 "complex_norm_squared": (z,), "complex_sqrt": (mf.ComplexAlgebraic(3, 4),),
                 "quadratic_roots": (1, 2, 5), "complex_conjugate": (z,), "complex_pow": (z, 3)}
        for operation, args in sound.items():
            self.assertIsNot(getattr(lean.functions, operation)(*args), lean.FALLBACK, operation)
        # The reduction of a quotient of integers is off by one in the numerator: every check that reads it refuses.
        reduced = module._reduced
        with patch.object(module, "_reduced", lambda n, d: (reduced(n, d)[0] + 1, reduced(n, d)[1])):
            for operation in ("complex_add", "complex_sub", "complex_mul", "complex_div", "complex_norm_squared",
                              "quadratic_roots"):
                with self.subTest(wrong="_reduced", operation=operation):
                    self.assertIs(getattr(lean.functions, operation)(*sound[operation]), lean.FALLBACK)
        # A conjugate whose imaginary part keeps its sign.
        build = module._complex
        with patch.object(module, "_complex", lambda xn, xd, yn, yd: build(xn, xd, yn + 1, yd)):
            self.assertIs(lean.functions.complex_conjugate(z), lean.FALLBACK)
        # A power that the left-to-right recomputation does not reproduce, for a positive and a negative exponent.
        power = module._gaussian_power
        with patch.object(module, "_gaussian_power", lambda a, b, m: (power(a, b, m)[0] + 1, power(a, b, m)[1])):
            for n in (3, -3):
                with self.subTest(wrong="_gaussian_power", n=n):
                    self.assertIs(lean.functions.complex_pow(z, n), lean.FALLBACK)
        # A root that does not square to the operand, and a root that squares to it on the wrong branch.
        exact = module._exact_root
        three_plus_four_i = mf.ComplexAlgebraic(3, 4)
        with patch.object(module, "_exact_root", lambda n, d: (lambda r: r and (r[0] + 1, r[1]))(exact(n, d))):
            self.assertIs(lean.functions.complex_sqrt(three_plus_four_i), lean.FALLBACK)
        with patch.object(module, "_exact_root", lambda n, d: (lambda r: r and (-r[0], r[1]))(exact(n, d))):
            self.assertIs(lean.functions.complex_sqrt(three_plus_four_i), lean.FALLBACK)  # -2 - i: no principal root
        # Two roots in the wrong order.
        check = module._quadratic_check
        with patch.object(module, "_quadratic_check",
                          lambda an, ad, bn, bd, cn, cd, roots, m: check(an, ad, bn, bd, cn, cd, roots[::-1], m)):
            self.assertIs(lean.functions.quadratic_roots(1, 2, 5), lean.FALLBACK)


if __name__ == "__main__":
    unittest.main()
