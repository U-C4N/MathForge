"""The independent checker of mathforge.areas.series.power_series (mathforge.certify.series.power_series), tested on its
own before the family's producer exists.

Every true claim is computed here, never by the producer and never by the checker's evaluator: by closed forms of the
textbooks (1/k! for exp, Im((1 + i)^k)/k! for sin(x)*exp(x), binomial series), and for the other functions, seeded
compositions among them, by a short reference of this module. It composes truncated power series over Q
(fractions.Fraction) with the Maclaurin series of each function about the value of its argument at the center: 0 for
exp, the trigonometric and hyperbolic functions, atan and asin, 1 for log, q^2 for sqrt, any value but 0 for powers,
quotients and abs. That is not the checker's method, the first-order ODE recurrences of
mathforge.certify.series.taylor, and test_reference_agrees_with_identities checks the reference against identities
that it does not build in. Every claim carries the evidence () of the card. The checker answers False to every
corrupted claim and to every claim outside the card's scope, never with an exception, and refuses an oversized forgery
within FAST seconds under the default budget. ResourceLimitError is expected only for an order above 4096, which the
card makes a resource limit, for work past the budget, and for arguments beyond the budget (an integer longer than
max_integer_bits, an expression deeper than MAX_DEPTH), which run() and the verifier refuse before any checker runs.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from fractions import Fraction
import inspect
from math import comb, factorial, isqrt
import random
import time
from types import MappingProxyType
from unittest.mock import patch

import mathforge as mf
from mathforge import contracts, registry
from mathforge import functions as f
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.series import power_series as family
from mathforge.certify.series import power_series as checker
from mathforge.certify.series import taylor
from mathforge.contracts import CheckStatus
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Assumption, Rational, Reals, Symbol, Truth, trusted_rational
from mathforge.polynomial import Polynomial
from mathforge.rational_function import RationalFunction
from mathforge.registry import Argument, OperationSpec

from .. import support

FAST = 2.0  # Seconds: an oversized forgery is refused well within this under the default budget.
OPERATION = "series_coefficients"
SCOPE = "exact rationals and real functions; certified closed forms of sums, series and trigonometric integrals"
SUMMARY = "Compute the exact power series coefficients of a real function at a rational center up to an order."
X, Y = Symbol("x"), Symbol("y")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.
LONG = 2 ** 60_000 + 1  # Within max_integer_bits, and far longer than any coefficient of the true claims below.
F0, F1 = Fraction(0), Fraction(1)
T = ("x",)  # The tree of the variable; reference() describes the trees.
VANISHING = ("exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan", "asin")  # Expanded about an argument of 0.

Number = Fraction | int
Tree = tuple
Series = tuple[Fraction, ...]


class Ratio(Rational):
    """A Rational subclass: equal in value to a Rational, but no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


class Liar(str):
    """A str subclass that compares equal to everything: a Symbol forged with it would equal every Symbol."""
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


Evidence = type("Evidence", (tuple,), {})  # A tuple subclass, for the evidence or the coefficients.


# ---------------------------------------------------------------------------------------------------------------
# The reference: every true claim of the tests comes from it or from a closed form.

def fraction(number: Number | Rational) -> Fraction:
    """A Fraction of an int, a Fraction or a mathforge Rational (which Fraction itself does not take)."""
    if type(number) is Rational:
        return Fraction(number.numerator, number.denominator)
    return Fraction(number)


def rational(number: Number | Rational) -> Rational:
    number = fraction(number)
    return Rational(number.numerator, number.denominator)


def exact(number: Number | Rational) -> int | Rational:
    """An argument of the kind "rational": an int stays an int, anything else becomes a Rational."""
    return number if type(number) is int else rational(number)


def constant(value: Number, n: int) -> Series:
    return (Fraction(value),) + (F0,) * (n - 1)


def times(a: Series, b: Series) -> Series:
    """The product of two truncated series of one length, truncated to it."""
    out = [F0] * len(a)
    for i, u in enumerate(a):
        if u:
            for j in range(len(a) - i):
                out[i + j] += u * b[j]
    return tuple(out)


def compose(table: Sequence[Fraction], v: Series) -> Series:
    """sum_k table[k] v^k for a truncated series v without constant term: v^k starts at t^k, so the sum stops at the
    length of v."""
    assert v[0] == 0
    out, power_k = [F0] * len(v), constant(1, len(v))
    for coefficient in table[:len(v)]:
        out = [o + coefficient * p for o, p in zip(out, power_k)]
        power_k = times(power_k, v)
    return tuple(out)


def binomial(r: Fraction | int, k: int) -> Fraction:
    """The binomial coefficient of r over k, r (r - 1) ... (r - k + 1)/k!, for any rational r."""
    out = F1
    for i in range(k):
        out = out * (r - i) / (i + 1)
    return out


def zigzag(n: int) -> list[int]:
    """Euler's zigzag numbers E_0, ..., E_{n-1} (1, 1, 1, 2, 5, 16, 61, 272, ...) by the Seidel-Entringer triangle:
    E(m, 0) = 0 and E(m, k) = E(m, k - 1) + E(m - 1, m - k) for m >= 1, E_m = E(m, m). The odd ones are the tangent
    numbers: tan t = sum over odd k of E_k t^k/k!."""
    row, out = [1], [1]
    for m in range(1, n):
        new = [0]
        for k in range(1, m + 1):
            new.append(new[-1] + row[m - k])
        row = new
        out.append(row[-1])
    return out[:n]


def maclaurin(name: str, n: int) -> list[Fraction]:
    """The first n Maclaurin coefficients of exp, sin, cos, sinh, cosh, tan, tanh, atan and asin at 0, and of log
    about 1, that is of log(1 + t)."""
    zig = zigzag(n)
    table = []
    for k in range(n):
        odd, sign = k % 2 == 1, (-1) ** (k // 2)
        if name == "exp":
            c = Fraction(1, factorial(k))
        elif name in ("sin", "cos", "sinh", "cosh"):
            wanted = odd if name in ("sin", "sinh") else not odd
            c = Fraction(sign if name in ("sin", "cos") else 1, factorial(k)) if wanted else F0
        elif name in ("tan", "tanh"):
            c = Fraction(zig[k] * (sign if name == "tanh" else 1), factorial(k)) if odd else F0
        elif name == "atan":
            c = Fraction(sign, k) if odd else F0
        elif name == "asin":
            c = Fraction(comb(k - 1, (k - 1) // 2), 4 ** ((k - 1) // 2) * k) if odd else F0
        elif name == "log":
            c = Fraction((-1) ** (k + 1), k) if k else F0
        else:
            raise ValueError(name)
        table.append(c)
    return table


def square_root(value: Fraction) -> Fraction:
    """The positive rational square root of a positive Fraction that has one."""
    p, q = isqrt(value.numerator), isqrt(value.denominator)
    assert value > 0 and p * p == value.numerator and q * q == value.denominator, value
    return Fraction(p, q)


def power(u: Series, m: int) -> Series:
    """u^m for an int m: by the binomial series about u[0] when u[0] != 0 (a negative m needs that), by repeated
    products when u[0] = 0, where u^m starts at t^m."""
    n, head = len(u), u[0]
    if head:
        return compose([binomial(m, k) * head ** (m - k) for k in range(n)], (F0,) + tuple(u[1:]))
    assert m >= 0
    if m >= n:
        return constant(0, n)
    out = constant(1, n)
    for _ in range(m):
        out = times(out, u)
    return out


def reference(tree: Tree, center: Number, n: int) -> Series:
    """The first n Taylor coefficients at the center of the function that a tree describes, by composition with
    Maclaurin series. A tree is ("x",), ("c", rational), (op, tree, tree) for add, mul and div, ("pow", tree, int), or
    (name, tree) for exp, sin, cos, tan, sinh, cosh, tanh, atan, asin, log, sqrt and abs; every inner value at the
    center must be one about which its function is expanded (AssertionError otherwise)."""
    op = tree[0]
    if op == "x":
        return (Fraction(center), F1, *([F0] * (n - 2)))[:n]
    if op == "c":
        return constant(tree[1], n)
    if op in ("add", "mul", "div"):
        a, b = reference(tree[1], center, n), reference(tree[2], center, n)
        if op == "add":
            return tuple(p + q for p, q in zip(a, b))
        return times(a, b if op == "mul" else power(b, -1))
    if op == "pow":
        return power(reference(tree[1], center, n), tree[2])
    u = reference(tree[1], center, n)
    head, v = u[0], (F0,) + u[1:]
    if op == "abs":
        assert head != 0
        return u if head > 0 else tuple(-c for c in u)
    if op == "log":
        assert head == 1
        return compose(maclaurin("log", n), v)
    if op == "sqrt":
        q = square_root(head)
        return compose([binomial(Fraction(1, 2), k) * q ** (1 - 2 * k) for k in range(n)], v)
    assert head == 0, (op, head)
    return compose(maclaurin(op, n), u)


def expression(tree: Tree) -> f.RealExpr:
    """The RealExpr of a tree, built with the operators of mathforge.functions."""
    op = tree[0]
    if op == "x":
        return f.as_real_expr(X)
    if op == "c":
        return f.as_real_expr(rational(tree[1]))
    if op in ("add", "mul", "div"):
        a, b = expression(tree[1]), expression(tree[2])
        return a + b if op == "add" else a * b if op == "mul" else a / b
    if op == "pow":
        return expression(tree[1]) ** tree[2]
    return getattr(f, op)(expression(tree[1]))


def real(tree: Tree) -> f.RealFunction:
    """The RealFunction of a tree in the variable X: real_function simplifies the expression and keeps its domain."""
    return f.real_function(expression(tree), (X,))


def parsed(text: str) -> f.RealFunction:
    return mf.parse_function(text, symbols={"x": X})


def value_at(tree: Tree, center: Number) -> Fraction:
    return reference(tree, center, 1)[0]


def shifted(tree: Tree, center: Number, target: Number) -> Tree:
    """The tree plus a constant, so that its value at the center is target."""
    offset = Fraction(target) - value_at(tree, center)
    return tree if offset == 0 else ("add", tree, ("c", offset))


def random_tree(rng: random.Random, center: Number, depth: int) -> Tree:
    """A seeded tree of at most depth levels of functions, every node in the card's scope at the center."""
    if depth == 0 or rng.random() < 0.2:
        return T if rng.random() < 0.7 else ("c", Fraction(rng.randint(-4, 4), rng.randint(1, 3)))
    child = random_tree(rng, center, depth - 1)
    kind = rng.choice(("add", "mul", "div", "pow", "abs", "log", "sqrt") + VANISHING)
    nonzero = Fraction(rng.choice((-3, -1, 1, 2)), rng.choice((1, 2)))
    if kind in ("add", "mul"):
        return (kind, child, random_tree(rng, center, depth - 1))
    if kind == "div":
        return ("div", random_tree(rng, center, depth - 1), shifted(child, center, nonzero))
    if kind == "pow":
        k = rng.choice((-2, -1, 0, 2, 3))
        return ("pow", shifted(child, center, nonzero) if k < 0 else child, k)
    if kind == "abs":
        return ("abs", shifted(child, center, nonzero))
    if kind == "log":
        return ("log", shifted(child, center, 1))
    if kind == "sqrt":
        return ("sqrt", shifted(child, center, Fraction(rng.choice((1, 2, 3)), rng.choice((1, 2))) ** 2))
    return (kind, shifted(child, center, 0))


def sin_exp(n: int) -> list[Fraction]:
    """The first n coefficients of sin(x) exp(x) = Im(exp((1 + i) x)) at 0: Im((1 + i)^k)/k! = 2^(k/2) sin(k pi/4)/k!,
    the card's formula."""
    out, re, im = [], 1, 0
    for k in range(n):
        out.append(Fraction(im, factorial(k)))
        re, im = re - im, re + im  # Times 1 + i.
    return out


def claim(function: object, center: Number | Rational, order: int, coefficients: Sequence[Number | Rational],
          variable: Symbol = X) -> tuple[dict[str, object], family.PowerSeries]:
    """(args, value) of series_coefficients: the center as an argument stays an int or becomes a Rational, the record
    holds it as a Rational."""
    args = dict(function=function, variable=variable, center=exact(center), order=order)
    return args, family.PowerSeries(variable, rational(center), tuple(rational(c) for c in coefficients))


def true_claim(tree: Tree, center: Number, order: int) -> tuple[dict[str, object], family.PowerSeries]:
    """The claim on the function of a tree, its coefficients by the reference."""
    return claim(real(tree), center, order, reference(tree, center, order + 1))


def node(**fields: object) -> f.RealExpr:
    """A RealExpr made with object.__new__, whose fields RealExpr itself would refuse."""
    return support.forged(f.RealExpr, **fields)


def function_of(expression_: object, domain: object = f.TRUE, variables: object = (X,)) -> f.RealFunction:
    """A RealFunction made with object.__new__: its domain is exactly the given one, and nothing walks its tree."""
    return support.forged(f.RealFunction, expression=expression_, variables=variables, domain=domain)


def claimed(args: dict[str, object], value: object, evidence: tuple = ()) -> mf.Result:
    """The v6 Result that operations_v6.run builds for a claim, made by hand, since the family has no producer yet."""
    request = contracts.OperationRequest(OPERATION, tuple(args.items()), 6)
    step = contracts.Step("v6." + OPERATION, (request,), (value, evidence), relation=contracts.StepRelation.EVALUATION)
    return contracts.Result(execution_status=contracts.ExecutionStatus.COMPLETED, outcome=contracts.Outcome.VALUE,
                            value=value, exactness=contracts.Exactness.EXACT,
                            completeness=contracts.Completeness.COMPLETE, method=OPERATION, request=request,
                            for_=args["variable"], domain=Reals, steps=(step,))


class PowerSeriesCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def accept(self, args: dict[str, object], value: object) -> None:
        self.assertAccepts(OPERATION, args, value, ())

    def assertRejectsAll(self, cases: Sequence[tuple[str, object, object, object]]) -> None:
        """The checker answers False to each (label, args, value, evidence) case."""
        for label, args, value, evidence in cases:
            with self.subTest(case=label):
                self.assertRejects(OPERATION, args, value, evidence)

    # -----------------------------------------------------------------------------------------------------------
    # The family module

    def test_family_declares_the_card(self):
        spec = OperationSpec(OPERATION, "series",
                             (Argument("function", "RealFunction", None, "function"),
                              Argument("variable", "Symbol", None, "symbol"),
                              Argument("center", "Rational | int", None, "rational"),
                              Argument("order", "int", None, "int")),
                             "PowerSeries", SUMMARY, SCOPE, version=6)
        self.assertEqual(family.SPECS, (spec,))
        self.assertEqual(family.TYPES, {"power_series": family.PowerSeries})
        self.assertEqual(family.CHECKER, "mathforge.certify.series.power_series")
        self.assertEqual(set(family.RENDER), {family.PowerSeries})
        self.assertEqual(registry.spec(OPERATION), spec)
        self.assertIs(registry.checker(OPERATION), checker)
        self.assertIs(support.checker(family), checker)
        self.assertEqual(checker.RECORD, (family.PowerSeries.__module__, family.PowerSeries.__qualname__))
        self.assertEqual((checker.MAX_ORDER, checker.MAX_DEPTH), (family.MAX_ORDER, family.MAX_DEPTH))
        # The producer exists now: the card's first known value comes back verified by this checker.
        args = dict(function=parsed("exp(x)"), variable=X, center=0, order=4)
        expected = tuple(Rational(1, factorial(k)) for k in range(5))
        self.assertEqual(family.produce(OPERATION, args), (family.PowerSeries(X, Rational(0), expected), ()))
        self.assertEqual(support.value(OPERATION, **args), family.PowerSeries(X, Rational(0), expected))
        with self.assertRaises(UnsupportedOperation):
            family.produce("series_coefficient", args)

    def test_records_hold_the_canonical_form(self):
        value = family.PowerSeries(X, Rational(1, 2), (Rational(1), Rational(0), Rational(-3, 4)))
        self.assertEqual((value.variable, value.center, value.coefficients[2]), (X, Rational(1, 2), Rational(-3, 4)))
        one = (Rational(1),)
        for label, fields in (
                ("an int center", dict(variable=X, center=1, coefficients=one)),
                ("a bool center", dict(variable=X, center=True, coefficients=one)),
                ("a float center", dict(variable=X, center=0.5, coefficients=one)),
                ("a Fraction center", dict(variable=X, center=Fraction(1, 2), coefficients=one)),
                ("a str variable", dict(variable="x", center=Rational(0), coefficients=one)),
                ("a RealExpr variable", dict(variable=f.as_real_expr(X), center=Rational(0), coefficients=one)),
                ("no coefficient", dict(variable=X, center=Rational(0), coefficients=())),
                ("a list of coefficients", dict(variable=X, center=Rational(0), coefficients=[Rational(1)])),
                ("an int coefficient", dict(variable=X, center=Rational(0), coefficients=(Rational(1), 0))),
                ("a bool coefficient", dict(variable=X, center=Rational(0), coefficients=(True,))),
                ("a float coefficient", dict(variable=X, center=Rational(0), coefficients=(0.5,))),
                ("a str coefficient", dict(variable=X, center=Rational(0), coefficients=("1",))),
                ("a Rational subclass", dict(variable=X, center=Rational(0), coefficients=(Ratio(1, 2),))),
                ("a negative denominator",
                 dict(variable=X, center=Rational(0), coefficients=(trusted_rational(1, -2),))),
                ("a zero denominator", dict(variable=X, center=Rational(0), coefficients=(trusted_rational(1, 0),))),
        ):
            with self.subTest(case=label), self.assertRaises(InvalidInput):
                family.PowerSeries(**fields)
        with self.assertRaises(ResourceLimitError):
            family.PowerSeries(X, Rational(0), (Rational(GIANT),))

    def test_rendering_follows_the_card(self):
        cases = (
            (0, (1, 1, Fraction(1, 2)), "1 + x + 1/2*x^2 + O(x^3)",
             r"1 + x + \frac{1}{2}\, x^{2} + \mathcal{O}\left(x^{3}\right)"),
            (1, (0, 1, Fraction(-1, 2), Fraction(1, 3)), "(x - 1) - 1/2*(x - 1)^2 + 1/3*(x - 1)^3 + O((x - 1)^4)",
             r"\left(x - 1\right) - \frac{1}{2}\, \left(x - 1\right)^{2} + \frac{1}{3}\, \left(x - 1\right)^{3}"
             r" + \mathcal{O}\left(\left(x - 1\right)^{4}\right)"),
            (Fraction(-1, 2), (1, -1), "1 - (x + 1/2) + O((x + 1/2)^2)",
             r"1 - \left(x + \frac{1}{2}\right) + \mathcal{O}\left(\left(x + \frac{1}{2}\right)^{2}\right)"),
            (0, (0, 0, 0), "O(x^3)", r"\mathcal{O}\left(x^{3}\right)"),
            (3, (0,), "O(x - 3)", r"\mathcal{O}\left(x - 3\right)"),
            (0, (-1, 0, Fraction(-2, 3), -1), "-1 - 2/3*x^2 - x^3 + O(x^4)",
             r"-1 - \frac{2}{3}\, x^{2} - x^{3} + \mathcal{O}\left(x^{4}\right)"),
            (2, (Fraction(-7, 3), 0, 4), "-7/3 + 4*(x - 2)^2 + O((x - 2)^3)",
             r"-\frac{7}{3} + 4\, \left(x - 2\right)^{2} + \mathcal{O}\left(\left(x - 2\right)^{3}\right)"),
        )
        for center, coefficients, text, latex in cases:
            value = claim(parsed("x"), center, len(coefficients) - 1, coefficients)[1]
            with self.subTest(text=text):
                self.assertEqual(mf.render_text(value), text)
                self.assertEqual(mf.to_latex(value), latex)

    # -----------------------------------------------------------------------------------------------------------
    # The reference of these tests

    def test_reference_agrees_with_identities(self):
        """sin^2 + cos^2 = 1, cosh^2 - sinh^2 = 1, tan = sin/cos, tanh = sinh/cosh, exp(t) exp(-t) = 1,
        exp(log(1 + t)) = 1 + t, sqrt(4 + t)^2 = 4 + t, sin(asin t) = t, tan(atan t) = t, u u^-1 = 1, and the first
        terms that the textbooks list: the reference builds none of these in."""
        n = 12
        t = (F0, F1) + (F0,) * (n - 2)
        one, one_plus_t = constant(1, n), (F1, F1) + (F0,) * (n - 2)
        s = {name: compose(maclaurin(name, n), t) for name in VANISHING}

        def plus(a, b):
            return tuple(p + q for p, q in zip(a, b))

        def minus(a, b):
            return tuple(p - q for p, q in zip(a, b))

        self.assertEqual(plus(times(s["sin"], s["sin"]), times(s["cos"], s["cos"])), one)
        self.assertEqual(minus(times(s["cosh"], s["cosh"]), times(s["sinh"], s["sinh"])), one)
        self.assertEqual(s["tan"], times(s["sin"], power(s["cos"], -1)))
        self.assertEqual(s["tanh"], times(s["sinh"], power(s["cosh"], -1)))
        self.assertEqual(times(s["exp"], compose(maclaurin("exp", n), tuple(-c for c in t))), one)
        self.assertEqual(compose(maclaurin("exp", n), compose(maclaurin("log", n), t)), one_plus_t)
        self.assertEqual(power(reference(("sqrt", ("add", ("c", 4), T)), 0, n), 2), plus(constant(4, n), t))
        self.assertEqual(compose(maclaurin("sin", n), s["asin"]), t)
        self.assertEqual(compose(maclaurin("tan", n), s["atan"]), t)
        u = (Fraction(-2, 3), F1, Fraction(5, 7), F0, Fraction(-1, 2)) + (F1,) * (n - 5)
        self.assertEqual(times(u, power(u, -1)), one)
        self.assertEqual(times(u, times(u, u)), power(u, 3))
        self.assertEqual(zigzag(10), [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936])
        textbook = {
            "tan": (0, 1, 0, Fraction(1, 3), 0, Fraction(2, 15), 0, Fraction(17, 315)),
            "tanh": (0, 1, 0, Fraction(-1, 3), 0, Fraction(2, 15), 0, Fraction(-17, 315)),
            "atan": (0, 1, 0, Fraction(-1, 3), 0, Fraction(1, 5), 0, Fraction(-1, 7)),
            "asin": (0, 1, 0, Fraction(1, 6), 0, Fraction(3, 40), 0, Fraction(5, 112)),
            "sin": (0, 1, 0, Fraction(-1, 6), 0, Fraction(1, 120), 0, Fraction(-1, 5040)),
            "cosh": (1, 0, Fraction(1, 2), 0, Fraction(1, 24), 0, Fraction(1, 720), 0),
        }
        for name, terms in textbook.items():
            with self.subTest(name=name):
                self.assertEqual(s[name][:8], terms)
        self.assertEqual(maclaurin("log", 5), [0, 1, Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 4)])
        self.assertEqual(reference(T, Fraction(2, 3), 3), (Fraction(2, 3), 1, 0))
        self.assertEqual(reference(("sqrt", T), 4, 6),  # The values of tests/v08/series/test_series_arithmetic.py.
                         (2, Fraction(1, 4), Fraction(-1, 64), Fraction(1, 512), Fraction(-5, 16384),
                          Fraction(7, 131072)))
        with self.assertRaises(AssertionError):
            reference(("exp", T), 1, 3)  # exp(1) is no rational: the reference refuses what the card leaves out.

    # -----------------------------------------------------------------------------------------------------------
    # True claims, which the checker accepts

    def test_known_values_of_the_card(self):
        # exp(x) at 0, order 4, with the center as an int and as a Rational.
        args, value = claim(parsed("exp(x)"), 0, 4, (1, 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24)))
        self.assertEqual(reference(("exp", T), 0, 5), tuple(fraction(c) for c in value.coefficients))
        self.accept(args, value)
        self.accept(dict(args, center=Rational(0)), value)
        # sin(x)*exp(x) at 0, order 12, the v0.8 Taylor cliff, within a sliver of the default budget.
        expected = sin_exp(13)
        self.assertEqual(expected[:5], [0, 1, 1, Fraction(1, 3), 0])
        self.assertEqual(reference(("mul", ("sin", T), ("exp", T)), 0, 13), tuple(expected))
        args, value = claim(parsed("sin(x)*exp(x)"), 0, 12, expected)
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(OPERATION, args, value, ()), True)
        self.assertLess(budget.work, 5_000)
        # log(x) at 1, order 3.
        self.assertEqual(reference(("log", T), 1, 4), (0, 1, Fraction(-1, 2), Fraction(1, 3)))
        self.accept(*claim(parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3))))
        self.accept(*claim(parsed("log(x)"), Rational(1), 3, (0, 1, Fraction(-1, 2), Fraction(1, 3))))
        # 1/(1 - x) at 0: every coefficient is 1, at every order (40 and 300 pass the PREFIX test first).
        geometric = parsed("1/(1 - x)")
        for order in (0, 1, 2, 5, 17, 40, 300):
            with self.subTest(function="1/(1 - x)", order=order):
                self.accept(*claim(geometric, 0, order, (1,) * (order + 1)))
        # exp(x) at 1 is Unevaluated: no value passes, not even exp's coefficients at 0 or rational approximations.
        candidates = ((1, 1, Fraction(1, 2)), (Fraction(2718, 1000),) * 3, (0, 0, 0),
                      (Fraction(2718281828, 10 ** 9), Fraction(2718281828, 10 ** 9), Fraction(1359140914, 10 ** 9)))
        self.assertRejectsAll([(f"exp(x) at 1: {c}", *claim(parsed("exp(x)"), 1, 2, c), ()) for c in candidates])

    def test_every_rule_of_the_scope_is_accepted(self):
        """Each rule of the card's scope, at several centers: exp, sin, cos, tan, sinh, cosh, tanh, atan and asin of
        an argument that vanishes at the center, log of an argument equal to 1, sqrt of the square of a positive
        rational, abs of an argument other than 0, integer powers, quotients, scalars, the variable and order 0."""
        cases = [
            (("exp", ("add", T, ("c", -1))), 1, 6),
            (("sin", ("mul", ("c", 2), ("add", T, ("c", Fraction(-1, 2))))), Fraction(1, 2), 7),
            (("cos", ("mul", T, T)), 0, 8),
            (("tan", T), 0, 9),
            (("tan", ("add", T, ("c", 3))), -3, 5),
            (("sinh", ("mul", ("c", -3), T)), 0, 7),
            (("cosh", ("add", ("pow", T, 2), ("c", -4))), 2, 6),
            (("tanh", T), 0, 9),
            (("atan", ("add", ("mul", ("c", 2), T), ("c", -1))), Fraction(1, 2), 6),
            (("asin", ("div", T, ("c", 2))), 0, 7),
            (("asin", ("add", T, ("c", Fraction(2, 3)))), Fraction(-2, 3), 5),
            (("log", T), 1, 6),
            (("log", ("add", ("c", 2), ("mul", ("c", -1), T))), 1, 5),
            (("log", ("pow", T, 2)), -1, 5),
            (("log", ("div", T, ("c", 2))), 2, 4),
            (("sqrt", T), 4, 6),
            (("sqrt", T), Fraction(9, 4), 5),
            (("sqrt", T), Fraction(1, 9), 4),
            (("sqrt", ("add", ("c", 4), T)), 0, 6),
            (("sqrt", ("add", ("c", 1), ("pow", T, 2))), 0, 8),
            (("abs", T), -1, 3),
            (("abs", T), 3, 3),
            (("abs", ("add", T, ("c", -2))), 0, 4),
            (("abs", ("add", ("c", 1), ("mul", ("c", -1), ("pow", T, 2)))), 2, 4),
            (("pow", T, 3), 2, 5),
            (("pow", T, -2), 1, 6),
            (("pow", ("add", ("c", 1), ("mul", ("c", -1), T)), -3), 0, 6),
            (("pow", T, 5), 0, 3),
            (("pow", ("sin", T), 0), 0, 3),
            (("div", T, ("add", T, ("c", 1))), 1, 6),
            (("div", ("add", ("pow", T, 2), ("c", 1)), ("add", T, ("c", -3))), 1, 5),
            (("c", Fraction(-7, 3)), 5, 4),
            (T, Fraction(-2, 3), 3),
            (("exp", T), 0, 0),
            (("log", T), 1, 0),
            (("div", ("sin", ("mul", T, ("exp", T))), ("sqrt", ("add", ("c", 1), ("tan", T)))), 0, 8),
        ]
        for tree, center, order in cases:
            with self.subTest(tree=tree, center=center, order=order):
                args, value = true_claim(tree, center, order)
                self.accept(args, value)
                if type(center) is int:  # The same claim with the center as a Rational argument.
                    self.accept(dict(args, center=Rational(center)), value)
        closed = [  # Closed forms of the textbooks, against the reference.
            (("abs", T), -1, (1, -1, 0, 0)),
            (("pow", T, 3), 2, (8, 12, 6, 1, 0, 0)),
            (("pow", T, -2), 1, (1, -2, 3, -4, 5, -6, 7)),
            (("pow", ("add", ("c", 1), ("mul", ("c", -1), T)), -3), 0, (1, 3, 6, 10, 15, 21, 28)),
            (("pow", T, 5), 0, (0, 0, 0, 0)),
            (("pow", ("sin", T), 0), 0, (1, 0, 0, 0)),
            (("c", Fraction(-7, 3)), 5, (Fraction(-7, 3), 0, 0, 0, 0)),
            (("sqrt", T), Fraction(9, 4), (Fraction(3, 2), Fraction(1, 3), Fraction(-1, 27))),
            (("log", ("pow", T, 2)), -1, (0, -2, -1)),
        ]
        for tree, center, terms in closed:
            with self.subTest(closed_form=tree, center=center):
                self.assertEqual(reference(tree, center, len(terms)), tuple(Fraction(c) for c in terms))
                self.accept(*claim(real(tree), center, len(terms) - 1, terms))

    def test_seeded_compositions_are_accepted(self):
        """Seeded trees of up to three levels of functions at several centers: the checker accepts the reference's
        coefficients and refuses them with one coefficient changed."""
        rng = random.Random(20261112)
        centers = (0, 0, 1, -1, 2, Fraction(1, 2), Fraction(-2, 3))
        for index in range(150):
            center = rng.choice(centers)
            tree = random_tree(rng, center, rng.randint(1, 3))
            order = rng.randint(0, 8)
            args, value = true_claim(tree, center, order)
            with self.subTest(index=index, tree=tree, center=center, order=order):
                self.accept(args, value)
                k = rng.randrange(order + 1)
                wrong = value.coefficients[:k] + (value.coefficients[k] + 1,) + value.coefficients[k + 1:]
                self.assertRejects(OPERATION, args, family.PowerSeries(X, value.center, wrong))

    def test_shared_subtrees_count_once(self):
        """A node shared by several parents is validated and expanded once: (1 + x)^(2^60) as 60 nested squares of
        one node each, a tree of 2^60 paths, is accepted at once; so is sin(x)*sin(x) + sin(x) with one sin node."""
        x = f.as_real_expr(X)
        shared = x + 1
        for _ in range(60):
            shared = f.RealExpr("mul", (shared, shared))
        start = time.perf_counter()
        self.accept(*claim(function_of(shared), 0, 5, [binomial(2 ** 60, k) for k in range(6)]))
        self.accept(*claim(function_of(shared), -1, 3, (0, 0, 0, 0)))  # x^(2^60) at 0: nothing up to order 3.
        self.assertLess(time.perf_counter() - start, FAST)
        s = f.sin(x)
        tree = ("add", ("mul", ("sin", T), ("sin", T)), ("sin", T))
        self.accept(*claim(f.real_function(s * s + s, (X,)), 0, 6, reference(tree, 0, 7)))

    def test_domains_that_hold_near_the_center(self):
        """A domain holds near the center when each of its atoms holds there with a strict sign: a narrowed domain,
        the excluded points of a RationalFunction away from the center, and the atoms of tan and asin."""
        x = f.as_real_expr(X)
        positive = f.real_function(2 * x, (X,), domain=f.condition("positive", x))
        self.assertEqual(positive.domain, f.DomainPredicate("positive", (x,)))
        self.accept(*claim(positive, 1, 3, (2, 2, 0, 0)))
        self.accept(*claim(positive, Fraction(1, 5), 2, (Fraction(2, 5), 2, 0)))
        self.accept(*claim(f.real_function(x, (X,), domain=f.DomainPredicate("nonnegative", (x,))), 3, 2, (3, 1, 0)))
        removable = self.removable()
        self.accept(*claim(removable, 0, 3, (1, 1, 0, 0)))
        self.accept(*claim(removable, 2, 2, (3, 1, 0)))
        self.accept(*claim(parsed("(x**2 - 1)/(x - 1)"), 0, 3, (1, 1, 0, 0)))
        for tree in (("tan", T), ("asin", T), ("tan", ("asin", ("mul", ("c", Fraction(1, 2)), T)))):
            with self.subTest(tree=tree):
                self.assertNotEqual(real(tree).domain, f.TRUE)
                self.accept(*true_claim(tree, 0, 7))
        # A nested "and", "true" among its operands and one atom twice: not the canonical form, and still a domain.
        atom = f.DomainPredicate("positive", (x + 1,))
        nested = f.DomainPredicate("and", (f.DomainPredicate("and", (atom, f.DomainPredicate("nonzero", (x + 2,)))),
                                           f.TRUE, atom))
        self.accept(*claim(function_of(x, nested), 0, 1, (0, 1)))

    def removable(self) -> f.RealFunction:
        """(x^2 - 1)/(x - 1) as a RationalFunction made a RealFunction: x + 1 with the domain nonzero(x - 1)."""
        one = Polynomial((Rational(1),), X)
        quotient = (RationalFunction(Polynomial((Rational(-1), Rational(0), Rational(1)), X), one)
                    / RationalFunction(Polynomial((Rational(-1), Rational(1)), X), one))
        function = f.real_function(quotient)
        self.assertEqual(str(function.expression), "(1 + x)")
        self.assertEqual(function.domain.kind, "nonzero")
        return function

    # -----------------------------------------------------------------------------------------------------------
    # Claims without a value, which the checker refuses

    def test_rejects_domains_that_fail_at_the_center(self):
        """A function has no power series at a center outside the interior of its domain, also where its expression
        would have one: the checker requires a strict sign of every atom at the center."""
        x = f.as_real_expr(X)
        positive = f.real_function(2 * x, (X,), domain=f.condition("positive", x))
        nonnegative = f.real_function(x, (X,), domain=f.DomainPredicate("nonnegative", (x,)))
        removable = self.removable()
        nested = f.DomainPredicate("and", (f.DomainPredicate("positive", (x + 1,)),
                                           f.DomainPredicate("and", (f.TRUE, f.DomainPredicate("nonzero", (x,))))))
        cases = [
            ("x > 0 at 0", positive, 0, (0, 2, 0, 0)),
            ("x > 0 at -1", positive, -1, (-2, 2, 0)),
            ("x >= 0 at 0, a domain on one side", nonnegative, 0, (0, 1, 0)),
            ("x + 1 where x != 1, at 1", removable, 1, (2, 1, 0)),
            ("(x^2 - 1)/(x - 1) at 1", parsed("(x**2 - 1)/(x - 1)"), 1, (2, 1, 0)),
            ("sin(x)/x at 0, the series of its limit", parsed("sin(x)/x"), 0, (1, 0, Fraction(-1, 6), 0)),
            ("0/x at 0, 0 where x != 0", parsed("0/x"), 0, (0, 0, 0)),
            ("the domain false", f.RealFunction(x, (X,), f.FALSE), 0, (0, 1)),
            ("a false atom", function_of(x, f.DomainPredicate("positive", (f.as_real_expr(-1),))), 0, (0, 1)),
            ("a nested atom that fails", function_of(x, nested), 0, (0, 1)),
            ("an atom out of scope", function_of(x, f.DomainPredicate("positive", (f.exp(x),))), 1, (1, 1)),
            ("an atom of another symbol", function_of(x, f.DomainPredicate("nonzero", (f.as_real_expr(Y),))), 0,
             (0, 1)),
        ]
        self.assertEqual(parsed("0/x").expression, f.ZERO)
        self.assertRejectsAll([(label, *claim(function, center, len(c) - 1, c), ()) for label, function, center, c
                               in cases])

    def test_rejects_functions_outside_the_scope(self):
        """The card leaves these Unevaluated, so no value passes, whatever it claims: a node whose series at the
        center is not rational, or whose function is not analytic there."""
        x = f.as_real_expr(X)
        keeps_product = f.RealFunction(f.RealExpr("mul", (f.ZERO, f.exp(x))), (X,))
        self.assertEqual(f.real_function(f.ZERO * f.exp(x), (X,)).expression, f.ZERO)  # Simplified to 0.
        root = f.real_function(f.as_real_expr(RealAlgebraicRoot((-2, 1), 0)) * x, (X,))
        cases = [
            ("exp(x) at 1", parsed("exp(x)"), 1, (1, 1, Fraction(1, 2))),
            ("exp(1 + x) at 0", parsed("exp(1 + x)"), 0, (1, 1, Fraction(1, 2))),
            ("cos(x) at 1", parsed("cos(x)"), 1, (Fraction(1, 2), 0, 0)),
            ("tan(x) at 1", parsed("tan(x)"), 1, (Fraction(3, 2), 0, 0)),
            ("atan(x) at 1, pi/4 there", parsed("atan(x)"), 1, (Fraction(3, 4), Fraction(1, 2), Fraction(-1, 4))),
            ("asin(x) at 1, the end of its domain", parsed("asin(x)"), 1, (Fraction(3, 2), 0, 0)),
            ("log(x) at 2", parsed("log(x)"), 2, (0, Fraction(1, 2), Fraction(-1, 8))),
            ("sqrt(x) at 2", parsed("sqrt(x)"), 2, (1, 1, 1)),
            ("sqrt(x) at 0", parsed("sqrt(x)"), 0, (0, 0, 0)),
            ("sqrt(x**2) at 0, which is abs(x)", parsed("sqrt(x**2)"), 0, (0, 0, 0)),
            ("abs(x) at 0", parsed("abs(x)"), 0, (0, 1, 0)),
            ("abs(x) at 0, the other side", parsed("abs(x)"), 0, (0, -1, 0)),
            ("1/x at 0", parsed("1/x"), 0, (0, 0, 0)),
            ("x**-1 at 0", parsed("x**-1"), 0, (0, 0)),
            ("acos(x) at 0", parsed("acos(x)"), 0, (Fraction(3, 2), -1, 0)),
            ("acos(x) at 1", parsed("acos(x)"), 1, (0, 0, 0)),
            ("x + pi", parsed("x + pi"), 0, (3, 1)),
            ("E*x", parsed("E*x"), 0, (0, 3)),
            ("sqrt(2)*x", f.real_function(f.sqrt(2) * x, (X,)), 0, (0, 1)),
            ("a RealAlgebraicRoot scalar, even one equal to 2", root, 0, (0, 2)),
            ("0*exp(x) at 1, in a tree that keeps the product", keeps_product, 1, (0, 0)),
        ]
        self.assertRejectsAll([(label, *claim(function, center, len(c) - 1, c), ()) for label, function, center, c
                               in cases])

    def test_rejects_claims_on_invalid_inputs(self):
        """Inputs that the card makes invalid have no value: an order below 0, and a variable that is not the
        function's only variable. An order above 4096 is the card's resource limit."""
        exp = parsed("exp(x)")
        args, value = claim(exp, 0, 2, (1, 1, Fraction(1, 2)))
        self.accept(args, value)
        same_name = Symbol("x")
        assumed = Symbol("x", assumptions=(Assumption("positive", Truth.TRUE),), uid=X.uid)
        cases = [
            ("order -1", dict(args, order=-1), value),
            ("order -1, one coefficient", dict(args, order=-1), family.PowerSeries(X, Rational(0), (Rational(1),))),
            ("order -1, no coefficient", dict(args, order=-1),
             support.forged(family.PowerSeries, variable=X, center=Rational(0), coefficients=())),
            ("another variable", dict(args, variable=Y), claim(exp, 0, 2, (1, 1, Fraction(1, 2)), Y)[1]),
            ("a variable of the same name", dict(args, variable=same_name),
             claim(exp, 0, 2, (1, 1, Fraction(1, 2)), same_name)[1]),
            ("the variable with an assumption", dict(args, variable=assumed), value),
            ("a function of two variables", dict(args, function=mf.parse_function("exp(x)", symbols={"x": X, "y": Y})),
             value),
            ("a constant without variables", dict(args, function=f.real_function(Rational(5))),
             claim(exp, 0, 2, (5, 0, 0))[1]),
        ]
        self.assertRejectsAll([(label, a, v, ()) for label, a, v in cases])
        self.assertEqual(f.real_function(Rational(5)).variables, ())
        self.accept(*claim(f.real_function(Rational(5), (X,)), 0, 2, (5, 0, 0)))  # The constant as a function of x.
        # Above 4096 a value of the right length is beyond the card's limit; a value of another length is wrong.
        beyond = dict(args, order=4097)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, beyond, family.PowerSeries(X, Rational(0), (Rational(1),) * 4098))
        self.assertRejects(OPERATION, beyond, value)

    # -----------------------------------------------------------------------------------------------------------
    # Corrupted claims, which the checker refuses

    def test_rejects_corrupted_values(self):
        args, value = claim(parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3)))
        self.accept(args, value)
        c, one, record, forged = value.coefficients, Rational(1), family.PowerSeries, support.forged

        def with_coefficient(k: int, item: object) -> tuple:
            return c[:k] + (item,) + c[k + 1:]

        cases: list[tuple[str, object, object, object]] = [
            (f"coefficient {k} plus 1", args, record(X, one, with_coefficient(k, c[k] + 1)), ()) for k in range(4)]
        cases += [(label, args, item, ()) for label, item in (
            # Wrong values.
            ("coefficients reversed", record(X, one, c[::-1])),
            ("the signs flipped", record(X, one, tuple(-v for v in c))),
            ("the zero series", record(X, one, (Rational(0),) * 4)),
            ("another center", record(X, Rational(2), c)),
            ("the center 0, as log(1 + x) at 0 would have it", record(X, Rational(0), c)),
            ("another variable", record(Y, one, c)),
            ("a variable of the same name", record(Symbol("x"), one, c)),
            ("one more coefficient, the true one", record(X, one, c + (Rational(-1, 4),))),
            ("one coefficient fewer", record(X, one, c[:-1])),
            # Mathematically right, but not in the canonical form.
            ("an int center", forged(record, variable=X, center=1, coefficients=c)),
            ("a center not in lowest terms", record(X, trusted_rational(2, 2), c)),
            ("a Fraction center", forged(record, variable=X, center=Fraction(1), coefficients=c)),
            ("an int coefficient", forged(record, variable=X, center=one, coefficients=with_coefficient(1, 1))),
            ("a coefficient not in lowest terms", record(X, one, with_coefficient(2, trusted_rational(-2, 4)))),
            ("0 not in lowest terms", record(X, one, with_coefficient(0, trusted_rational(0, 3)))),
            ("a negative denominator",
             forged(record, variable=X, center=one, coefficients=with_coefficient(2, trusted_rational(1, -2)))),
            ("a Fraction coefficient",
             forged(record, variable=X, center=one, coefficients=with_coefficient(2, Fraction(-1, 2)))),
            ("a Rational subclass",
             forged(record, variable=X, center=one, coefficients=with_coefficient(2, Ratio(-1, 2)))),
            # Another class, a forged record, or a bool, a float, a str or None where a number belongs.
            ("an impostor record", support.impostor(value)),
            ("a record with no field", forged(record)),
            ("a record without coefficients", forged(record, variable=X, center=one)),
            ("a record without a center", forged(record, variable=X, coefficients=c)),
            ("a record without a variable", forged(record, center=one, coefficients=c)),
            ("coefficients as a list", forged(record, variable=X, center=one, coefficients=list(c))),
            ("coefficients as a tuple subclass", forged(record, variable=X, center=one, coefficients=Evidence(c))),
            ("no coefficient", forged(record, variable=X, center=one, coefficients=())),
            ("a str variable", forged(record, variable="x", center=one, coefficients=c)),
            ("a RealExpr variable", forged(record, variable=f.as_real_expr(X), center=one, coefficients=c)),
            ("a bool coefficient", forged(record, variable=X, center=one, coefficients=with_coefficient(1, True))),
            ("a float coefficient", forged(record, variable=X, center=one, coefficients=with_coefficient(1, 1.0))),
            ("a str coefficient", forged(record, variable=X, center=one, coefficients=with_coefficient(1, "1"))),
            ("None as a coefficient", forged(record, variable=X, center=one, coefficients=with_coefficient(1, None))),
            ("an int-subclass coefficient",
             forged(record, variable=X, center=one, coefficients=with_coefficient(1, Integer(1)))),
            ("a bool center", forged(record, variable=X, center=True, coefficients=c)),
            ("a float center", forged(record, variable=X, center=1.0, coefficients=c)),
            ("a str center", forged(record, variable=X, center="1", coefficients=c)),
            ("None as the value", None),
            ("the bare coefficients", c),
            ("the coefficients as a Polynomial in x - 1", Polynomial(c, X)),
        )]
        self.assertRejectsAll(cases)
        # Past PREFIX coefficients, a wrong value is refused by the whole series too, wherever it is wrong.
        geometric, n = parsed("1/(1 - x)"), checker.PREFIX + 8
        for k in (0, 5, checker.PREFIX - 1, checker.PREFIX, n - 1):
            ones = [1] * n
            ones[k] = 2
            with self.subTest(order=n - 1, wrong=k):
                self.assertRejects(OPERATION, *claim(geometric, 0, n - 1, ones))

    def test_rejects_corrupted_evidence_operations_and_arguments(self):
        args, value = claim(parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3)))
        c = value.coefficients
        self.accept(args, value)
        evidence_cases = [
            ("evidence as a list", []), ("evidence None", None), ("evidence as a str", "()"),
            ("evidence as a tuple subclass", Evidence()), ("an empty item", ((),)), ("the coefficients", (c,)),
            ("the value itself", (value,)), ("a zero", (0,)), ("an empty str", ("",)),
            ("a certificate-like pair", (("series", c),)), ("the coefficients unwrapped", c),
        ]
        self.assertRejectsAll([(label, args, value, evidence) for label, evidence in evidence_cases])
        function = args["function"]
        argument_cases = [
            ("arguments as pairs", tuple(args.items())),
            ("arguments as a list of pairs", list(args.items())),
            ("arguments as a read-only mapping", MappingProxyType(args)),
            ("None as arguments", None),
            ("the order missing", {k: v for k, v in args.items() if k != "order"}),
            ("an extra argument", dict(args, point=1)),
            ("function_taylor's names", dict(expression=function, variable=X, point=1, order=3)),
            ("the center renamed point", dict(function=function, variable=X, point=1, order=3)),
            ("order True", dict(args, order=True)), ("order 3.0", dict(args, order=3.0)),
            ("order '3'", dict(args, order="3")), ("order None", dict(args, order=None)),
            ("an int-subclass order", dict(args, order=Integer(3))),
            ("a Rational order", dict(args, order=Rational(3))),
            ("center True", dict(args, center=True)), ("center 1.0", dict(args, center=1.0)),
            ("center '1'", dict(args, center="1")), ("center None", dict(args, center=None)),
            ("a Fraction center", dict(args, center=Fraction(1))),
            ("an int-subclass center", dict(args, center=Integer(1))),
            ("a Rational-subclass center", dict(args, center=Ratio(1))),
            ("a center not in lowest terms", dict(args, center=trusted_rational(3, 3))),
            ("a str variable", dict(args, variable="x")),
            ("a RealExpr variable", dict(args, variable=f.as_real_expr(X))),
            ("None as the variable", dict(args, variable=None)),
            ("the RealExpr instead of the function", dict(args, function=function.expression)),
            ("a str function", dict(args, function="log(x)")), ("None as the function", dict(args, function=None)),
            ("an impostor function", dict(args, function=support.impostor(function))),
            ("variables as a list", dict(args, function=function_of(function.expression, function.domain, [X]))),
            ("no variables", dict(args, function=function_of(function.expression, function.domain, ()))),
        ]
        self.assertRejectsAll([(label, a, value, ()) for label, a in argument_cases])
        # Rationals not in lowest terms in the arguments, matched by the value: the evaluator would carry them along.
        three_thirds = trusted_rational(3, 3)
        half = node(op="scalar", args=(), value=trusted_rational(2, 4))
        self.assertRejectsAll([
            ("a center 3/3 matched by the value", dict(args, function=parsed("x"), center=three_thirds, order=1),
             support.forged(family.PowerSeries, variable=X, center=three_thirds, coefficients=(three_thirds,
                                                                                                 Rational(1))), ()),
            ("a scalar 2/4 matched by the value", dict(args, function=function_of(half), center=0, order=0),
             family.PowerSeries(X, Rational(0), (trusted_rational(2, 4),)), ()),
        ])
        self.accept(dict(args, function=function_of(node(op="scalar", args=(), value=Rational(1, 2))), center=0,
                         order=0), family.PowerSeries(X, Rational(0), (Rational(1, 2),)))
        # A Symbol forged with str fields that compare equal to everything would pass for every symbol.
        liar = support.forged(Symbol, name=Liar("x"), domain=Reals, assumptions=(), uid=Liar(X.uid))
        self.assertEqual(liar, Y)  # What the checker must not rely on.
        exp_args, exp_value = claim(parsed("exp(x)"), 0, 2, (1, 1, Fraction(1, 2)))
        self.accept(exp_args, exp_value)
        lying_tree = function_of(f.exp(f.RealExpr("symbol", (), liar)))
        self.assertRejectsAll([
            ("a lying variable for a function of y",
             dict(exp_args, function=mf.parse_function("exp(y)", symbols={"y": Y}), variable=liar), exp_value, ()),
            ("a lying symbol in the tree", dict(exp_args, function=lying_tree), exp_value, ()),
            ("a lying variable in the value", exp_args, family.PowerSeries(liar, Rational(0), exp_value.coefficients),
             ()),
        ])
        for operation in ("series_coefficient", "Series_coefficients", "series_coefficients ", "function_taylor",
                          "taylor", "sum_polynomial", "", None, 7, (OPERATION,), b"series_coefficients", [OPERATION]):
            with self.subTest(operation=operation):
                self.assertRejects(operation, args, value)

    def test_rejects_malformed_functions(self):
        """Trees and domains that no RealFunction holds (forged with object.__new__), and shapes out of scope: the
        checker answers False, never with an exception."""
        x = f.as_real_expr(X)
        cyclic = node(op="add", args=(), value=None)
        object.__setattr__(cyclic, "args", (cyclic, x))
        first, second = node(op="sin", args=(), value=None), node(op="sin", args=(), value=None)
        object.__setattr__(first, "args", (second,))
        object.__setattr__(second, "args", (first,))
        expressions = [
            ("an op that is no str", node(op=1, args=(), value=None)),
            ("args as a list", node(op="sin", args=[x], value=None)),
            ("an unknown op", node(op="erf", args=(x,), value=None)),
            ("acos", f.acos(x)), ("pi", f.pi), ("E", f.E),
            ("a float scalar", node(op="scalar", args=(), value=0.5)),
            ("an int scalar", node(op="scalar", args=(), value=1)),
            ("a bool scalar", node(op="scalar", args=(), value=True)),
            ("a scalar not in lowest terms", node(op="scalar", args=(), value=trusted_rational(2, 4))),
            ("a scalar with a negative denominator", node(op="scalar", args=(), value=trusted_rational(1, -2))),
            ("a scalar with an operand", node(op="scalar", args=(x,), value=Rational(1))),
            ("another symbol", f.as_real_expr(Y)),
            ("a symbol of the same name", f.as_real_expr(Symbol("x"))),
            ("a str symbol", node(op="symbol", args=(), value="x")),
            ("a symbol with an operand", node(op="symbol", args=(x,), value=X)),
            ("add of three", node(op="add", args=(x, x, x), value=None)),
            ("add of one", node(op="add", args=(x,), value=None)),
            ("add with a value", node(op="add", args=(x, x), value=Rational(1))),
            ("sin of two", node(op="sin", args=(x, x), value=None)),
            ("sin with a value", node(op="sin", args=(x,), value=Rational(0))),
            ("pow with a float exponent", node(op="pow", args=(x,), value=2.0)),
            ("pow with a bool exponent", node(op="pow", args=(x,), value=True)),
            ("pow of two", node(op="pow", args=(x, x), value=2)),
            ("a str instead of a node", "x"), ("None", None), ("a Rational", Rational(0)), ("the Symbol", X),
            ("an impostor node", support.impostor(f.sin(x))),
            ("a node with no field", node()),
            ("a node below itself", cyclic),
            ("two nodes below each other", first),
            ("sin of a forged node", f.sin(node(op="erf", args=(x,), value=None))),
        ]
        cases = [(label, function_of(item), 0, (0, 1, 0)) for label, item in expressions]
        atom = f.DomainPredicate("positive", (x + 1,))
        cyclic_domain = support.forged(f.DomainPredicate, kind="and", operands=())
        object.__setattr__(cyclic_domain, "operands", (atom, cyclic_domain))
        domains = [
            ("domain None", None), ("a str domain", "true"), ("the domain False", False),
            ("an unknown kind", support.forged(f.DomainPredicate, kind="real", operands=())),
            ("a kind that is no str", support.forged(f.DomainPredicate, kind=1, operands=())),
            ("true with an operand", support.forged(f.DomainPredicate, kind="true", operands=(x,))),
            ("and of one", support.forged(f.DomainPredicate, kind="and", operands=(atom,))),
            ("operands as a list", support.forged(f.DomainPredicate, kind="nonzero", operands=[x + 1])),
            ("an atom of two", support.forged(f.DomainPredicate, kind="positive", operands=(x + 1, x + 1))),
            ("an atom of no expression", support.forged(f.DomainPredicate, kind="positive", operands=("x",))),
            ("an atom out of scope", f.DomainPredicate("positive", (f.acos(x),))),
            ("a predicate inside itself", cyclic_domain),
            ("an impostor predicate", support.impostor(f.TRUE)),
            ("a predicate with no field", support.forged(f.DomainPredicate)),
        ]
        cases += [(label, function_of(x, domain), 0, (0, 1, 0)) for label, domain in domains]
        cases += [
            ("a function without an expression", support.forged(f.RealFunction, variables=(X,), domain=f.TRUE), 0,
             (0, 1, 0)),
            ("a function without variables", support.forged(f.RealFunction, expression=x, domain=f.TRUE), 0,
             (0, 1, 0)),
            ("a function with no field", support.forged(f.RealFunction), 0, (0, 1, 0)),
        ]
        self.assertRejectsAll([(label, *claim(function, center, len(c) - 1, c), ()) for label, function, center, c
                               in cases])
        # The true claim on the same function with a well-formed tree and domain passes; a domain left unset reads
        # the class default, TRUE.
        self.accept(*claim(function_of(x, atom), 0, 2, (0, 1, 0)))
        self.accept(*claim(support.forged(f.RealFunction, expression=x, variables=(X,)), 0, 2, (0, 1, 0)))

    # -----------------------------------------------------------------------------------------------------------
    # Sizes, work and budget

    def test_oversized_forgeries_fail_fast(self):
        args, value = claim(parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3)))
        c, record, forged = value.coefficients, family.PowerSeries, support.forged
        long = Rational(LONG)
        tanh_args = dict(function=parsed("tanh(x)"), variable=X, center=0, order=1024)
        tanh_wrong = tuple(rational(t) for t in reference(("tanh", T), 0, 8)[:3]) + (Rational(2, 3),)
        tanh_wrong += (Rational(0),) * (1025 - len(tanh_wrong))
        cases = [
            ("evidence of 100,000 items", args, value, (0,) * 100_000),
            ("evidence holding a giant integer", args, value, (GIANT,)),
            ("100,000 coefficients", args, record(X, Rational(1), (Rational(0),) * 100_000), ()),
            ("a giant numerator", args, forged(record, variable=X, center=Rational(1), coefficients=c[:3] + (
                Rational(GIANT),)), ()),
            ("a giant denominator", args, forged(record, variable=X, center=Rational(1), coefficients=c[:3] + (
                Rational(1, GIANT),)), ()),
            ("a giant center", args, forged(record, variable=X, center=Rational(GIANT), coefficients=c), ()),
            ("an order of 10**9 with four coefficients", dict(args, order=10 ** 9), value, ()),
            ("exp(x) at order 4096, long coefficients after the first",
             dict(function=parsed("exp(x)"), variable=X, center=0, order=4096),
             record(X, Rational(0), (Rational(1),) + (long,) * 4096), ()),
            ("tanh(x) at order 1024, wrong in its fourth coefficient", tanh_args, record(X, Rational(0), tanh_wrong),
             ()),
        ]
        for label, a, v, e in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects(OPERATION, a, v, e)  # Under the default budget.
                self.assertLess(time.perf_counter() - start, FAST)
        # An order above 4096 with a value of that length: the card's resource limit, at once.
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, dict(args, order=10 ** 5),
                            record(X, Rational(1), (Rational(0),) * (10 ** 5 + 1)))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_arguments_beyond_the_budget(self):
        """run() and the verifier refuse these requests before any checker runs; called directly, the checker runs
        out of budget at once (or answers False), and never accepts."""
        x = f.as_real_expr(X)
        chain = x
        for _ in range(300):
            chain = f.sin(chain)
        cases = [
            ("a giant center", dict(function=parsed("exp(x)"), variable=X, center=GIANT, order=1),
             support.forged(family.PowerSeries, variable=X, center=Rational(GIANT), coefficients=(Rational(1),) * 2)),
            ("a giant Rational center", dict(function=parsed("exp(x)"), variable=X, center=Rational(1, GIANT), order=1),
             support.forged(family.PowerSeries, variable=X, center=Rational(1, GIANT),
                            coefficients=(Rational(1),) * 2)),
            ("a giant scalar", dict(function=function_of(x + node(op="scalar", args=(), value=Rational(GIANT))),
                                    variable=X, center=0, order=1), claim(parsed("x"), 0, 1, (0, 1))[1]),
            ("a giant exponent", dict(function=function_of(node(op="pow", args=(x,), value=GIANT)), variable=X,
                                      center=0, order=1), claim(parsed("x"), 0, 1, (0, 0))[1]),
            ("an expression 301 nodes deep", dict(function=function_of(chain), variable=X, center=0, order=1),
             claim(parsed("x"), 0, 1, (0, 1))[1]),
        ]
        for label, a, v in cases:
            with self.subTest(case=label):
                start = time.perf_counter()
                self.assertRejects(OPERATION, a, v, may_exhaust=True)
                self.assertLess(time.perf_counter() - start, FAST)

    def test_depth_limit(self):
        """MAX_DEPTH nodes along the deepest path pass; one more is beyond the budget (truncated_series recurses per
        level), which run() and the verifier reach only with a raised max_depth."""
        x = f.as_real_expr(X)
        chain, tree = x, T
        for _ in range(checker.MAX_DEPTH - 1):
            chain, tree = f.sin(chain), ("sin", tree)
        expected = reference(tree, 0, 4)
        self.assertEqual(expected, (0, 1, 0, Fraction(-(checker.MAX_DEPTH - 1), 6)))  # sin^k(x) = x - k x^3/6 + ...
        self.accept(*claim(function_of(chain), 0, 3, expected))
        deeper = claim(function_of(f.sin(chain)), 0, 3, reference(("sin", tree), 0, 4))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, *deeper)

    def test_work_is_charged_before_it_runs(self):
        """The ODE loops of truncated_series step over zero coefficients without charging, so the checker pays for
        them first; the charges depend on the claim alone; a heavy claim under a small budget runs out at once."""
        order = 2000
        args, value = claim(parsed("exp(x)"), 0, order, [Fraction(1, factorial(k)) for k in range(order + 1)])
        with computation(fresh=True) as budget:
            self.assertIs(checker.check_operation(OPERATION, args, value, ()), True)
        work, length = budget.work, order + 1
        self.assertGreaterEqual(work, length * (length - 1) // 2 // checker.LOOP_ITERATIONS_PER_UNIT)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, args, value, limits=ComputationLimits(max_work=work - 1))
        self.assertAccepts(OPERATION, args, value, limits=ComputationLimits(max_work=work))
        # tanh(x) at order 600 with its first PREFIX coefficients right: the whole series is needed, and refused.
        head = reference(("tanh", T), 0, checker.PREFIX)
        heavy = claim(parsed("tanh(x)"), 0, 600, head + (F0,) * (601 - checker.PREFIX))
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, *heavy, limits=ComputationLimits(max_work=50_000))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_long_coefficients_are_charged_by_their_length(self):
        """Regression: a Rational result used to cost one unit whatever its length, so a true claim with long
        coefficients kept the checker busy for tens of seconds within the default budget (a sum of 1000 nodes
        1/(x - c), c of 4000 bits, at order 15: accepted after 23 s for 653,353 units). Each series operation is now
        charged by the 64-bit words of its operands (trusted.series.operation_work), so that claim runs out of the
        default budget at once, and the same function with four nodes still verifies."""
        size = 4000
        rng = random.Random(1)
        c = Fraction(rng.getrandbits(size) | 1 << (size - 1), rng.getrandbits(size) | 1 << (size - 1))
        x, one, minus_c = f.as_real_expr(X), f.as_real_expr(Rational(1)), f.as_real_expr(rational(-c))

        def poles(count: int) -> f.RealFunction:  # count separate nodes 1/(x - c), summed as a balanced tree
            terms = [f.RealExpr("div", (one, f.RealExpr("add", (x, minus_c)))) for _ in range(count)]
            while len(terms) > 1:
                terms = [f.RealExpr("add", (terms[i], terms[i + 1])) if i + 1 < len(terms) else terms[i]
                         for i in range(0, len(terms), 2)]
            return f.RealFunction(terms[0], (X,))

        order = 15  # 1/(x - c) = -sum x^k / c^(k + 1): coefficients of up to 64,000 bits.
        self.accept(*claim(poles(4), 0, order, [-4 / c ** (k + 1) for k in range(order + 1)]))
        heavy = claim(poles(1000), 0, order, [-1000 / c ** (k + 1) for k in range(order + 1)])
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, *heavy)
        self.assertLess(time.perf_counter() - start, FAST)
        # The charge of one operation grows with the product of the lengths of its operands.
        long = Rational(2 ** 16_384 + 1, 3 ** 10_000)  # 503 words: products stay within max_integer_bits.
        for operation in (taylor.series.add, taylor.series.mul):
            with self.subTest(operation=operation.__name__), computation(fresh=True) as budget:
                operation((long,) * 4, (long,) * 4)
                self.assertGreater(budget.work, 4 * 1_000)  # About 1,050 units for each of the four sums.
        # sqrt(1 + x) at order 2000 (binomial coefficients of about 4,000 bits): the producer answers at once, and the
        # checker used to grind for 18 s before it ran out of budget; now it runs out within FAST seconds.
        order = 2000
        coefficients = [Fraction(comb(2 * k, k), (1 - 2 * k) * (-4) ** k) for k in range(order + 1)]
        self.assertEqual(coefficients[:4], [F1, Fraction(1, 2), Fraction(-1, 8), Fraction(1, 16)])
        start = time.perf_counter()
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, OPERATION, *claim(parsed("sqrt(1 + x)"), 0, order, coefficients))
        self.assertLess(time.perf_counter() - start, FAST)

    def test_full_size_claims_fit_the_default_budget(self):
        """exp(x) and cos(x) at order 4096, the card's largest order, verify under the default budget."""
        n = 4097
        exp_coefficients = [Fraction(1, factorial(k)) for k in range(n)]
        cos_coefficients = [Fraction((-1) ** (k // 2), factorial(k)) if k % 2 == 0 else F0 for k in range(n)]
        self.assertEqual(tuple(cos_coefficients[:12]), tuple(maclaurin("cos", 12)))
        for text, coefficients in (("exp(x)", exp_coefficients), ("cos(x)", cos_coefficients)):
            with self.subTest(function=text):
                start = time.perf_counter()
                self.accept(*claim(parsed(text), 0, n - 1, coefficients))
                self.assertLess(time.perf_counter() - start, FAST)

    # -----------------------------------------------------------------------------------------------------------
    # The checker's construction

    def test_checker_construction(self):
        """The checker imports only the limits, the model and the shared evaluator of mathforge.certify.series.taylor,
        whose public names it uses; it forms no power (pow and ** cannot be interrupted); the v0.8 names of the
        evaluator stay for tests/v08/series/test_series_arithmetic.py."""
        source = ast.parse(inspect.getsource(checker))
        imported = set()
        for item in ast.walk(source):
            if isinstance(item, ast.Import):
                imported.update(alias.name for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                imported.add("." * item.level + (item.module or ""))
        self.assertEqual(imported, {"__future__", "math", "mathforge.limits", "mathforge.model", ".taylor"})
        for module in (checker, taylor):
            with self.subTest(module=module.__name__):
                tree = ast.parse(inspect.getsource(module))
                called = {item.func.id for item in ast.walk(tree)
                          if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}
                self.assertNotIn("pow", called)
                self.assertEqual([item for item in ast.walk(tree) if isinstance(item, (ast.BinOp, ast.AugAssign))
                                  and isinstance(item.op, ast.Pow)], [])
        self.assertIs(checker.truncated_series, taylor.truncated_series)
        self.assertIs(checker.SeriesDeclined, taylor.SeriesDeclined)
        self.assertIs(taylor._series, taylor.truncated_series)
        self.assertIs(taylor._Decline, taylor.SeriesDeclined)

    def test_verify_consults_this_checker(self):
        """mf.verify runs this checker on a v6 Result, before and after the JSON round trip, without the producer; a
        wrong value fails the mathematics check."""
        cases = [
            (parsed("sin(x)*exp(x)"), 0, 12, sin_exp(13)),
            (parsed("log(x)"), 1, 3, (0, 1, Fraction(-1, 2), Fraction(1, 3))),
            (parsed("sqrt(x)"), Fraction(9, 4), 3, reference(("sqrt", T), Fraction(9, 4), 4)),
            (self.removable(), Fraction(-1, 2), 2, (Fraction(1, 2), 1, 0)),
        ]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for function, center, order, coefficients in cases:
                args, value = claim(function, center, order, coefficients)
                with self.subTest(function=str(function), center=center):
                    self.roundtrip(claimed(args, value))
                    last = value.coefficients[-1] + 1
                    wrong = family.PowerSeries(X, value.center, value.coefficients[:-1] + (last,))
                    report = support.statuses(mf.verify(claimed(args, wrong)))
                    self.assertIs(report["result.trace"], CheckStatus.VERIFIED)
                    self.assertIs(report["result.mathematics"], CheckStatus.FAILED)
