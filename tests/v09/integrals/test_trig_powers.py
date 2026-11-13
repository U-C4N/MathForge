"""The trig_powers family of mathforge.integrals: trig_power_antiderivative, trig_power_integral.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register integrals trig_powers

Every expected value comes from the card or from the references of this module, which share nothing with the
producer's reduction formula or its table: the odd powers of the sine integrate by the substitution u = cos(a*x),
S**(2m+1)*C**k = -(1/a) (1 - u**2)**m u**k du; the even ones are rewritten by power reduction, C**n = 2**-n sum
binomial(n, i) cos((n - 2i) a x), and cos(r a x) integrates to sin(r a x)/(r a) = S U_(r-1)(C)/(r a) with the
Chebyshev polynomials of the second kind. The values at the bounds use cos and sin of k*pi/12 computed here as powers
of the rotation by pi/12 in Q(sqrt 2, sqrt 3). Input trees are expanded by a reader of this module without the
reduction S**2 = 1 - C**2, and the producer's antiderivative is read back by another one. Nothing here is computed by
the producer. tests/v09/integrals/test_trig_powers_checker.py tests the checker on its own.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.areas.integrals import trig_powers as family
from mathforge.contracts import CheckStatus, ErrorInfo
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.functions import RealExpr, RealFunction, parse_function
from mathforge.limits import ComputationLimits, computation
from mathforge.model import Assumption, Integers, Mul, Rational, Symbol, Truth, add, mul, sqrt

from .. import support

FAST = 1.0  # Seconds: a request that its budget refuses fails well within this.
ANTIDERIVATIVE, INTEGRAL = "trig_power_antiderivative", "trig_power_integral"
X, Y = Symbol("x"), Symbol("y")
N = Symbol("n", Integers, (Assumption("positive", Truth.TRUE),))  # A variable with a domain and an assumption.
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # A float, a bool, a str, None and two containers.
OUTSIDE = "accepts polynomials in sin(a*x) and cos(a*x)"  # The card's text for a function outside the class.
FREQUENCIES = (Fraction(1), Fraction(-1), Fraction(2), Fraction(3), Fraction(1, 2), Fraction(-3, 4), Fraction(5, 3),
               Fraction(1, 6))

Number = Fraction | int
InputPoly = dict[tuple[int, int], Fraction]  # {(j, k): c}: sum c S**j C**k, S**2 not reduced.
CPoly = dict[int, Fraction]  # {n: c}: sum c C**n.
NormalForm = tuple[Fraction, CPoly, CPoly]  # (lambda, N0, N1): lambda*x + N0(C) + S*N1(C).
Element = tuple[Fraction, Fraction, Fraction, Fraction]  # c0 + c1 sqrt 2 + c2 sqrt 3 + c3 sqrt 6.


class Ratio(Rational):
    """A Rational subclass: no exact Rational."""
    __slots__ = ()


class Integer(int):
    """An int subclass, equal to an exact int but none."""


# ---------------------------------------------------------------------------------------------------------------
# Trees.

def q(number: Number | Rational) -> Rational:
    if type(number) is Rational:
        return number
    number = Fraction(number)
    return Rational(number.numerator, number.denominator)


def scalar(number: Number | Rational) -> RealExpr:
    return RealExpr("scalar", (), q(number))


def leaf(variable: Symbol = X) -> RealExpr:
    return RealExpr("symbol", (), variable)


def node(op: str, *args: RealExpr, value: object = None) -> RealExpr:
    return RealExpr(op, args, value)


def argument(a: Fraction, style: str = "mul", variable: Symbol = X) -> RealExpr:
    """A tree that denotes a*x, written in one of several styles."""
    x = leaf(variable)
    if style == "plain" and a == 1:
        return x
    if style == "div":  # x / (1/a)
        return node("div", node("mul", scalar(a.numerator), x), scalar(a.denominator))
    if style == "right":  # x * a
        return node("mul", x, scalar(a))
    if style == "detour":  # (a + 1)*x - x, with a power 0 that the class admits inside an argument.
        return node("add", node("mul", scalar(a + 1), x), node("mul", scalar(-1), node("pow", x, value=1)))
    return node("mul", scalar(a), x)


def balanced(terms: list[RealExpr]) -> RealExpr:
    """The sum of the terms as a balanced tree; 0 for none."""
    while len(terms) > 1:
        terms = [node("add", terms[i], terms[i + 1]) if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0] if terms else scalar(0)


def poly_tree(poly: InputPoly, u: RealExpr, rng: random.Random | None = None) -> RealExpr:
    """sum c sin(u)**j cos(u)**k, a term per entry; with rng, the powers are written in varied ways."""
    sine, cosine = node("sin", u), node("cos", u)
    terms = []
    for (j, k), c in sorted(poly.items()):
        factors = []
        for base, e in ((sine, j), (cosine, k)):
            if not e:
                continue
            if rng is not None and e > 1 and rng.random() < 0.3:
                factors.extend([base] * e)  # Repeated factors instead of a power.
            else:
                factors.append(base if e == 1 and (rng is None or rng.random() < 0.7) else node("pow", base, value=e))
        monomial = factors[0] if factors else None
        for item in factors[1:]:
            monomial = node("mul", monomial, item)
        if monomial is None:
            terms.append(scalar(c))
        elif rng is not None and c.denominator > 1 and rng.random() < 0.3:
            terms.append(node("div", node("mul", scalar(c.numerator), monomial), scalar(c.denominator)))
        else:
            terms.append(node("mul", scalar(c), monomial))
    return balanced(terms)


def function(expression: RealExpr, variable: Symbol = X) -> RealFunction:
    return RealFunction(expression, (variable,))


def text(source: str) -> RealFunction:
    return parse_function(source, symbols={"x": X})


def random_poly(rng: random.Random, degree: int, terms: int = 4) -> InputPoly:
    poly: InputPoly = {}
    for _ in range(terms):
        j = rng.randrange(degree + 1)
        k = rng.randrange(degree + 1 - j)
        c = Fraction(rng.randint(-20, 20), rng.randint(1, 9))
        if c:
            poly[(j, k)] = poly.get((j, k), Fraction(0)) + c
    return {key: c for key, c in poly.items() if c}


def random_tree(rng: random.Random, u: RealExpr, depth: int) -> RealExpr:
    """A random tree of add, mul, div by a nonzero constant, small powers, constants, sin(u) and cos(u)."""
    if depth == 0 or rng.random() < 0.25:
        choice = rng.random()
        if choice < 0.4:
            return node("sin", u)
        if choice < 0.8:
            return node("cos", u)
        return scalar(Fraction(rng.randint(-6, 6), rng.randint(1, 4)))
    op = rng.choice(("add", "add", "mul", "mul", "div", "pow"))
    if op == "div":
        divisor = scalar(Fraction(rng.choice((-3, -1, 2, 5)), rng.randint(1, 3)))
        return node("div", random_tree(rng, u, depth - 1), divisor)
    if op == "pow":
        return node("pow", random_tree(rng, u, depth - 1), value=rng.choice((0, 1, 2, 2, 3)))
    return node(op, random_tree(rng, u, depth - 1), random_tree(rng, u, depth - 1))


# ---------------------------------------------------------------------------------------------------------------
# References: every expected value comes from these or from the card.

def p_add(p: InputPoly, r: InputPoly) -> InputPoly:
    out = dict(p)
    for key, c in r.items():
        out[key] = out.get(key, Fraction(0)) + c
    return {key: c for key, c in out.items() if c}


def p_mul(p: InputPoly, r: InputPoly) -> InputPoly:
    out: InputPoly = {}
    for (j1, k1), c in p.items():
        for (j2, k2), d in r.items():
            out[(j1 + j2, k1 + k2)] = out.get((j1 + j2, k1 + k2), Fraction(0)) + c * d
    return {key: c for key, c in out.items() if c}


def expand_input(tree: RealExpr) -> tuple[InputPoly, Fraction | None]:
    """An input tree as sum c S**j C**k (without the reduction S**2 = 1 - C**2) and the a of its arguments, read
    recursively; the arguments are read as a*x by the same reader in x."""
    found: list[Fraction] = []

    def linear(arg: RealExpr) -> Fraction:
        def walk(t: RealExpr) -> dict[int, Fraction]:
            if t.op == "scalar":
                return {0: Fraction(t.value.numerator, t.value.denominator)} if t.value.numerator else {}
            if t.op == "symbol":
                return {1: Fraction(1)}
            if t.op in ("add", "mul"):
                left, right = walk(t.args[0]), walk(t.args[1])
                out: dict[int, Fraction] = {}
                if t.op == "add":
                    for part in (left, right):
                        for n, c in part.items():
                            out[n] = out.get(n, Fraction(0)) + c
                else:
                    for n, c in left.items():
                        for m, d in right.items():
                            out[n + m] = out.get(n + m, Fraction(0)) + c * d
                return {n: c for n, c in out.items() if c}
            if t.op == "div":
                d = t.args[1].value
                return {n: c * Fraction(d.denominator, d.numerator) for n, c in walk(t.args[0]).items()}
            assert t.op == "pow", t.op
            out = {0: Fraction(1)}
            for _ in range(t.value):
                base = walk(t.args[0])
                out = {n + m: c * d for n, c in out.items() for m, d in base.items()}
            return {n: c for n, c in out.items() if c}

        poly = walk(arg)
        assert set(poly) == {1}, poly
        return poly[1]

    def walk(t: RealExpr) -> InputPoly:
        if t.op == "scalar":
            return {(0, 0): Fraction(t.value.numerator, t.value.denominator)} if t.value.numerator else {}
        if t.op in ("sin", "cos"):
            found.append(linear(t.args[0]))
            return {(1, 0) if t.op == "sin" else (0, 1): Fraction(1)}
        if t.op == "add":
            return p_add(walk(t.args[0]), walk(t.args[1]))
        if t.op == "mul":
            return p_mul(walk(t.args[0]), walk(t.args[1]))
        if t.op == "div":
            d = t.args[1].value
            return {key: c * Fraction(d.denominator, d.numerator) for key, c in walk(t.args[0]).items()}
        assert t.op == "pow", t.op
        base, out = walk(t.args[0]), {(0, 0): Fraction(1)}
        for _ in range(t.value):
            out = p_mul(out, base)
        return out

    poly = walk(tree)
    assert len(set(found)) <= 1, found
    return poly, (found[0] if found else None)


def chebyshev_u(n: int) -> list[list[Fraction]]:
    """U_0, ..., U_n as coefficient lists in C: sin((m + 1) t) = sin(t) U_m(cos t)."""
    u = [[Fraction(1)], [Fraction(0), Fraction(2)]]
    while len(u) <= n:
        following = [Fraction(0)] + [2 * c for c in u[-1]]
        for i, c in enumerate(u[-2]):
            following[i] -= c
        u.append(following)
    return u


def antiderivative(poly: InputPoly, a: Fraction | None) -> NormalForm:
    """The canonical antiderivative of sum c sin(a x)**j cos(a x)**k: substitution for odd j, power reduction and
    Chebyshev polynomials for even j, no constant term."""
    slope, n0, n1 = Fraction(0), {}, {}
    for (j, k), c in poly.items():
        m = j // 2
        expanded = {2 * i + k: Fraction(comb(m, i) * (-1) ** i) for i in range(m + 1)}  # (1 - u**2)**m u**k
        for n, d in expanded.items():
            if j % 2:  # -(1/a) * u**(n+1)/(n+1)
                n0[n + 1] = n0.get(n + 1, Fraction(0)) - c * d / ((n + 1) * a)
                continue
            u = chebyshev_u(n)
            for i in range(n + 1):
                r, w = abs(n - 2 * i), c * d * comb(n, i) / 2 ** n
                if r == 0:
                    slope += w
                    continue
                for e, coefficient in enumerate(u[r - 1]):
                    if coefficient:
                        n1[e] = n1.get(e, Fraction(0)) + w * coefficient / (r * a)
    return slope, {n: c for n, c in n0.items() if c}, {n: c for n, c in n1.items() if c}


def e_add(p: Element, r: Element) -> Element:
    return tuple(x + y for x, y in zip(p, r))


def e_scale(p: Element, c: Fraction) -> Element:
    return tuple(x * c for x in p)


def e_mul(p: Element, r: Element) -> Element:
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = r
    return (a0 * b0 + 2 * a1 * b1 + 3 * a2 * b2 + 6 * a3 * b3,
            a0 * b1 + a1 * b0 + 3 * (a2 * b3 + a3 * b2),
            a0 * b2 + a2 * b0 + 2 * (a1 * b3 + a3 * b1),
            a0 * b3 + a3 * b0 + a1 * b2 + a2 * b1)


ZERO_E: Element = (Fraction(0),) * 4
ONE_E: Element = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))


def rotations() -> list[tuple[Element, Element]]:
    """(cos(k pi/12), sin(k pi/12)) for k = 0..23, as powers of the rotation by pi/12: cos(pi/12) = (sqrt 6 + sqrt 2)/4,
    sin(pi/12) = (sqrt 6 - sqrt 2)/4."""
    c1 = (Fraction(0), Fraction(1, 4), Fraction(0), Fraction(1, 4))
    s1 = (Fraction(0), Fraction(-1, 4), Fraction(0), Fraction(1, 4))
    table = [(ONE_E, ZERO_E)]
    for _ in range(23):
        c, s = table[-1]
        table.append((e_add(e_mul(c, c1), e_scale(e_mul(s, s1), Fraction(-1))), e_add(e_mul(s, c1), e_mul(c, s1))))
    return table


ROTATIONS = rotations()


def trig_value(n0: CPoly, n1: CPoly, k: int) -> Element:
    """N0(C) + S*N1(C) at the angle k*pi/12."""
    c, s = ROTATIONS[k % 24]
    total, power = ZERO_E, ONE_E
    for n in range(max(list(n0) + list(n1), default=-1) + 1):
        total = e_add(total, e_scale(power, n0.get(n, Fraction(0))))
        total = e_add(total, e_scale(e_mul(s, power), n1.get(n, Fraction(0))))
        power = e_mul(power, c)
    return total


def integral(poly: InputPoly, a: Fraction | None, lower: Fraction, upper: Fraction) -> tuple[Element, Fraction]:
    """(algebraic part, pi coefficient) of the integral from lower*pi to upper*pi."""
    slope, n0, n1 = antiderivative(poly, a)
    if not n0 and not n1:
        return ZERO_E, slope * (upper - lower)
    k_lower, k_upper = 12 * a * lower, 12 * a * upper
    assert k_lower.denominator == 1 and k_upper.denominator == 1
    high, low = trig_value(n0, n1, int(k_upper)), trig_value(n0, n1, int(k_lower))
    return e_add(high, e_scale(low, Fraction(-1))), slope * (upper - lower)


def pi_linear(element: Element, pi_coefficient: Fraction) -> family.PiLinear:
    c0, c1, c2, c3 = (q(c) for c in element)
    return family.PiLinear(add(c0, mul(c1, sqrt(2)), mul(c2, sqrt(3)), mul(c3, sqrt(6))), q(pi_coefficient))


def read_antiderivative(value: RealFunction, variable: Symbol = X) -> tuple[NormalForm, Fraction | None]:
    """The producer's F read back as (lambda, N0, N1) and its a; asserts the shape of the module docstring: every sine
    and cosine has the argument x (a = 1) or mul(scalar(a), x), and S**2 is reduced away."""
    found: list[Fraction] = []
    NF = dict[tuple[int, int, int], Fraction]

    def times(p: NF, r: NF) -> NF:
        out: NF = {}
        for (i1, j1, k1), c in p.items():
            for (i2, j2, k2), d in r.items():
                terms = [((i1 + i2, j1 + j2, k1 + k2), c * d)]
                if j1 + j2 == 2:
                    terms = [((i1 + i2, 0, k1 + k2), c * d), ((i1 + i2, 0, k1 + k2 + 2), -c * d)]
                for key, e in terms:
                    out[key] = out.get(key, Fraction(0)) + e
        return {key: c for key, c in out.items() if c}

    def walk(t: RealExpr) -> NF:
        if t.op == "scalar":
            return {(0, 0, 0): Fraction(t.value.numerator, t.value.denominator)} if t.value.numerator else {}
        if t.op == "symbol":
            assert t.value == variable
            return {(1, 0, 0): Fraction(1)}
        if t.op in ("sin", "cos"):
            arg = t.args[0]
            if arg.op == "symbol":
                assert arg.value == variable
                found.append(Fraction(1))
            else:
                assert arg.op == "mul" and arg.args[0].op == "scalar" and arg.args[1].op == "symbol", arg
                assert arg.args[1].value == variable and arg.args[0].value != q(1)
                found.append(Fraction(arg.args[0].value.numerator, arg.args[0].value.denominator))
            return {(0, 1, 0) if t.op == "sin" else (0, 0, 1): Fraction(1)}
        if t.op == "add":
            out = dict(walk(t.args[0]))
            for key, c in walk(t.args[1]).items():
                out[key] = out.get(key, Fraction(0)) + c
            return {key: c for key, c in out.items() if c}
        if t.op == "mul":
            return times(walk(t.args[0]), walk(t.args[1]))
        assert t.op == "pow" and t.value >= 2, t
        base, out = walk(t.args[0]), {(0, 0, 0): Fraction(1)}
        for _ in range(t.value):
            out = times(out, base)
        return out

    nf = walk(value.expression)
    assert all(i <= 1 and j <= 1 and (i == 0 or (j, k) == (0, 0)) for i, j, k in nf), nf
    assert len(set(found)) <= 1, found
    slope = nf.pop((1, 0, 0), Fraction(0))
    n0 = {k: c for (i, j, k), c in nf.items() if j == 0}
    n1 = {k: c for (i, j, k), c in nf.items() if j == 1}
    return (slope, n0, n1), (found[0] if found else None)


# The known values of the card: (input text, lambda, N0, N1) and (input text, lower, upper, element, pi coefficient).
KNOWN_ANTIDERIVATIVES = (
    ("sin(x)^2", Fraction(1, 2), {}, {1: Fraction(-1, 2)}),
    ("cos(x)^2", Fraction(1, 2), {}, {1: Fraction(1, 2)}),
    ("sin(x)^3", Fraction(0), {1: Fraction(-1), 3: Fraction(1, 3)}, {}),
    ("sin(2*x)*cos(2*x)^2", Fraction(0), {3: Fraction(-1, 6)}, {}),
    ("sin(x)*cos(x)", Fraction(0), {2: Fraction(-1, 2)}, {}),
)
KNOWN_INTEGRALS = (
    ("sin(x)^2", 0, 1, ZERO_E, Fraction(1, 2)),
    ("cos(x)^2", 0, Fraction(1, 2), ZERO_E, Fraction(1, 4)),
    ("sin(x)^3", 0, 1, (Fraction(4, 3), Fraction(0), Fraction(0), Fraction(0)), Fraction(0)),
    ("sin(x)^2", 0, Fraction(1, 3), (Fraction(0), Fraction(0), Fraction(-1, 8), Fraction(0)), Fraction(1, 6)),
)
# One small request per operation: the SMALL_INPUTS of the card's integration notes.
SMALL = {
    ANTIDERIVATIVE: lambda: dict(function=text("sin(x)^2"), variable=X),
    INTEGRAL: lambda: dict(function=text("sin(x)^2"), variable=X, lower_pi=0, upper_pi=1),
}


def bound(number: Fraction) -> int | Rational:
    return number.numerator if number.denominator == 1 else q(number)


def random_bound(rng: random.Random, a: Fraction) -> Fraction:
    """A rational t with 12*a*t an integer."""
    return Fraction(rng.randint(-40, 40)) / (12 * a)


class TrigPowersTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object = None) -> None:
        """A completed v6 Result, with its four checks, which mf.verify accepts."""
        self.assertIsNotNone(result.value, result.error)
        if value is not None:
            self.assertEqual(result.value, value)
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for source, slope, n0, n1 in KNOWN_ANTIDERIVATIVES:
            with self.subTest(operation=ANTIDERIVATIVE, function=source):
                poly, a = expand_input(text(source).expression)
                self.assertEqual(antiderivative(poly, a), (slope, n0, n1))  # The card's value, by the reference.
                result = support.run(ANTIDERIVATIVE, function=text(source), variable=X)
                self.assertVerified(result)
                self.assertIs(type(result.value), RealFunction)
                self.assertEqual(result.value.variables, (X,))
                self.assertEqual(read_antiderivative(result.value), ((slope, n0, n1), a))
                self.assertEqual(result.steps[0].outputs[1], ())
                self.roundtrip(result)
        for source, lower, upper, element, pi_coefficient in KNOWN_INTEGRALS:
            with self.subTest(operation=INTEGRAL, function=source, lower=lower, upper=upper):
                poly, a = expand_input(text(source).expression)
                self.assertEqual(integral(poly, a, Fraction(lower), Fraction(upper)), (element, pi_coefficient))
                result = support.run(INTEGRAL, function=text(source), variable=X, lower_pi=bound(Fraction(lower)),
                                     upper_pi=bound(Fraction(upper)))
                self.assertVerified(result, pi_linear(element, pi_coefficient))
                self.assertIs(type(result.value.pi_coefficient), Rational)
                evidence = result.steps[0].outputs[1]
                self.assertEqual(evidence, (support.run(ANTIDERIVATIVE, function=text(source), variable=X).value,))
                self.roundtrip(result)
        # The card's renderings of the values.
        self.assertEqual(mf.render_text(support.run(INTEGRAL, **SMALL[INTEGRAL]()).value), "pi/2")
        self.assertEqual(mf.to_latex(support.run(INTEGRAL, **SMALL[INTEGRAL]()).value), r"\frac{\pi}{2}")
        # A constant integrates to c*x and 0 to 0; a constant function takes any rational bounds.
        cases = (("3", Fraction(3)), ("0", Fraction(0)), ("2/7", Fraction(2, 7)))
        for source, slope in cases:
            with self.subTest(constant=source):
                result = support.run(ANTIDERIVATIVE, function=text(source), variable=X)
                self.assertVerified(result)
                self.assertEqual(read_antiderivative(result.value), ((slope, {}, {}), None))
                result = support.run(INTEGRAL, function=text(source), variable=X, lower_pi=q(Fraction(1, 7)),
                                     upper_pi=-2)
                self.assertVerified(result, pi_linear(ZERO_E, slope * Fraction(-15, 7)))

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the class of the card."""
        for operation, make in SMALL.items():
            for name in make():
                for item in STAND_INS + (Integer(1), Fraction(1, 2)):
                    with self.subTest(operation=operation, argument=name, item=item):
                        self.assertInvalidInput(support.run(operation, **dict(make(), **{name: item})))
        sine, cosine = node("sin", leaf()), node("cos", leaf())
        outside = (
            ("x*sin(x)", node("mul", leaf(), sine)),
            ("x", leaf()),
            ("x**0*sin(x)", node("mul", node("pow", leaf(), value=0), sine)),
            ("sin(x) + cos(2x)", node("add", sine, node("cos", argument(Fraction(2))))),
            ("sin(2x)**0 + sin(x)", node("add", node("pow", node("sin", argument(Fraction(2))), value=0), sine)),
            ("0*sin(3x) + sin(x)", node("add", node("mul", scalar(0), node("sin", argument(Fraction(3)))), sine)),
            ("sin(x)**-1", node("pow", sine, value=-1)),
            ("1/sin(x)", node("div", scalar(1), sine)),
            ("sin(x)/0", node("div", sine, scalar(0))),
            ("sin(x + 1)", node("sin", node("add", leaf(), scalar(1)))),
            ("sin(0*x)", node("sin", node("mul", scalar(0), leaf()))),
            ("sin(x*x)", node("sin", node("mul", leaf(), leaf()))),
            ("sin(sin(x))", node("sin", sine)),
            ("sin(y)", node("sin", leaf(Y))),
            ("sin(x)*y", node("mul", sine, leaf(Y))),
            ("tan(x)", node("tan", leaf())),
            ("exp(x)", node("exp", leaf())),
            ("pi*sin(x)", node("mul", node("pi"), sine)),
            ("sqrt(sin(x))", node("sqrt", sine)),
            ("sin(sqrt(2)*x)", text("sin(sqrt(2)*x)").expression),
        )
        for label, tree in outside:
            for operation in (ANTIDERIVATIVE, INTEGRAL):
                with self.subTest(operation=operation, function=label):
                    try:
                        candidate = RealFunction(tree, (X,))
                    except InvalidInput:
                        continue  # RealFunction itself refuses it.
                    args = dict(function=candidate, variable=X)
                    if operation == INTEGRAL:
                        args.update(lower_pi=0, upper_pi=1)
                    result = support.run(operation, **args)
                    self.assertInvalidInput(result)
                    self.assertIn(OUTSIDE, result.error.message)
        f = text("sin(x)^2")
        wrong = (
            ("variable y", dict(function=f, variable=Y)),
            ("variable with assumptions", dict(function=f, variable=N)),
            ("two variables", dict(function=RealFunction(f.expression, (X, Y)), variable=X)),
            ("domain not TRUE", dict(function=text("tan(x)"), variable=X)),
            ("not a function", dict(function=f.expression, variable=X)),
            ("variable not a symbol", dict(function=f, variable=leaf())),
        )
        for label, args in wrong:
            for operation in (ANTIDERIVATIVE, INTEGRAL):
                with self.subTest(operation=operation, case=label):
                    if operation == INTEGRAL:
                        args = dict(args, lower_pi=0, upper_pi=1)
                    self.assertInvalidInput(support.run(operation, **args))
        # Bounds that are no multiples of pi/(12 a), also where the sines and cosines of f cancel (the card's rule).
        bounds = ((text("sin(x)^2"), 0, q(Fraction(1, 25))), (text("cos(x/2)"), q(Fraction(1, 7)), 0),
                  (text("sin(3*x)"), q(Fraction(1, 72)), 1),
                  (function(node("add", sine, node("mul", scalar(-1), sine))), 0, q(Fraction(1, 25))))
        for f, lower, upper in bounds:
            with self.subTest(function=str(f.expression), lower=lower, upper=upper):
                result = support.run(INTEGRAL, function=f, variable=X, lower_pi=lower, upper_pi=upper)
                self.assertInvalidInput(result)
                self.assertIn("12*a*lower_pi and 12*a*upper_pi must be integers", result.error.message)
        # sin(3x) on multiples of pi/36 is fine.
        self.assertVerified(support.run(INTEGRAL, function=text("sin(3*x)"), variable=X, lower_pi=q(Fraction(1, 36)),
                                        upper_pi=q(Fraction(-5, 36))))
        # produce() refuses a stand-in itself, before operations_v6.run's checks.
        for operation, make in SMALL.items():
            for name in make():
                for item in (True, 0.5, "1", None, Fraction(1, 2), Integer(1), Ratio(1, 2)):
                    with self.subTest(operation=operation, argument=name, item=item, path="produce"):
                        with self.assertRaises(InvalidInput):
                            family.produce(operation, dict(make(), **{name: item}))
            with self.subTest(operation=operation, path="produce", arguments="renamed"):
                with self.assertRaises(InvalidInput):
                    family.produce(operation, dict(make(), extra=1))
                with self.assertRaises(InvalidInput):
                    family.produce(operation, tuple(make().items()))
        for operation in ("trig_power", "Trig_power_integral", "", None):
            with self.subTest(operation=operation), self.assertRaises(UnsupportedOperation):
                family.produce(operation, SMALL[ANTIDERIVATIVE]())
        # A forged function and a forged tree are invalid input as well.
        forged = support.forged(RealFunction, expression=support.forged(RealExpr, op="sin", args=(leaf(),),
                                                                          value=1), variables=(X,),
                                domain=f.domain)
        with self.assertRaises(InvalidInput):
            family.produce(ANTIDERIVATIVE, dict(function=forged, variable=X))
        # The value function raises the structured path's error.
        with self.assertRaises(mf.OperationError) as raised:
            support.value(ANTIDERIVATIVE, function=text("x*sin(x)"), variable=X)
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_resource_limits(self):
        """An integer of 2**65536 and max_work=1 exceed the budget. No operation of the family takes a matrix, so
        support.oversized_matrix() has nothing to exceed here. A degree, a node count or a product beyond the budget
        is refused at once."""
        for operation, make in SMALL.items():
            with self.subTest(operation=operation, max_work=1):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **make()))
        for name in ("lower_pi", "upper_pi"):
            with self.subTest(argument=name):
                self.assertResourceLimit(support.run(INTEGRAL, **dict(SMALL[INTEGRAL](), **{name: support.HUGE})))
        with computation(ComputationLimits(max_integer_bits=200_000), fresh=True):
            huge_scalar = node("mul", RealExpr("scalar", (), Rational(support.HUGE)), node("sin", leaf()))
            huge_power = node("pow", node("sin", leaf()), value=support.HUGE)
            huge_argument = node("sin", node("mul", RealExpr("scalar", (), Rational(support.HUGE)), leaf()))
        start = time.perf_counter()
        for label, tree in (("scalar", huge_scalar), ("exponent", huge_power), ("argument", huge_argument),
                            ("degree", node("pow", node("sin", leaf()), value=10 ** 7)),
                            ("product", node("pow", node("add", node("sin", leaf()), node("cos", leaf())),
                                             value=100_000))):
            for operation, make in SMALL.items():
                with self.subTest(operation=operation, function=label):
                    self.assertResourceLimit(support.run(operation, **dict(make(), function=function(tree))))
        self.assertLess(time.perf_counter() - start, 10 * FAST)
        # More distinct nodes than max_nodes.
        many = function(balanced([node("mul", scalar(c), node("sin", leaf())) for c in range(1, 60)]))
        self.assertResourceLimit(support.run(ANTIDERIVATIVE, function=many, variable=X,
                                             limits=ComputationLimits(max_nodes=50)))
        self.assertVerified(support.run(ANTIDERIVATIVE, function=many, variable=X))

    def test_work_follows_the_checkers_scale(self):
        """Regression (review of wave m3): the producer charged every pair of a polynomial product at the widest
        coefficient and every product in Q(sqrt 2, sqrt 3) as one of two long factors, about three times what the
        checker charges for the same request, so it refused requests that the checker verifies within the default
        budget: sin(x)**1500, whose binomial coefficients vary widely, and a degree-40 integral with 3000-bit
        coefficients. Both are answered and verified under the default limits now."""
        result = support.run(ANTIDERIVATIVE, function=function(node("pow", node("sin", leaf()), value=1500)),
                             variable=X)
        self.assertVerified(result)
        u = argument(Fraction(7, 3))
        poly = {(j, 40 - j): Fraction(2 ** 3000 + j, 2 ** 1500 + 2 * j + 1) for j in range(0, 41, 7)}
        lower, upper = Fraction(1, 28), Fraction(5, 28)
        result = support.run(INTEGRAL, function=function(poly_tree(poly, u)), variable=X, lower_pi=q(lower),
                             upper_pi=q(upper))
        self.assertVerified(result, pi_linear(*integral(poly, Fraction(7, 3), lower, upper)))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with an antiderivative that the producer would not have
        written: other trees for the same F, the coefficient divided out, the argument written another way."""
        rng = random.Random(20261113)
        for _ in range(20):
            a = rng.choice(FREQUENCIES)
            poly = random_poly(rng, 5)
            args = dict(function=function(poly_tree(poly, argument(a))), variable=X)
            slope, n0, n1 = antiderivative(poly, a)
            for style in ("div", "right"):
                u = argument(a, style)
                terms = [node("mul", scalar(slope), leaf())] if slope else []
                for k, c in sorted(n0.items()):
                    terms.append(node("div", node("mul", scalar(c.numerator), node("pow", node("cos", u), value=k)),
                                      scalar(c.denominator)))
                for k, c in sorted(n1.items()):
                    terms.append(node("mul", node("mul", node("sin", u), node("pow", node("cos", u), value=k)),
                                      scalar(c)))
                rng.shuffle(terms)
                claim = function(balanced(terms))
                with self.subTest(poly=poly, a=a, style=style):
                    self.assertAccepts(ANTIDERIVATIVE, args, claim, ())
                    lower, upper = random_bound(rng, a), random_bound(rng, a)
                    value = pi_linear(*integral(poly, a, lower, upper))
                    self.assertAccepts(INTEGRAL, dict(args, lower_pi=bound(lower), upper_pi=bound(upper)), value,
                                       (claim,))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where a value belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence; mf.verify fails the mathematics."""
        result = support.run(ANTIDERIVATIVE, function=text("sin(x)^2"), variable=X)
        args = dict(result.request.arguments)
        value = result.value
        shifted = function(node("add", value.expression, scalar(1)))
        doubled = function(node("mul", scalar(2), value.expression))
        corrupt_anti = [("constant term", shifted, ()), ("wrong value", doubled, ()), ("list evidence", value, []),
                        ("extra evidence", value, (value,)), ("bool", True, ()), ("float", 0.5, ()), ("str", "x", ()),
                        ("expression", value.expression, ()), ("none evidence", value, None)]
        for label, claim, evidence in corrupt_anti:
            with self.subTest(operation=ANTIDERIVATIVE, corruption=label):
                self.assertRejects(ANTIDERIVATIVE, args, claim, evidence)
        self.assertRejects(INTEGRAL, args, value, ())  # The wrong operation name: its arguments do not fit.
        self.assertRejects("trig_power", args, value, ())
        integral_result = support.run(INTEGRAL, function=text("sin(x)^2"), variable=X, lower_pi=0,
                                      upper_pi=q(Fraction(1, 3)))
        iargs = dict(integral_result.request.arguments)
        right, (evidence,) = integral_result.value, integral_result.steps[0].outputs[1]
        wrong_pi = family.PiLinear(right.algebraic, q(Fraction(1, 5)))
        wrong_algebraic = pi_linear((Fraction(0), Fraction(0), Fraction(1, 8), Fraction(0)), Fraction(1, 6))
        fraction_pi = support.forged(family.PiLinear, algebraic=right.algebraic, pi_coefficient=Fraction(1, 6))
        swapped = support.forged(Mul, factors=(sqrt(3), q(Fraction(-1, 8))))  # Model's order is the Rational first.
        unsorted = support.forged(family.PiLinear, algebraic=swapped, pi_coefficient=right.pi_coefficient)
        corrupt_integral = [("wrong pi", wrong_pi, (evidence,)), ("wrong algebraic", wrong_algebraic, (evidence,)),
                            ("fraction pi", fraction_pi, (evidence,)), ("not canonical", unsorted, (evidence,)),
                            ("impostor", support.impostor(right), (evidence,)),
                            ("unset", support.forged(family.PiLinear), (evidence,)),
                            ("no evidence", right, ()), ("list evidence", right, [evidence]),
                            ("extra evidence", right, (evidence, evidence)),
                            ("wrong antiderivative", right, (shifted,)), ("bool", True, (evidence,)),
                            ("float", 0.5, (evidence,)), ("str", "pi/6", (evidence,))]
        for label, claim, claimed_evidence in corrupt_integral:
            with self.subTest(operation=INTEGRAL, corruption=label):
                self.assertRejects(INTEGRAL, iargs, claim, claimed_evidence)
        # An int where the Rational pi coefficient belongs: the integral of 1 from 0 to pi is pi.
        one = support.run(INTEGRAL, function=text("1"), variable=X, lower_pi=0, upper_pi=1)
        self.assertEqual(one.value, family.PiLinear(q(0), q(1)))
        int_pi = support.forged(family.PiLinear, algebraic=q(0), pi_coefficient=1)
        with self.subTest(corruption="int pi"):
            self.assertRejects(INTEGRAL, dict(one.request.arguments), int_pi, one.steps[0].outputs[1])
        with self.subTest(corruption="oversized evidence"):
            start = time.perf_counter()
            self.assertRejects(INTEGRAL, iargs, right, (evidence,) + (0,) * 100_000, may_exhaust=True)
            self.assertLess(time.perf_counter() - start, 2 * FAST)
        for candidate, claim, claimed_evidence in ((result, doubled, ()), (integral_result, wrong_pi, (evidence,))):
            with self.subTest(verify=candidate.method):
                report = support.statuses(mf.verify(support.tampered(candidate, claim, claimed_evidence)))
                self.assertIs(report["result.mathematics"], CheckStatus.FAILED)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        requests = [(operation, make()) for operation, make in SMALL.items()]
        requests += [(INTEGRAL, dict(function=text("cos(2*x)^3*sin(2*x)^2"), variable=X, lower_pi=q(Fraction(1, 8)),
                                     upper_pi=q(Fraction(-5, 24)))),
                     (ANTIDERIVATIVE, dict(function=text("sin(x/3)^5 + 2"), variable=X))]
        results = [support.run(operation, **args) for operation, args in requests]
        texts = [mf.dumps(result) for result in results]
        with patch.object(family, "produce", side_effect=AssertionError("verification ran the producer")):
            for (operation, _), result, text_ in zip(requests, results, texts):
                with self.subTest(operation=operation):
                    self.assertTrue(mf.verify(result).verified)
                    decoded = mf.loads(text_)
                    self.assertEqual(decoded, result)
                    self.assertTrue(mf.verify(decoded).verified)
            # The patch is live: a new run reaches the patched producer and fails.
            failed = support.run(ANTIDERIVATIVE, **SMALL[ANTIDERIVATIVE]())
            self.assertEqual(failed.error, ErrorInfo("internal_error", "The producer failed unexpectedly"))

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX: a RealFunction, and
        PiLinear values with a pi term only, an algebraic part only, both, several radicals and zero."""
        rendered = (
            (ANTIDERIVATIVE, dict(function=text("sin(x)^2"), variable=X), "((1/2*x) + (-1/2*(sin(x)*cos(x))))",
             r"\left(\left(\frac{1}{2} \cdot x\right) + \left(-\frac{1}{2} \cdot \left(\operatorname{sin}\left(x"
             r"\right) \cdot \operatorname{cos}\left(x\right)\right)\right)\right)"),
            (ANTIDERIVATIVE, dict(function=text("sin(2*x)*cos(2*x)^2"), variable=X), "(-1/6*(cos((2*x)))^3)",
             r"\left(-\frac{1}{6} \cdot \left(\operatorname{cos}\left(\left(2 \cdot x\right)\right)\right)^{3}"
             r"\right)"),
            (ANTIDERIVATIVE, dict(function=text("3"), variable=X), "(3*x)", r"\left(3 \cdot x\right)"),
            (ANTIDERIVATIVE, dict(function=text("0"), variable=X), "0", "0"),
            (INTEGRAL, dict(function=text("sin(x)^2"), variable=X, lower_pi=0, upper_pi=1), "pi/2", r"\frac{\pi}{2}"),
            (INTEGRAL, dict(function=text("sin(x)^3"), variable=X, lower_pi=0, upper_pi=1), "4/3", r"\frac{4}{3}"),
            (INTEGRAL, dict(function=text("sin(x)^2"), variable=X, lower_pi=0, upper_pi=q(Fraction(1, 3))),
             "-sqrt(3)/8 + pi/6", r"-\frac{\sqrt{3}}{8} + \frac{\pi}{6}"),
            (INTEGRAL, dict(function=text("cos(x)"), variable=X, lower_pi=0, upper_pi=q(Fraction(1, 12))),
             "-sqrt(2)/4 + sqrt(6)/4", r"-\frac{\sqrt{2}}{4} + \frac{\sqrt{6}}{4}"),
            (INTEGRAL, dict(function=text("cos(x)"), variable=X, lower_pi=q(Fraction(1, 12)),
                            upper_pi=q(Fraction(1, 4))),
             "3*sqrt(2)/4 - sqrt(6)/4", r"\frac{3 \sqrt{2}}{4} - \frac{\sqrt{6}}{4}"),
            (INTEGRAL, dict(function=text("sin(x)"), variable=X, lower_pi=0, upper_pi=2), "0", "0"),
        )
        for operation, args, text_, latex in rendered:
            with self.subTest(operation=operation, text=text_):
                result = support.run(operation, **args)
                decoded = self.roundtrip(result)
                self.assertEqual(decoded.steps[0].outputs, result.steps[0].outputs)
                self.assertEqual(mf.render_text(result.value), text_)
                self.assertEqual(mf.to_latex(result.value), latex)
                self.assertEqual(mf.render_text(result), "completed / value [exact; complete]: " + text_)

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        rng = random.Random(20261114)
        cases = []
        for _ in range(10):
            a = rng.choice(FREQUENCIES)
            f = function(poly_tree(random_poly(rng, 4), argument(a, "plain")))
            cases.append((ANTIDERIVATIVE, dict(function=f, variable=X)))
            cases.append((INTEGRAL, dict(function=f, variable=X, lower_pi=bound(random_bound(rng, a)),
                                         upper_pi=bound(random_bound(rng, a)))))
        for operation, args in cases:
            with self.subTest(operation=operation, function=str(args["function"].expression)):
                expected = support.run(operation, **args).value
                self.assertIsNotNone(expected)
                found = support.lean_value(operation, **args)
                self.assertIsNot(found, lean.FALLBACK)
                self.assertEqual(found, expected)
                self.assertEqual(support.value(operation, **args), expected)

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        requests = [(operation, make()) for operation, make in SMALL.items()]
        requests.append((INTEGRAL, dict(function=text("cos(x/2)^3"), variable=X, lower_pi=q(Fraction(-1, 3)),
                                        upper_pi=q(Fraction(5, 6)))))
        for operation, args in requests:
            with self.subTest(operation=operation, function=str(args["function"].expression)):
                response = support.call_values(operation, **args)
                self.assertFalse(response.is_error, response.payload)
                result = self.answer(response)
                self.assertEqual(result, support.run(operation, **args))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call_values(INTEGRAL, function=text("sin(x)"), variable=X, lower_pi=0,
                                                         upper_pi=q(Fraction(1, 25))), "invalid_input")
        self.assertIn("must be integers", error["message"])
        error = self.assertToolError(support.call_values(ANTIDERIVATIVE, function=text("x*sin(x)"), variable=X),
                                     "invalid_input")
        self.assertIn(OUTSIDE, error["message"])

    def test_seeded_agreement_with_references(self):
        """produce() agrees with the references of this module on 500 seeded inputs per operation, and the
        checker accepts every answer: random polynomials written term by term, random trees, every style of
        argument."""
        rng = random.Random(20261115)
        styles = ("plain", "mul", "div", "right", "detour")
        for index in range(500):
            a, style = rng.choice(FREQUENCIES), rng.choice(styles)
            u = argument(a, style)
            if index % 2:
                tree = random_tree(rng, u, 4)
            else:
                tree = poly_tree(random_poly(rng, rng.randint(0, 7), rng.randint(1, 5)), u, rng)
            poly, found = expand_input(tree)
            args = {"function": function(tree), "variable": X}
            with self.subTest(index=index, tree=str(tree)):
                value, evidence = family.produce(ANTIDERIVATIVE, dict(args))
                self.assertEqual(evidence, ())
                (slope, n0, n1), read = read_antiderivative(value)
                self.assertEqual((slope, n0, n1), antiderivative(poly, found))
                if n0 or n1:
                    self.assertEqual(read, found)
                self.assertIs(support.verdict(family, ANTIDERIVATIVE, args, value, evidence), True)
                # trig_power_integral on the same function, between multiples of pi/(12 a), or anywhere for a
                # function without a sine or a cosine.
                if found is None:
                    lower, upper = Fraction(rng.randint(-50, 50), rng.randint(1, 30)), Fraction(rng.randint(-9, 9))
                else:
                    lower, upper = random_bound(rng, found), random_bound(rng, found)
                iargs = dict(args, lower_pi=bound(lower), upper_pi=bound(upper))
                ivalue, ievidence = family.produce(INTEGRAL, dict(iargs))
                self.assertEqual(ivalue, pi_linear(*integral(poly, found, lower, upper)))
                self.assertIs(type(ivalue.pi_coefficient), Rational)
                self.assertEqual(ievidence, (value,))
                self.assertIs(support.verdict(family, INTEGRAL, iargs, ivalue, ievidence), True)

    def test_antiderivative_is_canonical(self):
        """F depends only on the reduced form of f, a and the variable: differently written inputs of the same
        function give the same F, a balanced sum that shares one sine and one cosine node, with u = x for a = 1."""
        same = ("sin(x)^2", "1 - cos(x)^2", "(1 - cos(x))*(1 + cos(x))", "sin(x)*sin(x)")
        values = [support.run(ANTIDERIVATIVE, function=text(source), variable=X).value for source in same]
        self.assertTrue(all(value == values[0] for value in values), values)
        value, _ = family.produce(ANTIDERIVATIVE, dict(function=text("sin(3*x)^4*cos(3*x) + cos(3*x)^5"), variable=X))
        nodes, pending = {}, [value.expression]
        while pending:
            item = pending.pop()
            nodes.setdefault((item.op, id(item)), item)
            pending.extend(item.args)
        sines = {key for key in nodes if key[0] == "sin"}
        cosines = {key for key in nodes if key[0] == "cos"}
        self.assertEqual((len(sines), len(cosines)), (1, 1))
        arg = next(iter(nodes[next(iter(sines))].args))
        self.assertEqual((arg.op, arg.args[0].value, arg.args[1].value), ("mul", q(3), X))
        unit, _ = family.produce(ANTIDERIVATIVE, dict(function=text("cos(x)"), variable=X))
        self.assertEqual(unit.expression, node("sin", leaf()))
        # A degree-48 request fits the default budget well within the time of a request.
        start = time.perf_counter()
        result = support.run(INTEGRAL, function=text("(sin(x/2) + 2*cos(x/2))^48"), variable=X, lower_pi=0,
                             upper_pi=q(Fraction(1, 3)))
        self.assertVerified(result)
        self.assertLess(time.perf_counter() - start, 5 * FAST)
