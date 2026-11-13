"""The piecewise family of mathforge.equations: solve_abs, solve_sqrt.

These tests are the definition of done. tests/v09/support.py holds the shared helpers, and
tests/v08/test_continued_fraction.py tests the pilot family. A FamilyTestCase keeps FAMILY registered while its tests
run, before and after

    python tools/new_family.py register equations piecewise

Every expected value comes from the card (docs/v09/cards/m2-algebra-equations-sets-numeric.md) or from the reference
of this module, written here and slow on purpose; never from the producer. The reference reads a problem in a small
language of tuples, which also builds its RealExpr trees unsimplified. It takes as candidate points the real roots of
the atoms and of F with every abs(P) replaced by +P or -P, for every choice of signs (solve_abs), or of P, A, B and
A^2 P - B^2 (solve_sqrt): rational roots by the rational root theorem, the others isolated by Sturm sequences and
bisection over Fraction. It decides the relation at a rational point by evaluating the tree exactly (in Q(sqrt p) for a
square root), and at an irrational point from signs found by a gcd and a refined isolating interval. An irrational end
of the producer's value is compared with the reference's by value: a common factor of their polynomials has exactly one
root in the meet of the two isolating intervals. tests/v09/equations/test_piecewise_checker.py tests the checker on its
own.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
import itertools
from math import gcd
import random
import time
from unittest.mock import patch

import mathforge as mf
from mathforge import lean
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.equations import piecewise as family
from mathforge.contracts import EmptySet, FiniteSet, UniversalSet
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.functions import DomainPredicate, RealExpr, RealFunction, parse_function
from mathforge.intervals import Interval, IntervalSet
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Integers, Reals

from .. import support

Q = mf.Rational
F = Fraction
XS = mf.symbol("x")
YS = mf.symbol("y")
RELATIONS = ("=", "<", "<=", ">", ">=")
OPERATIONS = ("solve_abs", "solve_sqrt")
STAND_INS = (0.5, True, "1", None, (1, 2), [1])  # No RealFunction, Symbol or relation: a float, a bool, a str, ...
FAST = 2.0  # Seconds: a request that its budget refuses fails well within this.
SEEDED = 200  # Seeded problems per operation in the agreement test.


# ---------------------------------------------------------------------------------------------------------------
# The expression language: ("x",), ("num", q), ("add", a, b), ("mul", a, b), ("pow", a, n), ("div", a, b),
# ("abs", a), ("sqrt", a).

X = ("x",)


def lift(value: object) -> tuple:
    return value if type(value) is tuple else ("num", F(value))


def add(*terms: object) -> tuple:
    return reduce(lambda a, b: ("add", a, b), [lift(term) for term in terms])


def mul(*factors: object) -> tuple:
    return reduce(lambda a, b: ("mul", a, b), [lift(factor) for factor in factors])


def sub(a: object, b: object) -> tuple:
    return ("add", lift(a), ("mul", ("num", F(-1)), lift(b)))


def pw(a: object, n: int) -> tuple:
    return ("pow", lift(a), n)


def dv(a: object, b: object) -> tuple:
    return ("div", lift(a), lift(b))


def ab(a: object) -> tuple:
    return ("abs", lift(a))


def sq(a: object) -> tuple:
    return ("sqrt", lift(a))


def poly_node(coefficients: list) -> tuple:
    """The node of sum c_k x^k (lowest degree first), each term c * x^k, zero terms left out."""
    terms = [mul(c, pw(X, k)) if k > 1 else mul(c, X) if k == 1 else lift(c)
             for k, c in enumerate(coefficients) if c]
    return add(*terms) if terms else lift(0)


def real(node: tuple, symbol: mf.Symbol = XS) -> RealExpr:
    """The RealExpr tree of a node, built with the RealExpr constructor and never simplified."""
    op = node[0]
    if op == "x":
        return RealExpr("symbol", (), symbol)
    if op == "num":
        return RealExpr("scalar", (), Q(node[1].numerator, node[1].denominator))
    if op == "pow":
        return RealExpr("pow", (real(node[1], symbol),), node[2])
    return RealExpr(op, tuple(real(child, symbol) for child in node[1:]))


def function(node: object, variables: tuple = (XS,)) -> RealFunction:
    return RealFunction(real(lift(node)), variables)


def parsed(text: str) -> RealFunction:
    """A RealFunction of x from the parser, which simplifies its tree."""
    return parse_function(text, symbols={"x": XS})


def arguments(lhs: object, rhs: object, relation: str = "=") -> dict:
    """The arguments of a request: nodes become RealFunctions of x, RealFunctions stay."""
    as_function = (lambda item: item if type(item) is RealFunction else function(item))
    return dict(lhs=as_function(lhs), rhs=as_function(rhs), variable=XS, relation=relation)


# ---------------------------------------------------------------------------------------------------------------
# Expected values

def end(value: object) -> object:
    return None if value is None else value if type(value) is not int and type(value) is not F \
        else Q(F(value).numerator, F(value).denominator)


def closed(a: object, b: object) -> Interval:
    return Interval(end(a), end(b), True, True)


def opened(a: object, b: object) -> Interval:
    return Interval(end(a), end(b), False, False)


def solution(*intervals: Interval) -> object:
    return IntervalSet(tuple(intervals)).to_solution_set()


def points(*values: object) -> object:
    return solution(*[closed(value, value) for value in values])


EMPTY = EmptySet(Reals)
ALL = UniversalSet(Reals)
ROOT3 = (RealAlgebraicRoot((-3, 0, 1), 0), RealAlgebraicRoot((-3, 0, 1), 1))  # -sqrt(3), sqrt(3).
SQRT_X2_MINUS_X = sub(sq(pw(X, 2)), X)  # Built directly: real_function would rewrite sqrt(x^2) as abs(x).

# The known values of the card, and a few more: (operation, lhs, rhs, relation, value).
KNOWN = (
    ("solve_abs", ab(sub(X, 1)), 3, "=", points(-2, 4)),
    ("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3, "=", solution(closed(-2, 1))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), X, "<", solution(opened(1, 2))),
    ("solve_abs", ab(X), -1, "=", EMPTY),
    ("solve_abs", ab(X), 0, ">=", ALL),
    ("solve_abs", ab(sub(X, 1)), 2, "<=", solution(closed(-1, 3))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=", solution(closed(ROOT3[0], -1), closed(1, ROOT3[1]))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), 1, ">",
     solution(opened(None, ROOT3[0]), opened(-1, 1), opened(ROOT3[1], None))),
    ("solve_abs", sub(X, 2), 0, ">=", solution(Interval(Q(2), None, True, False))),  # No atom: a polynomial.
    ("solve_abs", sub(ab(X), ab(X)), 0, "=", ALL),  # Every piece is zero.
    ("solve_abs", ab(-2), 2, "=", ALL),  # A constant atom.
    ("solve_sqrt", sq(add(X, 2)), X, "=", points(2)),
    ("solve_sqrt", sq(sub(X, 1)), sub(X, 3), "<", solution(opened(5, None))),
    ("solve_sqrt", sq(X), -1, "=", EMPTY),
    ("solve_sqrt", SQRT_X2_MINUS_X, 0, "=", solution(Interval(Q(0), None, True, False))),
    ("solve_sqrt", sq(mul(-1, pw(X, 2))), 0, "=", points(0)),
    ("solve_sqrt", add(mul(2, sq(X)), sq(X)), 3, "=", points(1)),
    ("solve_sqrt", sq(add(X, 2)), X, "<=", solution(Interval(Q(2), None, True, False))),
    ("solve_sqrt", mul(sq(2), X), 1, "=", points(RealAlgebraicRoot((-1, 0, 2), 1))),  # x = 1/sqrt(2).
    ("solve_sqrt", sq(-1), 0, "<=", EMPTY),  # The domain is FALSE.
    ("solve_sqrt", sq(sub(X, X)), X, ">=", solution(Interval(None, Q(0), False, True))),  # A zero radicand.
    ("solve_sqrt", dv(sq(X), 2), 1, "=", points(4)),
    ("solve_sqrt", pw(sq(X), 0), 1, "=", solution(Interval(Q(0), None, True, False))),  # (sqrt x)^0 = 1 on x >= 0.
)

# The evidence of the checker's module docstring for three of them: (operation, lhs, rhs, relation, evidence).
KNOWN_EVIDENCE = (
    ("solve_abs", ab(sub(X, 1)), 3, "=", ((-2, 1, 4), (-3, 0, 2, 5))),
    ("solve_sqrt", sq(add(X, 2)), X, "=", ((-2, -1, 0, 2), (-3, F(-3, 2), F(-1, 2), 1, 3))),
)

_X_EXPR = RealExpr("symbol", (), XS)
_SQRT_X = RealExpr("sqrt", (_X_EXPR,))
_ONE = RealExpr("scalar", (), Q(1))


class _StrKind(str):
    """A str subclass: equal to its text, but never an exact str."""

# Inputs outside the class of the card, each invalid input: (operation, lhs, rhs, relation).
INVALID = (
    ("solve_abs", parsed("sin(x)"), 0, "="),  # Another function.
    ("solve_abs", ab(sub(ab(X), 1)), 0, "="),  # An abs inside an abs.
    ("solve_abs", sq(X), 1, "="),  # A sqrt in solve_abs.
    ("solve_abs", parsed("abs(x)/x"), 1, "="),  # A division by a non-constant.
    ("solve_abs", parsed("abs(x)/(x - x + 2)"), 1, "="),  # Its nonzero domain predicate.
    ("solve_abs", parsed("pi*abs(x)"), 1, "="),
    ("solve_abs", parsed("E + abs(x)"), 1, "="),
    ("solve_abs", parsed("abs(x) + x^(-1)"), 1, "="),  # A negative power.
    ("solve_abs", dv(ab(X), 0), 1, "="),  # A division by zero.
    ("solve_abs", ab(dv(X, sub(X, X))), 1, "="),  # A division by the zero polynomial.
    ("solve_abs", RealFunction(RealExpr("scalar", (), RealAlgebraicRoot((-2, 0, 1), 1)), (XS,)), 1, "="),
    ("solve_abs", RealFunction(RealExpr("abs", (_X_EXPR,)), (XS,), DomainPredicate("positive", (_X_EXPR,))), 1,
     "="),
    ("solve_abs", function(ab(X)), function(1), "!="),  # Another relation.
    ("solve_abs", function(ab(X)), function(1), "=="),
    ("solve_abs", function(ab(X), (XS, YS)), function(1), "="),  # A function of two variables.
    ("solve_abs", RealFunction(RealExpr("add", (RealExpr("abs", (_X_EXPR,)), RealExpr("symbol", (), YS))),
                               (XS, YS)), 1, "="),
    ("solve_abs", RealFunction(real(ab(X), YS), (YS,)), 1, "="),  # A function of another variable.
    ("solve_sqrt", ab(X), 1, "="),  # No sqrt, and an abs.
    ("solve_sqrt", add(ab(X), sq(X)), 1, "="),
    ("solve_sqrt", X, 1, "="),  # No sqrt at all.
    ("solve_sqrt", add(sq(X), sq(add(X, 1))), 1, "="),  # A second radicand.
    ("solve_sqrt", sq(add(sq(X), 1)), 1, "="),  # A sqrt inside a sqrt.
    ("solve_sqrt", mul(sq(X), sq(X)), 1, "="),  # A product of two square roots.
    ("solve_sqrt", pw(sq(X), 2), 1, "="),  # A power of a square root.
    ("solve_sqrt", pw(sq(X), 3), 1, "="),
    ("solve_sqrt", dv(sq(X), add(X, 1)), 1, "="),
    ("solve_sqrt", parsed("sqrt(x) + sin(x)"), 1, "="),
    ("solve_sqrt", RealFunction(_SQRT_X, (XS,), DomainPredicate("nonnegative", (RealExpr("add", (_X_EXPR, _ONE)),))),
     1, "="),  # A domain predicate on another polynomial.
    ("solve_sqrt", RealFunction(_SQRT_X, (XS,), DomainPredicate("positive", (_X_EXPR,))), 1, "="),
    ("solve_sqrt", RealFunction(_SQRT_X, (XS,), DomainPredicate("false")), 1, "="),  # FALSE, P not negative.
    ("solve_sqrt", function(sq(X)), function(1), "<>"),
    # A conjunction whose kinds are a str subclass equal to "nonnegative", which the public constructors admit (two
    # operands of the polynomial x, so that conjunction() keeps both): invalid input, never a claim the checker
    # refuses (regression: the producer read the inner kinds with == alone).
    ("solve_sqrt", RealFunction(_SQRT_X, (XS,), DomainPredicate("and", (
        DomainPredicate(_StrKind("nonnegative"), (RealExpr("add", (_X_EXPR, RealExpr("scalar", (), Q(0))),),)),
        DomainPredicate(_StrKind("nonnegative"), (RealExpr("mul", (_ONE, _X_EXPR)),))))), 1, "="),
)


# ---------------------------------------------------------------------------------------------------------------
# The reference: polynomials over Fraction (lists, lowest degree first, no trailing zero)

def trim(p: list) -> list:
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def padd(p: list, q: list) -> list:
    n = max(len(p), len(q))
    return trim([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])


def pscale(p: list, c: object) -> list:
    return trim([c * a for a in p])


def psub(p: list, q: list) -> list:
    return padd(p, pscale(q, -1))


def pmul(p: list, q: list) -> list:
    if not p or not q:
        return []
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return trim(out)


def pdivmod(p: list, q: list) -> tuple[list, list]:
    p, quotient = [F(a) for a in p], [F(0)] * max(len(p) - len(q) + 1, 1)
    while len(p) >= len(q):
        c = p[-1] / q[-1]
        shift = len(p) - len(q)
        quotient[shift] = c
        p = trim([a - (c * q[i - shift] if 0 <= i - shift < len(q) else 0) for i, a in enumerate(p)])
    return trim(quotient), p


def pgcd(p: list, q: list) -> list:
    """The monic gcd ([] for two zeros)."""
    while q:
        p, q = q, pdivmod(p, q)[1]
    return pscale(p, 1 / F(p[-1])) if p else []


def pderiv(p: list) -> list:
    return trim([k * p[k] for k in range(1, len(p))])


def pvalue(p: list, t: F) -> F:
    value = F(0)
    for c in reversed(p):
        value = value * t + c
    return value


def sign(value: F) -> int:
    return (value > 0) - (value < 0)


def squarefree(p: list) -> list:
    return pdivmod(p, pgcd(p, pderiv(p)))[0] if len(p) > 2 else p


def divisors(n: int) -> list[int]:
    n = abs(n)
    return [d for d in range(1, n + 1) if n % d == 0]


def rational_roots(p: list) -> set[F]:
    """Every rational root of a nonzero polynomial, by the rational root theorem."""
    found = set()
    if len(p) <= 1:
        return found
    while p and p[0] == 0:
        found.add(F(0))
        p = p[1:]
    if len(p) <= 1:
        return found
    scale = reduce(lambda a, b: a * b // gcd(a, b), [F(c).denominator for c in p], 1)
    integers = [int(c * scale) for c in p]
    for a in divisors(integers[0]):
        for b in divisors(integers[-1]):
            for t in (F(a, b), F(-a, b)):
                if pvalue(p, t) == 0:
                    found.add(t)
    return found


def strip(p: list, roots: set[F]) -> list:
    """p without its rational roots (each divided out as often as it divides)."""
    for r in roots:
        while len(p) > 1 and pvalue(p, r) == 0:
            p = pdivmod(p, [-r, F(1)])[0]
    return p


def sturm(p: list) -> list[list]:
    chain = [p, pderiv(p)]
    while chain[-1]:
        chain.append(pscale(pdivmod(chain[-2], chain[-1])[1], -1))
    return chain[:-1]


def variations(chain: list[list], t: F) -> int:
    signs = [sign(pvalue(p, t)) for p in chain]
    signs = [s for s in signs if s]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def count(chain: list[list], a: F, b: F) -> int:
    """The number of distinct roots in (a, b) of the square-free first polynomial of a Sturm chain, a and b not
    roots."""
    return variations(chain, a) - variations(chain, b)


def isolate(p: list) -> list[list[F]]:
    """Disjoint isolating intervals [a, b], in increasing order, of the real roots of a square-free polynomial
    without rational roots (so that no bisection point is a root)."""
    chain = sturm(p)
    bound = 1 + max(abs(F(c) / p[-1]) for c in p[:-1])
    pending, found = [[-bound, bound]], []
    while pending:
        a, b = pending.pop()
        n = count(chain, a, b)
        if n == 1:
            found.append([a, b])
        elif n > 1:
            m = (a + b) / 2
            pending += [[a, m], [m, b]]
    return sorted(found)


class Point:
    """A candidate point: a rational one (value), or the one root of the square-free polynomial s, which has no
    rational root, in the open interval (low, high)."""

    def __init__(self, value: F | None = None, s: list | None = None, low: F | None = None, high: F | None = None):
        self.value, self.s, self.low, self.high = value, s, low, high
        if s is not None:
            self.below = sign(pvalue(s, low))

    def bounds(self) -> tuple[F, F]:
        return (self.value, self.value) if self.s is None else (self.low, self.high)

    def halve(self) -> None:
        m = (self.low + self.high) / 2
        if sign(pvalue(self.s, m)) == self.below:
            self.low = m
        else:
            self.high = m

    def sign_of(self, g: list) -> int:
        """The sign of a polynomial at the point."""
        if not g:
            return 0
        if self.s is None:
            return sign(pvalue(g, self.value))
        common = pgcd(g, self.s)
        if len(common) > 1 and count(sturm(common), self.low, self.high) == 1:
            return 0
        rationals = rational_roots(g)
        core = squarefree(strip(g, rationals))
        while (any(self.low <= r <= self.high for r in rationals)
               or (len(core) > 1 and count(sturm(core), self.low, self.high) > 0)):
            self.halve()
        return sign(pvalue(g, (self.low + self.high) / 2))


def candidate_points(polynomials: list[list]) -> list[Point]:
    """Every distinct real root of the nonzero polynomials, sorted, with disjoint closed isolating intervals."""
    rationals: set[F] = set()
    basis: list[list] = []  # Pairwise coprime square-free polynomials without rational roots.
    for p in polynomials:
        if len(p) <= 1:
            continue
        roots = rational_roots(p)
        rationals |= roots
        rest = squarefree(strip(p, roots))
        for i, q in enumerate(list(basis)):
            if len(rest) <= 1:
                break
            g = pgcd(rest, q)
            if len(g) > 1:
                basis[i] = pdivmod(q, g)[0]
                basis.append(g)
                rest = pdivmod(rest, g)[0]
        if len(rest) > 1:
            basis.append(rest)
    found = [Point(value=r) for r in rationals]
    for s in basis:
        if len(s) > 1:
            found += [Point(s=s, low=a, high=b) for a, b in isolate(s)]
    while True:  # Refine until the closed intervals are pairwise disjoint.
        found.sort(key=lambda point: point.bounds()[0])
        meeting = False
        for first, second in zip(found, found[1:]):
            if second.bounds()[0] <= first.bounds()[1]:
                meeting = True
                for point in (first, second):
                    if point.s is not None:
                        point.halve()
        if not meeting:
            return found


def sign_with_root(a: int, b: int, p: int, gap: int) -> int | None:
    """The sign of a + b sqrt(p) from the signs of a, b, p and a^2 - b^2 p; None when p < 0."""
    if p < 0:
        return None
    if b == 0 or p == 0:
        return a
    if a == 0 or a == b:
        return b if a == 0 else a
    return a if gap > 0 else b if gap < 0 else 0


def evaluate(node: tuple, t: F, root: F | None = None) -> object:
    """The value of a node at x = t: a Fraction, or, with root = p (the radicand's value), a pair (a, b) for
    a + b sqrt(p); None outside the domain (a negative radicand)."""
    op = node[0]
    if op == "x":
        return t if root is None else (t, F(0))
    if op == "num":
        return node[1] if root is None else (node[1], F(0))
    if op == "sqrt":
        inner = evaluate(node[1], t)
        return None if inner is None or inner < 0 else (F(0), F(1))
    values = [evaluate(child, t, root) for child in node[1:2 if op in ("pow", "abs") else 3]]
    if any(value is None for value in values):
        return None
    if op == "abs":
        return abs(values[0])
    if root is None:
        if op == "add":
            return values[0] + values[1]
        if op == "mul":
            return values[0] * values[1]
        if op == "div":
            return values[0] / values[1]
        return values[0] ** node[2]
    (a1, b1) = values[0]
    if op == "pow":
        value = (F(1), F(0))
        for _ in range(node[2]):
            value = (value[0] * a1 + value[1] * b1 * root, value[0] * b1 + value[1] * a1)
        return value
    (a2, b2) = values[1]
    if op == "add":
        return (a1 + a2, b1 + b2)
    if op == "mul":
        return (a1 * a2 + b1 * b2 * root, a1 * b2 + a2 * b1)
    return (a1 / a2, b1 / a2)  # div by a constant: b2 = 0.


def polynomial(node: tuple, replace: dict | None = None) -> list:
    """The polynomial of a node free of sqrt, each abs node replaced by replace[the abs node]."""
    op = node[0]
    if op == "x":
        return [F(0), F(1)]
    if op == "num":
        return trim([node[1]])
    if op == "abs":
        return replace[node]
    if op == "pow":
        return reduce(pmul, [polynomial(node[1], replace)] * node[2], [F(1)])
    p, q = polynomial(node[1], replace), polynomial(node[2], replace)
    if op == "add":
        return padd(p, q)
    if op == "mul":
        return pmul(p, q)
    return pscale(p, 1 / q[0])


def nodes(node: tuple, op: str) -> list[tuple]:
    found = [node] if node[0] == op else []
    if node[0] not in ("x", "num"):
        for child in node[1:2 if node[0] in ("pow", "abs", "sqrt") else 3]:
            found += nodes(child, op)
    return found


def reference(operation: str, lhs: tuple, rhs: tuple, relation: str) -> list:
    """The solution set as a list of (lower, upper, left_closed, right_closed) with ends None, a Fraction or a Point;
    the empty list for the empty set."""
    problem = sub(lhs, rhs)
    holds = (lambda s: s is not None and {"=": s == 0, "<": s < 0, "<=": s <= 0, ">": s > 0, ">=": s >= 0}[relation])
    if operation == "solve_abs":
        atoms = list(dict.fromkeys(nodes(problem, "abs")))
        inner = {atom: polynomial(atom[1]) for atom in atoms}
        candidates = list(inner.values())
        for signs in itertools.product((1, -1), repeat=len(atoms)):
            candidates.append(polynomial(problem, {atom: pscale(inner[atom], s) for atom, s in zip(atoms, signs)}))

        def truth_at(point: Point) -> bool:
            if point.s is None:
                return holds(sign(evaluate(problem, point.value)))
            choice = {atom: pscale(inner[atom], -1 if point.sign_of(inner[atom]) < 0 else 1) for atom in atoms}
            return holds(point.sign_of(polynomial(problem, choice)))
    else:
        radicand = polynomial(nodes(problem, "sqrt")[0][1])
        # A and B of F = A sqrt(P) + B: F at sqrt(P) -> +s and -s, read off as (F+ - F-)/2 and (F+ + F-)/2 over Q.
        a, b = linear_parts(problem)
        gap = psub(pmul(b, b), pmul(pmul(a, a), radicand))
        candidates = [radicand, a, b, gap]

        def truth_at(point: Point) -> bool:
            if point.s is None:
                p = pvalue(radicand, point.value)
                value = evaluate(problem, point.value, p)
                if value is None:
                    return False
                va, vb = value
                return holds(sign_with_root(sign(va), sign(vb), sign(p), sign(va * va - vb * vb * p)))
            return holds(sign_with_root(point.sign_of(b), point.sign_of(a), point.sign_of(radicand),
                                        point.sign_of(gap)))
    found = candidate_points(candidates)
    samples = [found[0].bounds()[0] - 1] if found else [F(0)]
    for first, second in zip(found, found[1:]):
        samples.append((first.bounds()[1] + second.bounds()[0]) / 2)
    if found:
        samples.append(found[-1].bounds()[1] + 1)
    members = [truth_at(Point(value=samples[0]))]
    for k, point in enumerate(found):
        members += [truth_at(point), truth_at(Point(value=samples[k + 1]))]
    ends = [None] + [point.value if point.s is None else point for point in found] + [None]
    intervals, i = [], 0
    while i < len(members):
        if not members[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(members) and members[j + 1]:
            j += 1
        lower = ends[(i + 1) // 2]  # Member 2k is cell k (between ends k and k + 1), member 2k - 1 is point k.
        upper = ends[(j + 1) // 2] if j % 2 else ends[j // 2 + 1]
        intervals.append((lower, upper, i % 2 == 1, j % 2 == 1))
        i = j + 1
    return intervals


def linear_parts(problem: tuple) -> tuple[list, list]:
    """(A, B) with F = A sqrt(P) + B, from the pair evaluation of the tree with sqrt -> (0, 1) over polynomials."""

    def pair(node: tuple) -> tuple[list, list]:
        op = node[0]
        if op == "sqrt":
            return [F(1)], []
        if op in ("x", "num"):
            return [], polynomial(node)
        if op == "pow":
            value = ([], [F(1)])
            for _ in range(node[2]):
                a, b = pair(node[1])
                value = (padd(pmul(value[0], b), pmul(value[1], a)), pmul(value[1], b))  # Linear: a*a never occurs.
            return value
        (a1, b1), (a2, b2) = pair(node[1]), pair(node[2])
        if op == "add":
            return padd(a1, a2), padd(b1, b2)
        if op == "mul":
            return padd(pmul(a1, b2), pmul(a2, b1)), pmul(b1, b2)
        return pscale(a1, 1 / b2[0]), pscale(b1, 1 / b2[0])

    return pair(problem)


def same_algebraic(record: object, point: Point) -> bool:
    """Whether a RealAlgebraicRoot record names the irrational point."""
    if type(record) is not RealAlgebraicRoot:
        return False
    q = [F(c) for c in record.integer_coefficients]
    if rational_roots(q):
        return False
    intervals = isolate(squarefree(q))
    if record.real_index >= len(intervals):
        return False
    a, b = intervals[record.real_index]
    common = pgcd(q, point.s)
    low, high = max(a, point.low), min(b, point.high)
    return len(common) > 1 and low < high and count(sturm(common), low, high) == 1


def same_end(record: object, expected: object) -> bool:
    if expected is None:
        return record is None
    if type(expected) is F:
        return type(record) is Q and F(record.numerator, record.denominator) == expected
    return same_algebraic(record, expected)


def agrees(value: object, expected: list) -> bool:
    """Whether a solution set equals the reference's."""
    if not expected:
        return type(value) is EmptySet and value == EMPTY
    if expected == [(None, None, False, False)]:
        return type(value) is UniversalSet and value == ALL
    if type(value) is not IntervalSet or len(value.intervals) != len(expected):
        return False
    return all(interval.left_closed == lc and interval.right_closed == rc and same_end(interval.lower, lower)
               and same_end(interval.upper, upper)
               for interval, (lower, upper, lc, rc) in zip(value.intervals, expected))


# ---------------------------------------------------------------------------------------------------------------
# Seeded problems

def random_poly(rng: random.Random, degree: int, nonconstant: bool = False) -> list:
    """A polynomial of degree at most degree with coefficients in [-3, 3]; nonconstant ones have degree >= 1."""
    degree = max(degree, 1) if nonconstant else degree
    while True:
        p = trim([F(rng.randint(-3, 3)) for _ in range(degree + 1)])
        if not nonconstant or len(p) > 1:
            return p


def random_problem(rng: random.Random, operation: str) -> tuple[tuple, tuple, str]:
    """(lhs, rhs, relation) of a seeded problem in the class of the operation."""
    relation = rng.choice(RELATIONS)
    if operation == "solve_abs":
        atoms = [random_poly(rng, rng.choice((1, 1, 2)), nonconstant=rng.random() < 0.95)
                 for _ in range(rng.choice((0, 1, 1, 2, 2, 2)))]
        terms = []
        for atom in atoms:
            node = ab(poly_node(atom))
            shape = rng.random()
            if shape < 0.15:
                node = mul(poly_node(random_poly(rng, 1)), node)
            elif shape < 0.25:
                node = pw(node, 2)
            elif shape < 0.35:
                node = dv(node, rng.choice((2, -3)))
            terms.append(mul(rng.choice((1, 1, -1, 2, F(1, 2))), node))
        terms.append(poly_node(random_poly(rng, rng.choice((0, 1, 1, 2)))))
        if len(atoms) == 2 and rng.random() < 0.15:
            terms.append(mul(ab(poly_node(atoms[0])), ab(poly_node(atoms[1]))))  # A product of two atoms.
        rng.shuffle(terms)
        cut = rng.randint(1, len(terms))
        lhs = add(*terms[:cut])
        rhs = mul(-1, add(*terms[cut:])) if terms[cut:] else lift(rng.randint(-2, 2))
        return lhs, rhs, relation
    radicand = random_poly(rng, rng.choice((0, 1, 1, 2, 2)), nonconstant=rng.random() < 0.9)
    root = sq(poly_node(radicand))
    a1, a2 = random_poly(rng, rng.choice((0, 1))), random_poly(rng, rng.choice((0, 0, 1)))
    b1, b2 = random_poly(rng, rng.choice((0, 1, 2))), random_poly(rng, rng.choice((0, 1)))
    lhs = add(mul(poly_node(a1), root), poly_node(b1))
    if rng.random() < 0.2:
        lhs = add(lhs, dv(root, 2))  # The radicand a second time.
    rhs = add(mul(poly_node(a2), root), poly_node(b2)) if rng.random() < 0.5 else poly_node(b2)
    return lhs, rhs, relation


def seeded(operation: str, seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        yield random_problem(rng, operation)


def evidence_of(cuts: tuple, samples: tuple) -> tuple:
    rational = (lambda value: Q(F(value).numerator, F(value).denominator))
    return tuple(rational(c) for c in cuts), tuple(rational(s) for s in samples)


class PiecewiseTests(support.FamilyTestCase):
    FAMILY = family

    def assertVerified(self, result: mf.Result, value: object) -> None:
        """A completed v6 Result of that value, with its four checks, which mf.verify accepts."""
        self.assertEqual(result.value, value, result.error)
        self.assertIs(type(result.value), type(value))
        self.assertEqual(tuple(check.claim for check in result.verification), support.CLAIMS)
        report = mf.verify(result)
        self.assertTrue(report.verified, support.statuses(report))

    def test_reference_on_the_known_values(self):
        """The reference of this module gives the card's known values (its irrational ends by value)."""
        for operation, lhs, rhs, relation, value in KNOWN:
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                self.assertTrue(agrees(value, reference(operation, lift(lhs), lift(rhs), relation)))

    def test_known_values(self):
        """Known values of every operation, each from a Result that mf.verify accepts."""
        for operation, lhs, rhs, relation, value in KNOWN:
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                result = support.run(operation, **arguments(lhs, rhs, relation))
                self.assertVerified(result, value)
                self.roundtrip(result)
        for operation, lhs, rhs, relation, evidence in KNOWN_EVIDENCE:
            with self.subTest(operation=operation, evidence=evidence):
                result = support.run(operation, **arguments(lhs, rhs, relation))
                self.assertEqual(result.steps[0].outputs[1], evidence_of(*evidence))
        # From the parser, whose trees are simplified: the same values.
        for operation, lhs, rhs, relation, value in (
                ("solve_abs", "abs(x - 1) + abs(x + 2)", "3", "=", solution(closed(-2, 1))),
                ("solve_abs", "abs(x^2 - 2)", "x", "<", solution(opened(1, 2))),
                ("solve_abs", "sqrt(x^2)", "x", "=", solution(Interval(Q(0), None, True, False))),  # abs(x) = x.
                ("solve_sqrt", "sqrt(x + 2)", "x", "=", points(2)),
                ("solve_sqrt", "2*sqrt(x) + sqrt(x)", "3", "=", points(1)),
                ("solve_sqrt", "sqrt(-x^2)", "0", "=", points(0))):
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                result = support.run(operation, lhs=parsed(lhs), rhs=parsed(rhs), variable=XS, relation=relation)
                self.assertVerified(result, value)
        # The default relation is "=".
        self.assertVerified(support.run("solve_abs", lhs=function(ab(sub(X, 1))), rhs=function(3), variable=XS),
                            points(-2, 4))

    def test_every_relation(self):
        """Each relation on problems whose sets have open, closed and degenerate pieces."""
        for operation, lhs, rhs in (("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3),
                                    ("solve_abs", ab(sub(pw(X, 2), 2)), 1),
                                    ("solve_sqrt", sq(add(X, 2)), X),
                                    ("solve_sqrt", mul(sub(X, 1), sq(sub(4, pw(X, 2)))), 0)):
            for relation in RELATIONS:
                with self.subTest(operation=operation, lhs=lhs, relation=relation):
                    result = support.run(operation, **arguments(lhs, rhs, relation))
                    self.assertTrue(agrees(result.value, reference(operation, lift(lhs), lift(rhs), relation)),
                                    mf.render_text(result.value))
                    self.assertTrue(mf.verify(result).verified)

    def test_invalid_inputs(self):
        """0.5, True, "1", None and a wrong container are invalid_input, never the producer's internal_error, in
        every argument; so is every input outside the class of the card. produce() tests the types itself too."""
        for operation in OPERATIONS:
            base = arguments(ab(X) if operation == "solve_abs" else sq(X), 1, "<")
            for name in base:
                for item in STAND_INS + (support.oversized_matrix(3), function(1, (YS,))):
                    with self.subTest(operation=operation, argument=name, item=item):
                        self.assertInvalidInput(support.run(operation, **dict(base, **{name: item})))
                with self.subTest(operation=operation, argument=name, produce=True):
                    with self.assertRaises(InvalidInput):
                        family.produce(operation, dict(base, **{name: 0.5}))
            for variable in (mf.symbol("x", Integers), YS):  # A constrained symbol, another symbol.
                with self.subTest(operation=operation, variable=variable):
                    self.assertInvalidInput(support.run(operation, **dict(base, variable=variable)))
        for operation, lhs, rhs, relation in INVALID:
            with self.subTest(operation=operation, lhs=lhs, relation=relation):
                result = support.run(operation, **arguments(lhs, rhs, relation))
                self.assertInvalidInput(result)
        # A forged tree that contains itself, and a forged Rational constant out of lowest terms.
        loop = support.forged(RealExpr, op="add", value=None)
        object.__setattr__(loop, "args", (loop, RealExpr("abs", (_X_EXPR,))))
        looped = support.forged(RealFunction, expression=loop, variables=(XS,), domain=DomainPredicate("true"))
        half = RealExpr("scalar", (), support.forged(Q, numerator=2, denominator=4))
        unreduced = support.forged(RealFunction, expression=RealExpr("abs", (half,)), variables=(XS,),
                                   domain=DomainPredicate("true"))
        for lhs in (looped, unreduced, support.forged(RealFunction, expression=RealExpr("abs", (_X_EXPR,)))):
            with self.subTest(lhs=type(lhs)):
                with self.assertRaises(InvalidInput):
                    family.produce("solve_abs", dict(lhs=lhs, rhs=function(1), variable=XS, relation="="))
        with self.assertRaises(UnsupportedOperation):
            family.produce("solve_poly", arguments(ab(X), 1))
        with self.assertRaises(InvalidInput):
            family.produce("solve_abs", dict(arguments(ab(X), 1), extra=1))

    def test_resource_limits(self):
        """An integer of 2**65536 (a constant, an exponent), a 65 x 65 matrix where a function belongs, max_work=1,
        a factor of degree above 64 and a huge degree exceed the budget, each fast."""
        huge = RealExpr("scalar", (), Q(support.HUGE))
        for operation, lhs in (("solve_abs", RealFunction(RealExpr("abs", (huge,)), (XS,))),
                               ("solve_abs", RealFunction(RealExpr("abs", (RealExpr("pow", (_X_EXPR,), support.HUGE),)),
                                                          (XS,))),
                               ("solve_sqrt", RealFunction(RealExpr("add", (_SQRT_X, huge)), (XS,)))):
            with self.subTest(operation=operation, lhs=lhs):
                started = time.perf_counter()
                self.assertResourceLimit(support.run(operation, lhs=lhs, rhs=function(1), variable=XS))
                self.assertLess(time.perf_counter() - started, FAST)
        for operation in OPERATIONS:
            base = arguments(ab(sub(X, 1)) if operation == "solve_abs" else sq(add(X, 2)), X, "<")
            with self.subTest(operation=operation, limits="max_work=1"):
                self.assertResourceLimit(support.run(operation, limits=ComputationLimits(max_work=1), **base))
            # The family takes no Matrix: an oversized one is refused as no RealFunction, before any work.
            with self.subTest(operation=operation, matrix=True):
                self.assertInvalidInput(support.run(operation, **dict(base, lhs=support.oversized_matrix())))
        for lhs in (ab(sub(pw(X, 70), 2)), ab(sub(pw(X, 10 ** 6), 2)), sq(sub(pw(X, 70), 2))):
            operation = "solve_abs" if lhs[0] == "abs" else "solve_sqrt"
            with self.subTest(operation=operation, degree=lhs[1][1][2]):
                started = time.perf_counter()
                self.assertResourceLimit(support.run(operation, **arguments(lhs, 1)))
                self.assertLess(time.perf_counter() - started, FAST)
        with self.assertRaises(ResourceLimitError):
            with computation(ComputationLimits(max_work=5), fresh=True):
                family.produce("solve_abs", arguments(ab(sub(X, 1)), 3))

    def test_checker_accepts_true_claims_the_producer_did_not_make(self):
        """The checker accepts every true claim, also with evidence that the producer would not have chosen."""
        args = arguments(ab(sub(X, 1)), 3)
        value = points(-2, 4)
        for evidence in (((-2, 1, 4), (-3, 0, 2, 5)), ((-2, 4), (-3, 0, 2, 5)), ((-2, 1, 4), (-100, F(1, 2), 3, 9)),
                         ((), (-7, F(-1, 3), F(5, 2), 6))):
            with self.subTest(evidence=evidence):
                self.assertAccepts("solve_abs", args, value, evidence_of(*evidence))
        args = arguments(sq(add(X, 2)), X)
        for evidence in (((-2, -1, 0, 2), (-3, F(-3, 2), F(-1, 2), 1, 3)),
                         ((-2, 2), (-9, F(-7, 4), F(-1, 3), F(1, 7), 100))):
            with self.subTest(evidence=evidence):
                self.assertAccepts("solve_sqrt", args, points(2), evidence_of(*evidence))
        # sqrt(x^2 - 2) < 1 holds on (-sqrt(3), -sqrt(2)] and [sqrt(2), sqrt(3)): irrational ends only, no cut.
        args = arguments(sq(sub(pw(X, 2), 2)), 1, "<")
        root2 = (RealAlgebraicRoot((-2, 0, 1), 0), RealAlgebraicRoot((-2, 0, 1), 1))
        value = solution(Interval(ROOT3[0], root2[0], False, True), Interval(root2[1], ROOT3[1], True, False))
        self.assertTrue(agrees(value, reference("solve_sqrt", sq(sub(pw(X, 2), 2)), lift(1), "<")))
        self.assertAccepts("solve_sqrt", args, value, evidence_of((), (-2, F(-3, 2), 0, F(3, 2), 2)))
        self.assertAccepts("solve_sqrt", args, value, evidence_of((), (-3, F(-8, 5), 0, F(8, 5), 9)))

    def test_checker_rejects_corrupted_claims(self):
        """The checker answers False to a wrong value, a value that is not canonical, an impostor class, a record
        forged with object.__new__, a bool, a float or a str where an int belongs, a list for a tuple, missing or
        extra evidence, a wrong operation name and oversized forged evidence."""
        args = arguments(ab(sub(X, 1)), 3)
        value = points(-2, 4)
        evidence = evidence_of((-2, 1, 4), (-3, 0, 2, 5))
        cuts, samples = evidence
        forged_set = support.forged(IntervalSet, intervals=tuple(reversed(value.intervals)))
        corruptions = (
            ("solve_abs", args, points(-2, 5), evidence),
            ("solve_abs", args, points(-2), evidence),
            ("solve_abs", args, solution(closed(-2, 4)), evidence),
            ("solve_abs", args, EMPTY, evidence),
            ("solve_abs", args, ALL, evidence),
            ("solve_abs", args, FiniteSet((Q(-2), Q(4))), evidence),  # Not the canonical form.
            ("solve_abs", args, forged_set, evidence),  # Unsorted.
            ("solve_abs", args, support.impostor(value), evidence),
            ("solve_abs", args, (Q(-2), Q(4)), evidence),
            ("solve_abs", args, value, (cuts,)),  # Missing evidence.
            ("solve_abs", args, value, (cuts, samples, ())),  # Extra evidence.
            ("solve_abs", args, value, ()),
            ("solve_abs", args, value, [cuts, samples]),  # A list for a tuple.
            ("solve_abs", args, value, (list(cuts), samples)),
            ("solve_abs", args, value, ((-2, 1, 4), samples)),  # ints for Rationals.
            ("solve_abs", args, value, ((Q(-2), True, Q(4)), samples)),
            ("solve_abs", args, value, ((Q(-2), 1.0, Q(4)), samples)),
            ("solve_abs", args, value, ((Q(-2), "1", Q(4)), samples)),
            ("solve_abs", args, value, (cuts, samples[:-1])),  # A sample short.
            ("solve_abs", args, value, (cuts, samples[:2] + (Q(1),) + samples[3:])),  # A sample at a point.
            ("solve_abs", args, value, ((Q(-2), Q(1), Q(3), Q(4)), samples)),  # A cut that is no root.
            ("solve_abs", args, value, (cuts, tuple(Q(k) for k in range(-50_000, 50_000)))),  # Oversized.
            ("solve_poly", args, value, evidence),  # A wrong operation name.
            ("solve_sqrt", args, value, evidence),  # The other operation: abs lies outside its class.
            ("solve_abs", dict(args, relation="<"), value, evidence),
        )
        for operation, request, claim, proof in corruptions:
            with self.subTest(operation=operation, value=claim, evidence=str(proof)[:80]):
                self.assertRejects(operation, request, claim, proof, may_exhaust=True)
        # An irrational end named by a polynomial that is not minimal: (x^2 - 3)(x - 5) has the rational root 5.
        args = arguments(ab(sub(pw(X, 2), 2)), 1, "<=")
        wide = (RealAlgebraicRoot((15, -3, -5, 1), 0), RealAlgebraicRoot((15, -3, -5, 1), 1))
        evidence = evidence_of((-1, 1), (-2, F(-3, 2), F(-6, 5), 0, F(6, 5), F(3, 2), 2))
        self.assertAccepts("solve_abs", args, solution(closed(ROOT3[0], -1), closed(1, ROOT3[1])), evidence)
        self.assertRejects("solve_abs", args, solution(closed(wide[0], -1), closed(1, wide[1])), evidence)

    def test_verification_never_runs_the_producer(self):
        """mf.verify passes while produce() raises, on the Result and on the one decoded from its JSON text."""
        results = [support.run(operation, **arguments(lhs, rhs, relation))
                   for operation, lhs, rhs, relation, _ in KNOWN[::3]]
        with patch.object(family, "produce", side_effect=AssertionError("the producer ran")):
            for result in results:
                with self.subTest(value=mf.render_text(result.value)):
                    self.assertTrue(mf.verify(result).verified)
                    self.assertTrue(mf.verify(mf.loads(mf.dumps(result))).verified)
                    self.assertFalse(mf.verify(support.tampered(result, points(7))).verified)
            self.assertIsNotNone(support.run("solve_abs", **arguments(ab(X), 1)).error)  # The patch holds.

    def test_json_and_rendering(self):
        """A Result survives the JSON v6 round trip, and its value renders as text and as LaTeX."""
        for operation, lhs, rhs, relation, text, latex in (
                ("solve_abs", ab(X), -1, "=", "{}", "\\varnothing"),
                ("solve_abs", ab(X), 0, ">=", "All(reals)", "\\mathbb{R}"),
                ("solve_abs", ab(sub(X, 1)), 3, "=", "[-2, -2] union [4, 4]", "[-2, -2] \\cup [4, 4]"),
                ("solve_sqrt", sq(sub(X, 1)), sub(X, 3), "<", "(5, inf)", "(5, \\infty)"),
                ("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=",
                 "[RootOf(t^2 - 3; real_index=0), -1] union [1, RootOf(t^2 - 3; real_index=1)]",
                 "[\\operatorname{RootOf}_{\\mathbb{R}}\\left(t^{2} - 3; 0\\right), -1] \\cup "
                 "[1, \\operatorname{RootOf}_{\\mathbb{R}}\\left(t^{2} - 3; 1\\right)]")):
            with self.subTest(operation=operation, text=text):
                result = support.run(operation, **arguments(lhs, rhs, relation))
                decoded = self.roundtrip(result)
                self.assertEqual(mf.render_text(decoded.value), text)
                self.assertEqual(mf.to_latex(decoded.value), latex)

    def test_value_function_equals_the_structured_path(self):
        """On seeded inputs the value function, on its lean path, returns the value of the structured path."""
        for operation in OPERATIONS:
            for lhs, rhs, relation in seeded(operation, 404, 12):
                with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                    args = arguments(lhs, rhs, relation)
                    expected = support.run(operation, **args).value
                    found = support.lean_value(operation, **args)
                    if found is not lean.FALLBACK:
                        self.assertEqual(found, expected)
                    self.assertEqual(support.value(operation, **args), expected)
            with self.subTest(operation=operation, invalid=True):  # The structured path's error.
                with self.assertRaises(mf.OperationError) as raised:
                    support.value(operation, **arguments(sq(X) if operation == "solve_abs" else ab(X), 1))
                self.assertEqual(raised.exception.code, "invalid_input")

    def test_mcp_area_tool(self):
        """The MCP area tool runs the operation and returns the Result of the structured path."""
        for operation, lhs, rhs, relation in (("solve_abs", "abs(x^2 - 2)", "1", "<="),
                                              ("solve_sqrt", "sqrt(x + 2)", "x", "=")):
            with self.subTest(operation=operation):
                args = dict(lhs=parsed(lhs), rhs=parsed(rhs), variable=XS, relation=relation)
                expected = support.run(operation, **args)
                result = self.answer(support.call_values(operation, **args))
                self.assertEqual(result.value, expected.value)
                self.assertTrue(mf.verify(result).verified)
                # The JSON request written by hand: the values agree, the symbol's identity aside.
                response = support.call(operation, {"lhs": {"text": lhs, "variables": ["x"]},
                                                    "rhs": {"text": rhs, "variables": ["x"]},
                                                    "variable": "x", "relation": relation})
                result = self.answer(response)
                self.assertEqual(mf.render_text(result.value), mf.render_text(expected.value))
                self.assertTrue(mf.verify(result).verified)
        error = self.assertToolError(support.call("solve_sqrt", {"lhs": {"text": "abs(x)", "variables": ["x"]},
                                                                 "rhs": {"text": "1", "variables": []},
                                                                 "variable": "x"}), "invalid_input")
        self.assertIn("abs", error["message"])

    def test_seeded_agreement_with_the_reference(self):
        """On SEEDED seeded problems per operation, produce() succeeds, its value equals the reference's (irrational
        ends by value), and the checker accepts its claim; each irrational end is named by a polynomial without a
        rational root."""
        for operation, seed in (("solve_abs", 1729), ("solve_sqrt", 2718)):
            irrational = 0
            for lhs, rhs, relation in seeded(operation, seed, SEEDED):
                with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                    args = arguments(lhs, rhs, relation)
                    value, evidence = family.produce(operation, args)
                    expected = reference(operation, lhs, rhs, relation)
                    self.assertTrue(agrees(value, expected), (mf.render_text(value), expected))
                    self.assertIs(support.verdict(family, operation, args, value, evidence), True)
                    if type(value) is IntervalSet:
                        ends = [e for i in value.intervals for e in (i.lower, i.upper)]
                        irrational += sum(type(e) is RealAlgebraicRoot for e in ends)
            self.assertGreater(irrational, 10, operation)  # The seeds reach irrational ends.

