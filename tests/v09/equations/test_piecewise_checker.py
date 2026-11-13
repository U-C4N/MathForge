"""Tests of the independent checker of mathforge.areas.equations.piecewise (mathforge.certify.equations.piecewise)
alone, written before the family's producer exists. Every claim and its evidence come from the reference of this
module, never from a producer, and the checker answers them as the verifier asks (support.verdict, in a fresh budget
of the default limits unless a test passes others).

The reference. Problems are written in a small expression language of tuples (("x",), ("num", q), ("add", a, b),
("mul", a, b), ("pow", a, n), ("div", a, b), ("abs", a), ("sqrt", a)) that builds the RealExpr trees the checker reads,
unsimplified, and that this module evaluates itself with Fractions. Its own polynomials over Fraction give the problem
polynomials: for solve_abs the atoms and the pieces of the sign patterns on the cells of the atoms (the card's H0
cells, a route the checker does not take), for solve_sqrt P, A, B and N = A^2 P - B^2. Their real roots come from the
rational root theorem and, for the rest, from Sturm sequences and bisection. The truth at every sample is an exact
evaluation of the expression (in Q(sqrt p) for a square root); at a rational point as well; at an irrational point it
is the sign of a piece (solve_abs) or the signs of P, A, B and N (solve_sqrt), found by a gcd and a refined isolating
interval. The claim is assembled from the truth values in canonical form, with Rational ends and RealAlgebraicRoot ends
named by the residual of a problem polynomial (its square-free part without its rational roots), and the evidence is
(cuts, samples) in the layout of the checker's module docstring: every rational point, and one sample per open cell.

The checker answers False to every corrupted claim, evidence or argument, never with an exception: only an integer of
the arguments longer than max_integer_bits and an exhausted budget raise ResourceLimitError. It refuses an oversized
forged value or evidence within FAST seconds under the default budget.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
import math
import random
import time

import mathforge as mf
from mathforge.algebraic import RealAlgebraicRoot
from mathforge.areas.equations import piecewise as family
from mathforge.contracts import EmptySet, FiniteSet, UniversalSet
from mathforge.functions import DomainPredicate, RealExpr, RealFunction, parse_function
from mathforge.intervals import Interval, IntervalSet
from mathforge.limits import ComputationLimits, ResourceLimitError, computation
from mathforge.model import Integers, Rationals, Reals, trusted_rational

from .. import support

Q = mf.Rational
F = Fraction
FAST = 3.0  # Seconds: an oversized forgery is refused well within this under the default budget.
RELATIONS = ("=", "<", "<=", ">", ">=")
XS = mf.symbol("x")
YS = mf.symbol("y")
GIANT = 2 ** 70_000  # Longer than max_integer_bits allows.


# ---------------------------------------------------------------------------------------------------------------
# The expression language

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


def args_of(lhs: object, rhs: object, relation: str = "=") -> dict:
    return {"lhs": function(lhs), "rhs": function(rhs), "variable": XS, "relation": relation}


# ---------------------------------------------------------------------------------------------------------------
# Polynomials over Fraction: lists, lowest degree first, no trailing zero

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


def ppow(p: list, n: int) -> list:
    out = [F(1)]
    for _ in range(n):
        out = pmul(out, p)
    return out


def peval(p: list, x: Fraction) -> Fraction:
    value = F(0)
    for c in reversed(p):
        value = value * x + c
    return value


def pdivmod(p: list, q: list) -> tuple[list, list]:
    p, out = list(p), [F(0)] * max(len(p) - len(q) + 1, 1)
    while len(p) >= len(q) and p:
        c, k = p[-1] / q[-1], len(p) - len(q)
        out[k] = c
        p = psub(p, [F(0)] * k + pscale(q, c))
    return trim(out), p


def pderiv(p: list) -> list:
    return trim([i * c for i, c in enumerate(p)][1:])


def pgcd(p: list, q: list) -> list:
    while q:
        p, q = q, pdivmod(p, q)[1]
    return pscale(p, 1 / p[-1]) if p else []


def psqfree(p: list) -> list:
    g = pgcd(p, pderiv(p))
    return pdivmod(p, g)[0] if len(g) > 1 else p


def sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def integer(p: list) -> tuple[int, ...]:
    """The primitive integer polynomial with a positive leading coefficient: the form RealAlgebraicRoot stores."""
    lcm = reduce(lambda a, b: a * b // math.gcd(a, b), [c.denominator for c in p], 1)
    values = [int(c * lcm) for c in p]
    content = reduce(math.gcd, values) * (1 if values[-1] > 0 else -1)
    return tuple(v // content for v in values)


def divisors(n: int) -> list[int]:
    n = abs(n)
    return [d for d in range(1, n + 1) if n % d == 0]


def rational_roots(p: list) -> list[Fraction]:
    """The distinct rational roots of a nonzero polynomial with small coefficients (the rational root theorem)."""
    p = trim(p)
    roots = set()
    while len(p) > 1 and p[0] == 0:
        roots.add(F(0))
        p = p[1:]
    if len(p) < 2:
        return sorted(roots)
    values = integer(p)
    for u in divisors(values[0]):
        for v in divisors(values[-1]):
            for candidate in (F(u, v), F(-u, v)):
                if peval(p, candidate) == 0:
                    roots.add(candidate)
    return sorted(roots)


def residual(p: list) -> list:
    """The square-free part of p without its rational roots."""
    p = psqfree(p)
    for root in rational_roots(p):
        p = pdivmod(p, [-root, F(1)])[0]
    return p


def sturm(p: list) -> list[list]:
    chain = [p, pderiv(p)]
    while len(chain[-1]) > 1:
        rest = pdivmod(chain[-2], chain[-1])[1]
        if not rest:
            break
        chain.append(pscale(rest, -1))
    return chain


def variations(chain: list[list], x: Fraction) -> int:
    signs = [s for s in (sign(peval(member, x)) for member in chain) if s]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def count(p: list, lo: Fraction, hi: Fraction) -> int:
    """The number of distinct real roots in (lo, hi] of a square-free p vanishing at neither end."""
    chain = sturm(p)
    return variations(chain, lo) - variations(chain, hi)


def isolate(p: list) -> list[tuple[Fraction, Fraction]]:
    """Disjoint intervals (lo, hi), each holding one real root of a square-free p without rational roots."""
    if len(p) < 2:
        return []
    m = 1 + max(abs(c / p[-1]) for c in p[:-1])
    found, stack = [], [(-m, m)]
    while stack:
        lo, hi = stack.pop()
        n = count(p, lo, hi)
        if n == 1:
            found.append((lo, hi))
        elif n > 1:
            stack += [(lo, (lo + hi) / 2), ((lo + hi) / 2, hi)]
    return sorted(found)


class Root:
    """An irrational real root: the one root of the square-free s in (lo, hi), named RealAlgebraicRoot(name, index)."""

    def __init__(self, s: list, lo: Fraction, hi: Fraction, name: tuple, index: int) -> None:
        self.s, self.lo, self.hi, self.name, self.index = s, lo, hi, name, index

    def refine(self) -> None:
        mid = (self.lo + self.hi) / 2
        if sign(peval(self.s, self.lo)) != sign(peval(self.s, mid)):
            self.hi = mid
        else:
            self.lo = mid

    def record(self) -> RealAlgebraicRoot:
        return RealAlgebraicRoot(self.name, self.index)


def low(point: object) -> Fraction:
    return point if type(point) is Fraction else point.lo


def high(point: object) -> Fraction:
    return point if type(point) is Fraction else point.hi


def points_of(polynomials: list[list]) -> list:
    """The distinct real roots of nonzero polynomials, increasing: Fractions and Roots with disjoint intervals."""
    polynomials = [p for p in polynomials if len(p) > 1]
    rationals = sorted({root for p in polynomials for root in rational_roots(p)})
    residuals = [r for r in (residual(p) for p in polynomials) if len(r) > 1]
    roots = []
    if residuals:
        product = psqfree(reduce(pmul, residuals))
        for lo, hi in isolate(product):
            s = next(r for r in residuals if count(r, lo, hi) == 1)  # The interval holds one root of the product.
            m = 1 + max(abs(c / s[-1]) for c in s[:-1])
            roots.append(Root(s, lo, hi, integer(s), count(s, -m, lo)))
    changed = True
    while changed:  # Refine until no interval holds a rational point or meets another interval.
        changed = False
        for root in roots:
            while any(root.lo <= q <= root.hi for q in rationals) or any(
                    other is not root and root.lo <= other.hi and other.lo <= root.hi for other in roots):
                root.refine()
                for other in roots:
                    if other is not root and root.lo <= other.hi and other.lo <= root.hi:
                        other.refine()
                changed = True
    return sorted(rationals + roots, key=low)


def simplest(a: Fraction, b: Fraction) -> Fraction:
    """A rational of small height strictly between a < b."""
    m = math.floor(a)
    if m + 1 < b:
        return F(m + 1)
    lo = 1 / (b - m)
    y = F(math.floor(lo) + 1) if a == m else simplest(lo, 1 / (a - m))
    return m + 1 / y


def samples_of(points: list) -> list[Fraction]:
    if not points:
        return [F(0)]
    out = [F(math.floor(low(points[0])) - 1)]
    for left, right in zip(points, points[1:]):
        out.append(simplest(high(left), low(right)))
    out.append(F(math.floor(high(points[-1])) + 1))
    return out


def sign_at_point(p: list, point: object) -> int:
    """The sign of a polynomial at a point: exact at a rational; at a Root, 0 when gcd(p, s) changes sign across its
    interval, otherwise the sign at the interval's end once no root of p is left in it."""
    if type(point) is Fraction or not p:
        return sign(peval(p, point)) if p else 0
    g = pgcd(p, point.s)
    if len(g) > 1 and sign(peval(g, point.lo)) != sign(peval(g, point.hi)):
        return 0
    if len(p) == 1:
        return sign(p[0])
    part, lo, hi = psqfree(p), point.lo, point.hi
    while not peval(part, lo) or not peval(part, hi) or count(part, lo, hi):
        mid = (lo + hi) / 2
        if sign(peval(point.s, lo)) != sign(peval(point.s, mid)):
            hi = mid
        else:
            lo = mid
    return sign(peval(p, lo))


# ---------------------------------------------------------------------------------------------------------------
# The reference solver

def evaluate(node: tuple, x: Fraction) -> Fraction:
    """The exact value of an expression free of sqrt at a rational point."""
    op = node[0]
    if op == "x":
        return x
    if op == "num":
        return node[1]
    if op == "add":
        return evaluate(node[1], x) + evaluate(node[2], x)
    if op == "mul":
        return evaluate(node[1], x) * evaluate(node[2], x)
    if op == "pow":
        return evaluate(node[1], x) ** node[2]
    if op == "div":
        return evaluate(node[1], x) / evaluate(node[2], x)
    if op == "abs":
        return abs(evaluate(node[1], x))
    raise ValueError(op)


def polynomial(node: tuple, signs=None) -> list:
    """The polynomial of an expression, every abs(P) replaced by signs(P) * P."""
    op = node[0]
    if op == "x":
        return [F(0), F(1)]
    if op == "num":
        return trim([node[1]])
    if op == "add":
        return padd(polynomial(node[1], signs), polynomial(node[2], signs))
    if op == "mul":
        return pmul(polynomial(node[1], signs), polynomial(node[2], signs))
    if op == "pow":
        return ppow(polynomial(node[1], signs), node[2])
    if op == "div":
        return pscale(polynomial(node[1], signs), 1 / polynomial(node[2], signs)[0])
    if op == "abs":
        inner = polynomial(node[1], signs)
        return pscale(inner, signs(tuple(inner)))
    raise ValueError(op)


def atoms_of(node: tuple) -> list[list]:
    if node[0] == "abs":
        return [polynomial(node[1])]
    if node[0] in ("x", "num"):
        return []
    found = []
    for child in node[1:]:
        if type(child) is tuple:
            for atom in atoms_of(child):
                if atom not in found:
                    found.append(atom)
    return found


def has_sqrt(node: tuple) -> bool:
    return node[0] == "sqrt" or any(type(child) is tuple and has_sqrt(child) for child in node[1:])


def linear(node: tuple) -> tuple[list, list, list | None]:
    """(A, B, P) with node = A*sqrt(P) + B."""
    op = node[0]
    if op == "sqrt":
        return [F(1)], [], polynomial(node[1])
    if not has_sqrt(node):
        return [], polynomial(node), None
    if op == "add":
        (a1, b1, p1), (a2, b2, p2) = linear(node[1]), linear(node[2])
        return padd(a1, a2), padd(b1, b2), p1 if p1 is not None else p2
    if op == "mul":
        (a1, b1, p1), (a2, b2, p2) = linear(node[1]), linear(node[2])
        p = p1 if p1 is not None else p2
        return padd(pmul(a1, b2), pmul(a2, b1)), padd(pmul(b1, b2), pmul(pmul(a1, a2), p or [])), p
    if op == "div":
        a, b, p = linear(node[1])
        c = 1 / polynomial(node[2])[0]
        return pscale(a, c), pscale(b, c), p
    if op == "pow":
        a, b, p = linear(node[1])
        return ([], [F(1)], p) if node[2] == 0 else (a, b, p)
    raise ValueError(op)


def surd(node: tuple, x: Fraction):
    """(a, b, p) with the value a + b*sqrt(p) at a rational point, or None where a radicand is negative."""
    held = []

    def walk(node):
        op = node[0]
        if op == "x":
            return x, F(0)
        if op == "num":
            return node[1], F(0)
        if op == "sqrt":
            radicand = walk(node[1])[0]
            if radicand < 0:
                raise LookupError
            held.append(radicand)
            return F(0), F(1)
        if op == "add":
            (a1, b1), (a2, b2) = walk(node[1]), walk(node[2])
            return a1 + a2, b1 + b2
        if op == "mul":
            (a1, b1), (a2, b2) = walk(node[1]), walk(node[2])
            p = held[0] if held else F(0)
            return a1 * a2 + b1 * b2 * p, a1 * b2 + a2 * b1
        if op == "div":
            (a, b), (c, _) = walk(node[1]), walk(node[2])
            return a / c, b / c
        if op == "pow":
            out = (F(1), F(0))
            for _ in range(node[2]):
                (a1, b1), (a2, b2) = out, walk(node[1])
                p = held[0] if held else F(0)
                out = (a1 * a2 + b1 * b2 * p, a1 * b2 + a2 * b1)
            return out
        raise ValueError(op)

    try:
        a, b = walk(node)
    except LookupError:
        return None
    return a, b, (held[0] if held else F(0))


def surd_sign(a: Fraction, b: Fraction, p: Fraction) -> int:
    sb = sign(b) if p else 0
    if not sb:
        return sign(a)
    if not sign(a) or sign(a) == sign(b):
        return sb
    return sign(a) * sign(a * a - b * b * p)


def decision(sp: int, sa: int, sb: int, sn: int):
    if sp < 0:
        return None
    t = sa if sp > 0 else 0
    if not t:
        return sb
    return t if sb in (0, t) else t * sn


def assemble(points: list, cell_truth: list, point_truth: list) -> object:
    """The canonical solution set of the truth values, in the order cell 0, point 1, cell 1, ..., point r, cell r."""
    def end(point):
        return Q(point.numerator, point.denominator) if type(point) is Fraction else point.record()

    members = [cell_truth[0]]
    for k in range(len(points)):
        members += [point_truth[k], cell_truth[k + 1]]
    last, intervals, i = len(members) - 1, [], 0
    while i <= last:
        if not members[i]:
            i += 1
            continue
        j = i
        while j < last and members[j + 1]:
            j += 1
        if i % 2:  # Member 2k - 1 is the point number k, member 2k the cell number k.
            lower = (end(points[i // 2]), True)
        else:
            lower = (None, False) if i == 0 else (end(points[i // 2 - 1]), False)
        if j % 2:
            upper = (end(points[j // 2]), True)
        else:
            upper = (None, False) if j == last else (end(points[j // 2]), False)
        intervals.append(Interval(lower[0], upper[0], lower[1], upper[1]))
        i = j + 1
    return IntervalSet(tuple(intervals)).to_solution_set()


class Solution:
    """The reference's answer to one problem: the claim, its evidence, and what built them."""

    def __init__(self, points: list, samples: list, cell_truth: list, point_truth: list) -> None:
        self.points, self.samples = points, samples
        self.cell_truth, self.point_truth = cell_truth, point_truth
        self.value = assemble(points, cell_truth, point_truth)
        self.cuts = tuple(Q(p.numerator, p.denominator) for p in points if type(p) is Fraction)
        self.evidence = (self.cuts, tuple(Q(s.numerator, s.denominator) for s in samples))

    def flipped(self, member: int) -> object:
        """The set with the truth of one member (cell 0, point 1, cell 1, ...) reversed: a wrong claim whose ends are
        points of the same cut, so that the same evidence serves it."""
        cells, points = list(self.cell_truth), list(self.point_truth)
        if member % 2:
            points[member // 2] = not points[member // 2]
        else:
            cells[member // 2] = not cells[member // 2]
        return assemble(self.points, cells, points)


def solve_abs(lhs: object, rhs: object, relation: str) -> Solution:
    node = sub(lhs, rhs)
    atoms = atoms_of(node)
    atom_samples = samples_of(points_of(atoms))  # The card's H0 cells.

    def pattern_at(x):
        return {tuple(atom): sign(peval(atom, x)) for atom in atoms}

    pieces = [polynomial(node, pattern_at(s).__getitem__) for s in atom_samples]
    points = points_of(atoms + pieces)
    samples = samples_of(points)
    cell_truth = [holds(relation, sign(evaluate(node, s))) for s in samples]
    point_truth = []
    for k, point in enumerate(points):
        if type(point) is Fraction:
            point_truth.append(holds(relation, sign(evaluate(node, point))))
        else:  # F equals the piece of the cell below on its closure.
            piece = polynomial(node, pattern_at(samples[k]).__getitem__)
            point_truth.append(holds(relation, sign_at_point(piece, point)))
    return Solution(points, samples, cell_truth, point_truth)


def solve_sqrt(lhs: object, rhs: object, relation: str) -> Solution:
    node = sub(lhs, rhs)
    a, b, p = linear(node)
    n = psub(pmul(pmul(a, a), p), pmul(b, b))
    points = points_of([p, a, b, n])
    samples = samples_of(points)

    def truth_at(x):
        value = surd(node, x)
        return value is not None and holds(relation, surd_sign(*value))

    cell_truth = [truth_at(s) for s in samples]
    point_truth = []
    for point in points:
        if type(point) is Fraction:
            point_truth.append(truth_at(point))
        else:
            verdict = decision(*(sign_at_point(q, point) for q in (p, a, b, n)))
            point_truth.append(verdict is not None and holds(relation, verdict))
    return Solution(points, samples, cell_truth, point_truth)


def holds(relation: str, s: int) -> bool:
    return {"=": s == 0, "<": s < 0, "<=": s <= 0, ">": s > 0, ">=": s >= 0}[relation]


def solve(operation: str, lhs: object, rhs: object, relation: str) -> Solution:
    return (solve_abs if operation == "solve_abs" else solve_sqrt)(lhs, rhs, relation)


def partial_evidence(operation: str, lhs: object, rhs: object) -> tuple:
    """Evidence in the right layout for the cut of some of the problem polynomials only: the atoms of solve_abs (the
    card's H0 cells, without the pieces), and P, A and B of solve_sqrt (without N)."""
    node = sub(lhs, rhs)
    if operation == "solve_abs":
        polynomials = atoms_of(node)
    else:
        a, b, p = linear(node)
        polynomials = [p, a, b]
    points = points_of(polynomials)
    cuts = tuple(Q(q.numerator, q.denominator) for q in points if type(q) is Fraction)
    return cuts, tuple(Q(s.numerator, s.denominator) for s in samples_of(points))


def points_q(*values: object) -> tuple:
    return tuple(Q(F(v).numerator, F(v).denominator) for v in values)


def closed(a: object, b: object) -> Interval:
    return Interval(a, b, True, True)


def point(a: object) -> Interval:
    return Interval(a, a, True, True)


SQRT3 = (RealAlgebraicRoot((-3, 0, 1), 0), RealAlgebraicRoot((-3, 0, 1), 1))
SQRT2 = (RealAlgebraicRoot((-2, 0, 1), 0), RealAlgebraicRoot((-2, 0, 1), 1))
GOLDEN = RealAlgebraicRoot((1, -3, 1), 1)  # (3 + sqrt 5)/2, a root of x^2 - 3x + 1.

# The card's known values, (operation, lhs, rhs, relation, the set the card states).
KNOWN = (
    ("solve_abs", ab(sub(X, 1)), 3, "=", IntervalSet((point(Q(-2)), point(Q(4))))),
    ("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3, "=", IntervalSet((closed(Q(-2), Q(1)),))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), X, "<", IntervalSet((Interval(Q(1), Q(2)),))),
    ("solve_abs", ab(X), -1, "=", EmptySet(Reals)),
    ("solve_abs", ab(X), 0, ">=", UniversalSet(Reals)),
    ("solve_abs", ab(sub(X, 1)), 2, "<=", IntervalSet((closed(Q(-1), Q(3)),))),
    ("solve_sqrt", sq(add(X, 2)), X, "=", IntervalSet((point(Q(2)),))),
    ("solve_sqrt", sq(sub(X, 1)), sub(X, 3), "<", IntervalSet((Interval(Q(5), None),))),
    ("solve_sqrt", sq(X), -1, "=", EmptySet(Reals)),
    ("solve_sqrt", sub(sq(pw(X, 2)), X), 0, "=", IntervalSet((Interval(Q(0), None, True, False),))),
    ("solve_sqrt", sq(mul(-1, pw(X, 2))), 0, "=", IntervalSet((point(Q(0)),))),
    ("solve_sqrt", add(mul(2, sq(X)), sq(X)), 3, "=", IntervalSet((point(Q(1)),))),
)

# Further special cases, each with the set it must have.
SPECIAL = (
    # Irrational ends, and a piece whose roots lie outside its cell.
    ("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=",
     IntervalSet((closed(SQRT3[0], Q(-1)), closed(Q(1), SQRT3[1])))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), 0, ">",
     IntervalSet((Interval(None, SQRT2[0]), Interval(SQRT2[0], SQRT2[1]), Interval(SQRT2[1], None)))),
    ("solve_abs", ab(sub(pw(X, 2), 2)), 0, "=", IntervalSet((point(SQRT2[0]), point(SQRT2[1])))),
    # No atom at all: a polynomial relation.
    ("solve_abs", sub(pw(X, 2), 1), 0, "<", IntervalSet((Interval(Q(-1), Q(1)),))),
    # Constant functions; an abs of a constant and of zero (unsimplified trees).
    ("solve_abs", ab(-3), 3, "=", UniversalSet(Reals)),
    ("solve_abs", add(ab(0), X), 0, ">", IntervalSet((Interval(Q(0), None),))),
    # Division by a constant, a power 0, a product and a power of atoms.
    ("solve_abs", dv(ab(sub(X, 1)), add(1, 1)), pw(X, 0), "<", IntervalSet((Interval(Q(-1), Q(3)),))),
    ("solve_abs", mul(ab(X), ab(sub(X, 2))), 1, "=",
     IntervalSet((point(RealAlgebraicRoot((-1, -2, 1), 0)), point(Q(1)), point(RealAlgebraicRoot((-1, -2, 1), 1))))),
    ("solve_abs", pw(ab(sub(X, 1)), 2), sub(pw(X, 2), mul(2, X)), ">=", UniversalSet(Reals)),
    # The same atom twice, as P and as -P.
    ("solve_abs", add(ab(sub(X, 1)), ab(sub(1, X))), 4, "<", IntervalSet((Interval(Q(-1), Q(3)),))),
    # One irrational end of solve_sqrt; a radicand without real roots; a constant radicand.
    ("solve_sqrt", sq(X), sub(X, 1), "<", IntervalSet((Interval(GOLDEN, None),))),
    ("solve_sqrt", sq(add(pw(X, 2), 1)), 2, "=", IntervalSet((point(SQRT3[0]), point(SQRT3[1])))),
    ("solve_sqrt", X, sq(2), ">=", IntervalSet((Interval(SQRT2[1], None, True, False),))),
    # A vanishes identically, and a negative constant radicand (a FALSE domain).
    ("solve_sqrt", sub(sq(X), sq(X)), 0, "=", IntervalSet((Interval(Q(0), None, True, False),))),
    ("solve_sqrt", sq(-1), X, "<", EmptySet(Reals)),
    # Two spellings of one radicand, and a square root inside a division by a constant.
    ("solve_sqrt", add(sq(add(X, 1)), sq(add(1, X))), 2, "=", IntervalSet((point(Q(0)),))),
    ("solve_sqrt", dv(sq(X), 2), 1, ">", IntervalSet((Interval(Q(4), None),))),
)

# The claims that the corruption tests start from, (operation, lhs, rhs, relation): irrational ends, both kinds of
# end, and points only.
BASES = (("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<="),
         ("solve_sqrt", sq(X), sub(X, 1), "<"),
         ("solve_sqrt", sq(add(X, 2)), X, "="))


class PiecewiseCheckerTests(support.FamilyTestCase):
    FAMILY = family

    def accepts(self, operation: str, args: dict, value: object, evidence: object, **kwargs) -> None:
        self.assertAccepts(operation, args, value, evidence, **kwargs)

    def solved(self, operation: str, lhs: object, rhs: object, relation: str) -> tuple[dict, Solution]:
        return args_of(lhs, rhs, relation), solve(operation, lhs, rhs, relation)

    # --- True claims --------------------------------------------------------------------------------------------

    def test_known_values_of_the_card(self):
        """The card's known values: the reference finds the set the card states, and the checker accepts it with the
        reference's evidence."""
        for operation, lhs, rhs, relation, expected in KNOWN:
            with self.subTest(operation=operation, lhs=lhs, relation=relation):
                args, solution = self.solved(operation, lhs, rhs, relation)
                self.assertEqual(solution.value, expected)
                self.accepts(operation, args, solution.value, solution.evidence)

    def test_special_cases(self):
        """Irrational ends, pieces identically zero, no atom, constant functions, constant and vanishing atoms,
        divisions by constants, repeated atoms, N or A identically zero, a FALSE domain and two spellings of one
        radicand."""
        for operation, lhs, rhs, relation, expected in SPECIAL:
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                args, solution = self.solved(operation, lhs, rhs, relation)
                self.assertEqual(solution.value, expected)
                self.accepts(operation, args, solution.value, solution.evidence)

    def test_every_relation(self):
        problems = (("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3),
                    ("solve_abs", ab(sub(pw(X, 2), 2)), X),
                    ("solve_abs", mul(ab(sub(X, 1)), X), F(1, 4)),
                    ("solve_sqrt", sq(add(X, 2)), X),
                    ("solve_sqrt", sub(sq(pw(X, 2)), X), 0),
                    ("solve_sqrt", sq(mul(-1, pw(X, 2))), 0),
                    ("solve_sqrt", mul(sub(X, 2), sq(sub(3, X))), 1))
        for operation, lhs, rhs in problems:
            for relation in RELATIONS:
                with self.subTest(operation=operation, lhs=lhs, relation=relation):
                    args, solution = self.solved(operation, lhs, rhs, relation)
                    self.accepts(operation, args, solution.value, solution.evidence)

    def test_functions_from_the_parser_and_shared_nodes(self):
        """Functions as the MCP decoder builds them (simplified, a constant side without variables), and one RealExpr
        record shared by both sides and by several parents."""
        x = {"x": XS}
        args = {"lhs": parse_function("abs(x - 1) + abs(x + 2)", symbols=x), "rhs": parse_function("3", symbols={}),
                "variable": XS, "relation": "="}
        solution = solve("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3, "=")
        self.accepts("solve_abs", args, solution.value, solution.evidence)
        args = {"lhs": parse_function("sqrt(x - 1)", symbols=x), "rhs": parse_function("x - 3", symbols=x),
                "variable": XS, "relation": "<"}
        solution = solve("solve_sqrt", sq(sub(X, 1)), sub(X, 3), "<")
        self.accepts("solve_sqrt", args, solution.value, solution.evidence)
        atom = real(ab(sub(X, 1)))
        shared = RealExpr("add", (atom, RealExpr("mul", (atom, atom))))  # |x - 1| + |x - 1|^2 = 2
        args = {"lhs": RealFunction(shared, (XS,)), "rhs": function(2), "variable": XS, "relation": "="}
        solution = solve("solve_abs", add(ab(sub(X, 1)), mul(ab(sub(X, 1)), ab(sub(X, 1)))), 2, "=")
        self.accepts("solve_abs", args, solution.value, solution.evidence)
        same = function(sub(ab(X), 1))
        solution = solve("solve_abs", sub(ab(X), 1), sub(ab(X), 1), "<=")
        self.assertEqual(solution.value, UniversalSet(Reals))
        self.accepts("solve_abs", {"lhs": same, "rhs": same, "variable": XS, "relation": "<="}, solution.value,
                     solution.evidence)

    def test_random_problems_against_the_reference(self):
        """Seeded random problems: the reference's set is accepted, and a set with one member's truth reversed is
        refused with the same evidence."""
        rng = random.Random(9090)

        def small():
            return F(rng.randint(-4, 4), rng.choice((1, 1, 2)))

        def poly(degree):
            return add(*[mul(small(), pw(X, k)) for k in range(degree + 1)])

        checked = 0
        while checked < 36:
            if checked % 2 == 0:
                operation = "solve_abs"
                terms = [mul(small(), ab(poly(rng.choice((1, 1, 2))))) for _ in range(rng.randint(1, 3))]
                lhs, rhs = add(*terms), poly(rng.choice((0, 1, 2)))
            else:
                operation = "solve_sqrt"
                radicand = poly(rng.choice((1, 2)))
                lhs = add(mul(poly(rng.choice((0, 1))), sq(radicand)), poly(rng.choice((0, 1))))
                rhs = poly(rng.choice((0, 1, 2)))
                if not linear(sub(lhs, rhs))[0] and rng.random() < 0.8:
                    continue  # A vanished: keep such problems rare.
            relation = rng.choice(RELATIONS)
            args, solution = self.solved(operation, lhs, rhs, relation)
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs, relation=relation):
                self.accepts(operation, args, solution.value, solution.evidence)
                member = rng.randrange(2 * len(solution.points) + 1)
                self.assertRejects(operation, args, solution.flipped(member), solution.evidence)
            checked += 1

    def test_other_valid_evidence_and_names(self):
        """Evidence the producer would not choose: a rational root left inside the master polynomial instead of a cut
        point, and other samples; ends named by the minimal polynomial, and by a reducible polynomial without a
        rational root that divides a problem polynomial, as v0.7's solve path names the roots of a quartic factor."""
        args, solution = self.solved("solve_abs", ab(sub(pw(X, 2), 2)), X, "<")  # (1, 2); the cuts -2, -1, 1, 2.
        cuts, samples = solution.evidence
        self.accepts("solve_abs", args, solution.value, (points_q(-2, -1, 2), samples))  # 1 stays in the master.
        self.accepts("solve_abs", args, solution.value, ((), samples))
        self.accepts("solve_abs", args, solution.value,
                     (cuts, points_q(-100, F(-3, 2) - F(1, 10**9), F(-11, 10), F(1, 3), F(13, 10), F(199, 100), 7)))
        # |x^4 - 5x^2 + 6| <= 0: sqrt 2 and sqrt 3 as roots of (x^2 - 2)(x^2 - 3), the atom, or of their minimal
        # polynomials.
        quartic = (6, 0, -5, 0, 1)
        args, solution = self.solved("solve_abs", ab(add(pw(X, 4), mul(-5, pw(X, 2)), 6)), 0, "<=")
        named = IntervalSet(tuple(point(RealAlgebraicRoot(quartic, i)) for i in range(4)))
        minimal = IntervalSet((point(SQRT3[0]), point(SQRT2[0]), point(SQRT2[1]), point(SQRT3[1])))
        self.assertEqual(solution.value, named)
        self.accepts("solve_abs", args, named, solution.evidence)
        self.accepts("solve_abs", args, minimal, solution.evidence)
        # |(x - 1)(x^2 - 3)| = 0: the minimal names, with the rational root 1 a cut point or left in the master.
        args, solution = self.solved("solve_abs", ab(mul(sub(X, 1), sub(pw(X, 2), 3))), 0, "=")
        self.assertEqual(solution.value, IntervalSet((point(SQRT3[0]), point(Q(1)), point(SQRT3[1]))))
        self.accepts("solve_abs", args, solution.value, solution.evidence)
        self.accepts("solve_abs", args, solution.value, ((), solution.evidence[1]))

    def test_names_of_irrational_ends_are_verified(self):
        """An irrational end is named by a polynomial of degree two or more that divides one problem polynomial and has
        no rational root, as the minimal polynomial does; every other name is refused, whatever the evidence."""
        cubic = (3, -3, -1, 1)  # (x - 1)(x^2 - 3), the atom of |(x - 1)(x^2 - 3)| = 0, with the rational root 1.
        args, solution = self.solved("solve_abs", ab(mul(sub(X, 1), sub(pw(X, 2), 3))), 0, "=")
        with_cut, without_cut = solution.evidence, ((), solution.evidence[1])
        rows = {
            "the rational end 1 as a root of the cubic": IntervalSet(tuple(
                point(RealAlgebraicRoot(cubic, i)) for i in range(3))),
            "sqrt 3 as a root of the cubic, whose root 1 is rational": IntervalSet((
                point(RealAlgebraicRoot(cubic, 0)), point(Q(1)), point(RealAlgebraicRoot(cubic, 2)))),
        }
        for label, claim in rows.items():
            for evidence in (with_cut, without_cut):
                with self.subTest(label, cuts=evidence[0]):
                    self.assertRejects("solve_abs", args, claim, evidence)
        # |x| (x^2 - 3)^2 = 0: (x^2 - 3)^2 divides a piece, but a RealAlgebraicRoot holds a square-free polynomial.
        args, solution = self.solved("solve_abs", mul(ab(X), pw(sub(pw(X, 2), 3), 2)), 0, "=")
        self.assertEqual(solution.value, IntervalSet((point(SQRT3[0]), point(Q(0)), point(SQRT3[1]))))
        self.accepts("solve_abs", args, solution.value, solution.evidence)
        square = support.forged(RealAlgebraicRoot, integer_coefficients=(9, 0, -6, 0, 1), real_index=0)
        self.assertRejects("solve_abs", args, IntervalSet((point(square), point(Q(0)), point(SQRT3[1]))),
                           solution.evidence)
        # |x^2 - 2| <= 1, whose problem polynomials are x^2 - 2, x^2 - 3 and -(x^2 - 1).
        args, solution = self.solved("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=")

        def ends(lower, upper):
            return IntervalSet((closed(lower, Q(-1)), closed(Q(1), upper)))

        foreign, product = (-3, 0, -2, 0, 1), (6, 0, -5, 0, 1)
        g = 2 ** 60_000  # Within max_integer_bits, far above the Landau-Mignotte bound of every problem polynomial.
        long_name = support.forged(RealAlgebraicRoot, integer_coefficients=(-3 * g - 1, 0, g), real_index=1)
        rows = {
            "sqrt 3 as a root of (x^2 - 3)(x^2 + 1), a factor outside the problem":
                ends(RealAlgebraicRoot(foreign, 0), RealAlgebraicRoot(foreign, 1)),
            "sqrt 3 as a root of (x^2 - 2)(x^2 - 3), which divides only a product of problem polynomials":
                ends(RealAlgebraicRoot(product, 0), RealAlgebraicRoot(product, 3)),
            "the ends swapped": support.forged(IntervalSet, intervals=(
                support.forged(Interval, lower=SQRT3[1], upper=Q(-1), left_closed=True, right_closed=True),
                support.forged(Interval, lower=Q(1), upper=SQRT3[0], left_closed=True, right_closed=True))),
            "sqrt 2 in place of sqrt 3": ends(SQRT2[0], SQRT2[1]),
            "a name with 60,000-bit coefficients": ends(SQRT3[0], long_name),
        }
        for label, claim in rows.items():
            with self.subTest(label):
                start = time.perf_counter()
                self.assertRejects("solve_abs", args, claim, solution.evidence)
                self.assertLess(time.perf_counter() - start, FAST)
        self.accepts("solve_abs", args, ends(SQRT3[0], SQRT3[1]), solution.evidence)

    def test_a_sum_of_absolute_values_fits_the_budget(self):
        """sum |x - a_i| <= c over 20 atoms a_i = k/100: the atoms' and the pieces' roots are cut points, and the
        check needs far less than the default budget."""
        rng = random.Random(4242)
        values = sorted({F(rng.randint(-5000, 5000), 100) for _ in range(20)})
        middle = values[len(values) // 2]
        c = sum(abs(middle - a) for a in values) + F(3217, 100)
        lhs = add(*[ab(sub(X, a)) for a in values])
        args, solution = self.solved("solve_abs", lhs, c, "<=")
        self.assertEqual(len(solution.value.intervals), 1)
        with computation(fresh=True) as budget:
            start = time.perf_counter()
            verdict = support.checker(family).check_operation("solve_abs", args, solution.value, solution.evidence)
            seconds = time.perf_counter() - start
        self.assertIs(verdict, True)
        self.assertLess(budget.work, 200_000)
        self.assertLess(seconds, FAST)
        self.assertRejects("solve_abs", args, solution.flipped(2 * len(solution.points)), solution.evidence)

    # --- Corruptions --------------------------------------------------------------------------------------------

    def bases(self):
        """(operation, args, solution) of claims with rational and irrational ends, one for each row of BASES."""
        out = []
        for operation, lhs, rhs, relation in BASES:
            args, solution = self.solved(operation, lhs, rhs, relation)
            out.append((operation, args, solution))
        return out

    def test_wrong_values_are_refused(self):
        for operation, args, solution in self.bases():
            value = solution.value
            members = 2 * len(solution.points) + 1
            wrong = [solution.flipped(member) for member in range(members)]
            wrong += [EmptySet(Reals), UniversalSet(Reals)]
            for claim in wrong:
                with self.subTest(operation=operation, claim=claim):
                    self.assertNotEqual(claim, value)
                    self.assertRejects(operation, args, claim, solution.evidence)
        args, solution = self.solved("solve_abs", ab(sub(X, 1)), 3, "=")  # {-2, 4}
        for claim in (IntervalSet((point(Q(-2)),)), IntervalSet((point(Q(-2)), point(Q(4)), point(Q(5)))),
                      IntervalSet((closed(Q(-2), Q(4)),)), IntervalSet((point(Q(-2)), point(Q(3))))):
            with self.subTest(claim=claim):
                evidence = (tuple(sorted(set(solution.cuts) | {e for i in claim.intervals for e in (i.lower,)})),
                            solution.evidence[1])
                self.assertRejects("solve_abs", args, claim, solution.evidence)
                self.assertRejects("solve_abs", args, claim, evidence)

    def test_non_canonical_values_are_refused(self):
        """Mathematically correct sets in a form that is not the canonical one."""
        args, solution = self.solved("solve_abs", add(ab(sub(X, 1)), ab(add(X, 2))), 3, "=")  # [-2, 1]
        a, b, zero = Q(-2), Q(1), Q(0)
        rows = {
            "two touching intervals": support.forged(IntervalSet, intervals=(closed(a, zero),
                                                                             Interval(zero, b, False, True))),
            "unsorted pieces": support.forged(IntervalSet, intervals=(closed(zero, b), closed(a, zero))),
            "the interval twice": support.forged(IntervalSet, intervals=(closed(a, b), closed(a, b))),
            "an empty interval besides": support.forged(IntervalSet, intervals=(closed(a, b),
                                                                                Interval(Q(3), Q(3), False, False))),
            "an int end": support.forged(IntervalSet, intervals=(support.forged(Interval, lower=-2, upper=b,
                                                                                left_closed=True, right_closed=True),)),
            "an end outside lowest terms": IntervalSet((closed(trusted_rational(-4, 2), b),)),
            "a rational end as a RealAlgebraicRoot": IntervalSet((closed(RealAlgebraicRoot((2, 1), 0), b),)),
            "a list of intervals": support.forged(IntervalSet, intervals=[closed(a, b)]),
        }
        for label, claim in rows.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", args, claim, solution.evidence)
        args, solution = self.solved("solve_abs", ab(X), -1, "=")
        for label, claim in {"the empty IntervalSet": IntervalSet(()), "an empty FiniteSet": FiniteSet((), Reals),
                             "EmptySet over the integers": EmptySet(Integers),
                             "EmptySet with the domain 'reals'": support.forged(EmptySet, domain="reals"),
                             "EmptySet without its domain": support.forged(EmptySet)}.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", args, claim, solution.evidence)
        args, solution = self.solved("solve_abs", ab(X), 0, ">=")
        for label, claim in {"the whole line as an IntervalSet": IntervalSet((Interval(None, None),)),
                             "UniversalSet over the rationals": UniversalSet(Rationals),
                             "two halves": support.forged(IntervalSet, intervals=(Interval(None, Q(0), False, True),
                                                                                  Interval(Q(0), None)))}.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", args, claim, solution.evidence)
        args, solution = self.solved("solve_abs", ab(sub(X, 1)), 3, "=")
        self.assertRejects("solve_abs", args, FiniteSet((Q(-2), Q(4)), Reals), solution.evidence)
        # sqrt 3 named by (x - 1)(x^2 - 3), whose root 1 is a cut point.
        args, solution = self.solved("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=")
        cubic = (3, -3, -1, 1)
        claim = IntervalSet((closed(RealAlgebraicRoot(cubic, 0), Q(-1)), closed(Q(1), RealAlgebraicRoot(cubic, 2))))
        self.assertRejects("solve_abs", args, claim, solution.evidence)

    def test_impostors_and_forged_records_are_refused(self):
        def disguised(end):
            return support.impostor(end) if type(end) is RealAlgebraicRoot else end

        for operation, args, solution in self.bases():
            value = solution.value
            rows = {"an impostor IntervalSet": support.impostor(value),
                    "an impostor Interval inside": support.forged(IntervalSet, intervals=tuple(
                        support.impostor(interval) for interval in value.intervals)),
                    "a forged IntervalSet without intervals": support.forged(IntervalSet),
                    "a forged Interval without flags": support.forged(IntervalSet, intervals=tuple(
                        support.forged(Interval, lower=i.lower, upper=i.upper) for i in value.intervals))}
            if any(type(end) is RealAlgebraicRoot for i in value.intervals for end in (i.lower, i.upper)):
                rows["an impostor RealAlgebraicRoot end"] = support.forged(IntervalSet, intervals=tuple(
                    support.forged(Interval, lower=disguised(i.lower), upper=disguised(i.upper),
                                   left_closed=i.left_closed, right_closed=i.right_closed) for i in value.intervals))
            if any(i.lower != i.upper for i in value.intervals):
                rows["a reversed forged interval"] = support.forged(IntervalSet, intervals=tuple(
                    support.forged(Interval, lower=i.upper, upper=i.lower, left_closed=i.left_closed,
                                   right_closed=i.right_closed) for i in value.intervals))
            for label, claim in rows.items():
                with self.subTest(operation=operation, label=label):
                    self.assertRejects(operation, args, claim, solution.evidence)
        args, solution = self.solved("solve_abs", ab(X), -1, "=")
        self.assertRejects("solve_abs", args, support.impostor(EmptySet(Reals)), solution.evidence)
        args, solution = self.solved("solve_abs", ab(X), 0, ">=")
        self.assertRejects("solve_abs", args, support.impostor(UniversalSet(Reals)), solution.evidence)

    def test_malformed_ends_and_records_are_refused(self):
        """A bool, a float, a str or None where an int or a Rational belongs, and a list for a tuple, in the records of
        the claim and of the arguments."""
        args, solution = self.solved("solve_abs", ab(sub(pw(X, 2), 2)), 1, "<=")  # [-sqrt 3, -1] | [1, sqrt 3]

        def upper_end(**fields):
            root = support.forged(RealAlgebraicRoot, **dict({"integer_coefficients": (-3, 0, 1), "real_index": 1},
                                                            **fields))
            return support.forged(IntervalSet, intervals=(closed(SQRT3[0], Q(-1)), support.forged(
                Interval, lower=Q(1), upper=root, left_closed=True, right_closed=True)))

        self.accepts("solve_abs", args, upper_end(), solution.evidence)
        rows = {
            "a bool coefficient": upper_end(integer_coefficients=(-3, 0, True)),
            "a float coefficient": upper_end(integer_coefficients=(-3.0, 0, 1)),
            "a str coefficient": upper_end(integer_coefficients=("-3", 0, 1)),
            "a None coefficient": upper_end(integer_coefficients=(-3, None, 1)),
            "a list of coefficients": upper_end(integer_coefficients=[-3, 0, 1]),
            "a negative leading coefficient": upper_end(integer_coefficients=(3, 0, -1)),
            "a polynomial that is not primitive": upper_end(integer_coefficients=(-6, 0, 2)),
            "a trailing zero": upper_end(integer_coefficients=(-3, 0, 1, 0)),
            "a bool index": upper_end(real_index=True),
            "a str index": upper_end(real_index="1"),
            "a float index": upper_end(real_index=1.0),
            "a negative index": upper_end(real_index=-1),
            "an index past the real roots": upper_end(real_index=2),
            "no index": support.forged(IntervalSet, intervals=(closed(SQRT3[0], Q(-1)), support.forged(
                Interval, lower=Q(1), upper=support.forged(RealAlgebraicRoot, integer_coefficients=(-3, 0, 1)),
                left_closed=True, right_closed=True))),
            "int flags": support.forged(IntervalSet, intervals=(closed(SQRT3[0], Q(-1)), support.forged(
                Interval, lower=Q(1), upper=SQRT3[1], left_closed=1, right_closed=1))),
            "a float end": support.forged(IntervalSet, intervals=(closed(SQRT3[0], Q(-1)), support.forged(
                Interval, lower=1.0, upper=SQRT3[1], left_closed=True, right_closed=True))),
            "a str end": support.forged(IntervalSet, intervals=(closed(SQRT3[0], Q(-1)), support.forged(
                Interval, lower="1", upper=SQRT3[1], left_closed=True, right_closed=True))),
        }
        for label, claim in rows.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", args, claim, solution.evidence)
        lhs = args["lhs"]
        rows = {
            "variables in a list": support.forged(RealFunction, expression=lhs.expression, variables=[XS],
                                                  domain=lhs.domain),
            "a str for the variables": support.forged(RealFunction, expression=lhs.expression, variables="x",
                                                      domain=lhs.domain),
            "a domain of operands in a list": support.forged(RealFunction, expression=lhs.expression, variables=(XS,),
                                                             domain=support.forged(DomainPredicate, kind="true",
                                                                                   operands=[])),
            "a domain without its kind": support.forged(RealFunction, expression=lhs.expression, variables=(XS,),
                                                        domain=support.forged(DomainPredicate, operands=())),
            "an impostor domain": support.forged(RealFunction, expression=lhs.expression, variables=(XS,),
                                                 domain=support.impostor(DomainPredicate("true"))),
        }
        for label, function in rows.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", dict(args, lhs=function), solution.value, solution.evidence)

    def test_wrong_types_in_the_evidence_are_refused(self):
        for operation, args, solution in self.bases():
            value, (cuts, samples) = solution.value, solution.evidence
            rows = {
                "no evidence": (), "None": None, "one part": (cuts,), "three parts": (cuts, samples, ()),
                "swapped": (samples, cuts), "a list": [cuts, samples], "a list of cut points": (list(cuts), samples),
                "a list of samples": (cuts, list(samples)), "a nested tuple": ((cuts,), samples),
                "an int sample": (cuts, (int(samples[0].numerator),) + samples[1:]) if samples[0].denominator == 1
                else (cuts, (0,) + samples[1:]),
                "a bool sample": (cuts, (True,) + samples[1:]), "a float sample": (cuts, (-5.0,) + samples[1:]),
                "a str sample": (cuts, ("-5",) + samples[1:]), "a None sample": (cuts, (None,) + samples[1:]),
                "a sample outside lowest terms": (cuts, (trusted_rational(-10, 2),) + samples[1:]),
                "a sample with a zero denominator": (cuts, (trusted_rational(-1, 0),) + samples[1:]),
                "a sample with a negative denominator": (cuts, (trusted_rational(5, -1),) + samples[1:]),
                "a forged sample without fields": (cuts, (support.forged(Q),) + samples[1:]),
                "an impostor sample": (cuts, (support.impostor(samples[0]),) + samples[1:]),
            }
            if cuts:
                first = cuts[0].numerator / cuts[0].denominator
                rows.update({"a bool cut": ((True,) + cuts[1:], samples), "a float cut": ((first,) + cuts[1:], samples),
                             "a str cut": ((str(cuts[0]),) + cuts[1:], samples),
                             "an int cut": ((cuts[0].numerator,) + cuts[1:], samples)})
            for label, evidence in rows.items():
                with self.subTest(operation=operation, label=label):
                    self.assertRejects(operation, args, value, evidence)

    def test_consistent_but_wrong_evidence_is_refused(self):
        """Evidence that is well formed, or even consistent with itself, and still certifies nothing."""
        for (operation, args, solution), (_, lhs, rhs, _) in zip(self.bases(), BASES):
            value, (cuts, samples) = solution.value, solution.evidence
            rows = {
                "the cut of some problem polynomials only": partial_evidence(operation, lhs, rhs),
                "a sample missing": (cuts, samples[1:]), "a sample too many": (cuts, samples + (samples[-1] + 1,)),
                "two samples in one cell": (cuts, (samples[0] - 1,) + samples[:-1]),
                "a sample at a point": (cuts, samples[:1] + ((cuts or (samples[1],))[0],) + samples[2:]),
                "samples decreasing": (cuts, tuple(reversed(samples))),
                "a sample repeated": (cuts, samples[:1] + samples[:-1]),
                "a cut point that is no root": (tuple(sorted(cuts + (Q(1000),))), samples + (Q(1001),)),
                "the evidence of another problem": solve("solve_abs", ab(sub(X, 7)), 1, "=").evidence,
            }
            if cuts:
                rows["a cut point missing and its sample with it"] = (cuts[1:], samples[1:])
                rows["cut points unsorted"] = (tuple(reversed(cuts)), samples) if len(cuts) > 1 else \
                    (cuts + cuts, samples)
                rows["a cut point moved"] = (points_q(F(cuts[0].numerator, cuts[0].denominator) + F(1, 3)) + cuts[1:],
                                             samples)
            for label, evidence in rows.items():
                with self.subTest(operation=operation, label=label):
                    self.assertRejects(operation, args, value, evidence)

    def test_wrong_operations_and_arguments_are_refused(self):
        abs_args, abs_solution = self.solved("solve_abs", ab(sub(X, 1)), 3, "=")
        sqrt_args, sqrt_solution = self.solved("solve_sqrt", sq(add(X, 2)), X, "=")
        for operation in ("solve_sqrt", "solve", "", None, 1):
            with self.subTest(operation=operation):
                self.assertRejects(operation, abs_args, abs_solution.value, abs_solution.evidence)
        self.assertRejects("solve_abs", sqrt_args, sqrt_solution.value, sqrt_solution.evidence)

        def changed(base, **fields):
            return dict(base, **fields)

        rows = {
            "a missing argument": {k: v for k, v in abs_args.items() if k != "relation"},
            "an extra argument": changed(abs_args, expression=abs_args["lhs"]),
            "a list for the arguments": list(abs_args.items()),
            "another relation": changed(abs_args, relation="!="),
            "a relation of another type": changed(abs_args, relation=0),
            "no Symbol": changed(abs_args, variable="x"),
            "an integer variable": changed(abs_args, variable=mf.symbol("x", Integers)),
            "another variable": changed(abs_args, variable=YS),
            "a function of another variable": changed(abs_args, lhs=RealFunction(real(ab(sub(X, 1)), YS), (YS,))),
            "a function of two variables": changed(abs_args, lhs=RealFunction(real(ab(sub(X, 1))), (XS, YS))),
            "a Polynomial side": changed(abs_args, lhs=mf.Polynomial((Q(-1), Q(1)), XS)),
            "an int side": changed(abs_args, rhs=3),
            "an impostor function": changed(abs_args, lhs=support.impostor(abs_args["lhs"])),
            "a forged function without expression": changed(abs_args, lhs=support.forged(
                RealFunction, variables=(XS,), domain=DomainPredicate("true"))),
            "a sine": changed(abs_args, lhs=RealFunction(RealExpr("sin", (real(X),)), (XS,))),
            "pi": changed(abs_args, rhs=RealFunction(RealExpr("pi"), (XS,))),
            "an irrational constant": changed(abs_args, rhs=RealFunction(RealExpr("scalar", (), SQRT2[1]), (XS,))),
            "a nested abs": changed(abs_args, lhs=function(ab(sub(ab(X), 1)))),
            "a negative power": changed(abs_args, lhs=RealFunction(RealExpr("pow", (real(ab(X)),), -1), (XS,))),
            "a division by the variable": changed(abs_args, lhs=function(dv(ab(X), X))),
            "a division by zero": changed(abs_args, lhs=function(dv(ab(X), sub(X, X)))),
            "a square root in solve_abs": changed(abs_args, lhs=function(add(ab(X), sq(X)))),
            "a domain predicate": changed(abs_args, lhs=RealFunction(
                real(ab(sub(X, 1))), (XS,), DomainPredicate("positive", (real(X),)))),
        }
        for label, args in rows.items():
            with self.subTest(label):
                self.assertRejects("solve_abs", args, abs_solution.value, abs_solution.evidence)
        rows = {
            "no square root": changed(sqrt_args, lhs=function(add(X, 2))),
            "two radicands": changed(sqrt_args, lhs=function(add(sq(add(X, 2)), sq(X)))),
            "a product of square roots": changed(sqrt_args, lhs=function(mul(sq(add(X, 2)), sq(add(X, 2))))),
            "a square of a square root": changed(sqrt_args, lhs=function(pw(sq(add(X, 2)), 2))),
            "a nested square root": changed(sqrt_args, lhs=function(sq(sq(add(X, 2))))),
            "an abs in solve_sqrt": changed(sqrt_args, rhs=function(ab(X))),
            "a domain of another radicand": changed(sqrt_args, lhs=RealFunction(
                real(sq(add(X, 2))), (XS,), DomainPredicate("nonnegative", (real(add(X, 3)),)))),
            "a positive domain": changed(sqrt_args, lhs=RealFunction(
                real(sq(add(X, 2))), (XS,), DomainPredicate("positive", (real(add(X, 2)),)))),
            "a FALSE domain with a non-constant radicand": changed(sqrt_args, lhs=support.forged(
                RealFunction, expression=real(sq(add(X, 2))), variables=(XS,), domain=DomainPredicate("false"))),
        }
        for label, args in rows.items():
            with self.subTest(label):
                self.assertRejects("solve_sqrt", args, sqrt_solution.value, sqrt_solution.evidence)

    def test_forged_expressions_are_refused(self):
        """RealExpr records made with object.__new__: a cycle, a missing field, a wrong payload."""
        args, solution = self.solved("solve_abs", ab(sub(X, 1)), 3, "=")
        cycle = support.forged(RealExpr, op="add", value=None)
        object.__setattr__(cycle, "args", (cycle, real(X)))
        deep = support.forged(RealExpr, op="abs", value=None)  # A cycle below an abs, two levels down.
        object.__setattr__(deep, "args", (RealExpr("add", (real(X), real(lift(1)))),))
        object.__setattr__(deep.args[0], "args", (deep, real(X)))
        rows = {
            "a cycle": cycle,
            "a cycle further down": deep,
            "an add with one operand": support.forged(RealExpr, op="add", args=(real(X),), value=None),
            "a symbol without its Symbol": support.forged(RealExpr, op="symbol", args=()),
            "a scalar holding a float": support.forged(RealExpr, op="scalar", args=(), value=1.5),
            "a scalar holding an int": support.forged(RealExpr, op="scalar", args=(), value=3),
            "a power holding a bool": support.forged(RealExpr, op="pow", args=(real(X),), value=True),
            "a list of operands": support.forged(RealExpr, op="add", args=[real(X), real(X)], value=None),
            "an impostor node": support.impostor(real(X)),
        }
        for label, expression in rows.items():
            lhs = support.forged(RealFunction, expression=expression, variables=(XS,), domain=DomainPredicate("true"))
            with self.subTest(label):
                self.assertRejects("solve_abs", dict(args, lhs=lhs), solution.value, solution.evidence)

    def test_oversized_forgeries_fail_fast(self):
        """Oversized forged values and evidence are refused within FAST seconds, before any work on them."""
        for operation, args, solution in self.bases():
            value, (cuts, samples) = solution.value, solution.evidence
            many = tuple(Q(k) for k in range(200_000))
            long_rational = Q(GIANT + 1, 3)
            interval = value.intervals[0]
            rows = {
                "200,000 samples": (value, (cuts, many)),
                "200,000 cut points": (value, (many, samples)),
                "a sample longer than max_integer_bits": (value, (cuts, samples[:-1] + (long_rational,))),
                "a cut point longer than max_integer_bits": (value, ((-long_rational,) + cuts, samples)),
                "200,000 intervals": (support.forged(IntervalSet, intervals=(interval,) * 200_000), (cuts, samples)),
                "an end polynomial of 200,000 coefficients": (support.forged(IntervalSet, intervals=(
                    support.forged(Interval, lower=support.forged(RealAlgebraicRoot,
                                                                  integer_coefficients=(1,) * 200_000, real_index=0),
                                   upper=None, left_closed=True, right_closed=False),)), (cuts, samples)),
                "an end polynomial with a long coefficient": (support.forged(IntervalSet, intervals=(
                    support.forged(Interval, lower=support.forged(RealAlgebraicRoot,
                                                                  integer_coefficients=(-GIANT, 0, 1), real_index=1),
                                   upper=None, left_closed=True, right_closed=False),)), (cuts, samples)),
                "an end longer than max_integer_bits": (support.forged(IntervalSet, intervals=(
                    support.forged(Interval, lower=long_rational, upper=None, left_closed=True,
                                   right_closed=False),)), (cuts, samples)),
            }
            for label, (claim, evidence) in rows.items():
                with self.subTest(operation=operation, label=label):
                    start = time.perf_counter()
                    self.assertRejects(operation, args, claim, evidence)
                    self.assertLess(time.perf_counter() - start, FAST)

    def test_budget(self):
        """An exhausted budget is no verdict: it raises ResourceLimitError. So does an argument longer than
        max_integer_bits, which run() and the verifier refuse before any checker runs."""
        for operation, args, solution in self.bases():
            with self.subTest(operation=operation), self.assertRaises(ResourceLimitError):
                support.verdict(family, operation, args, solution.value, solution.evidence,
                                limits=ComputationLimits(max_work=40))
        args, solution = self.solved("solve_abs", ab(sub(X, 1)), 3, "=")
        huge = RealFunction(RealExpr("scalar", (), Q(GIANT)), (XS,))
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "solve_abs", dict(args, rhs=huge), solution.value, solution.evidence)
        with self.assertRaises(ResourceLimitError):
            support.verdict(family, "solve_abs", dict(args, lhs=RealFunction(
                RealExpr("pow", (real(ab(X)),), GIANT), (XS,))), solution.value, solution.evidence)
