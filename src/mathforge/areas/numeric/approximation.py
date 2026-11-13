"""Best rational approximations of exact real numbers (mathforge.numeric): rational_approximation.

rational_approximation(value, max_denominator) takes an exact real number, value (MCP kind "scalar", annotation
"Expression | int"), the values that digits accepts: an int, a Rational, a RealAlgebraicRoot, or a closed legacy
expression built from Rationals and square roots of Rationals (a tree of the mathforge.model nodes Add, Mul, Pow with
an exact int exponent >= 0 and Sqrt with a Rational radicand >= 0). max_denominator is an exact int N with
1 <= N <= MAX_DENOMINATOR = 2**64 (kind "int"); any other N, and any other value (a bool, a float, a str, a Symbol, an
expression with a RealAlgebraicRoot inside it, a record made with object.__new__ outside its canonical form), is
invalid input.

The value is the Rational r (always a mathforge.model.Rational, never an int: Rational(0) for 0) with denominator
<= N that minimizes |value - r|; on a tie, the value exactly midway between two candidates, the one with the smaller
denominator, and of two with equal denominators the smaller one. The family has no record classes.

The evidence, in the layout that the checker, mathforge.certify.numeric.approximation, verifies:

- () when the value is a rational with denominator <= N: then r equals the value;
- (l, r_) otherwise, two Rationals (never ints): the neighbours of the value in the Farey sequence of order N, so
  r_.numerator * l.denominator - l.numerator * r_.denominator == 1, both denominators <= N, their sum > N, and
  l < value < r_; the value r is the one of l and r_ that is closer to the value (the tie rule above on a tie).

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/numeric/test_approximation.py is the definition of done.

How produce() computes (docs/v09/cards/m7-extended.md). Nothing here is a proof: the checker verifies the Farey
certificate and decides each comparison with methods of its own.

- Every kind of value is reduced to an exact comparison sign(x - p/q) of the number x with rationals, and to floor(x).
  - An int or a Rational a/b: the sign of a*q - p*b, and a // b. A denominator b <= N answers r = a/b with ().
  - A RealAlgebraicRoot (S, i), validated as digits validates it (mathforge.areas.numeric.digits._root): digits'
    isolating interval (low, high] of alpha_i, a point or a linear S for a rational root (then as above), and
    digits._root_floor at scale 2**k, k = 2 * bits(N) + 2, for f = floor(alpha_i * 2**k) and whether alpha_i is
    f/2**k (a dyadic rational root, then as above); floor(alpha_i) = f >> k. Otherwise alpha_i is the only root of
    the square-free S in the open bracket max(low, f/2**k) < alpha_i < min(high, (f + 1)/2**k), so for t in the
    bracket the exact sign of S at t, times the sign of S at high, is -1, 0 or 1 as t lies below, at or above
    alpha_i; each comparison narrows the bracket to t. The bracket is narrower than 1/(4 N**2), so at most one
    fraction of denominator <= N lies inside it and nearly every comparison of the walk is decided by its ends.
  - A closed expression, parsed as digits parses it: the outward-rounded enclosures of the checker
    (digits._enclosure, the very rules of mathforge.certify.numeric.digits) from b = 4 * bits(N) + 64 bits, doubling
    up to max_integer_bits, each computed once. A rational t that an enclosure does not separate from x makes
    produce() decide, once, whether x is rational, in the multiquadratic algebra over the natural coprime base of the
    radicands (digits._exact_value; the products of the square roots of pairwise coprime non-squares are linearly
    independent over Q). A rational x is compared exactly from then on; an irrational x is separated from t by a
    finer enclosure. floor(x) comes from the first enclosure narrower than 1 and two exact comparisons.
- The continued fraction convergents and semiconvergents of x, as a Stern-Brocot walk with batched steps: from the
  Farey neighbours l = floor(x)/1 < x < u = (floor(x) + 1)/1, the steps l + t*u toward u stay below x for t up to
  the next partial quotient, and the steps u + t*l toward l stay above it for t up to the one after; the largest t
  with a denominator <= N is found by doubling and bisection, a few comparisons per partial quotient. Each step
  keeps u.numerator * l.denominator - l.numerator * u.denominator == 1. A step that lands on x shows that x is a
  rational with denominator <= N: r = x with (). The walk ends when no step fits, that is when
  l.denominator + u.denominator > N: l and u are the Farey neighbours of x of order N, and the exact comparison of x
  with their midpoint (l + u)/2 picks r, by the tie rule when x is the midpoint.
- An expression is compared as the checker compares it, so that what produce() claims the checker can verify: for
  each of l, u and the midpoint (or r for ()), produce() replays the checker's sequence of enclosures and charges
  their work again (the checker recomputes each one, produce() keeps them), and refuses with ResourceLimitError a
  comparison that the first enclosure leaves open when the radicands hold more than digits.MAX_RADICALS coprime
  non-squares, where the checker's formal value raises ResourceLimitError.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges a unit for the call, refuses an integer longer than max_integer_bits,
including a numerator of r, l or u, with ResourceLimitError (the checker answers False to such a claim), and charges
every product of the walk before it runs, on the scale of trusted.upoly; digits' helpers charge their own work. The
producer's work is meant to bound the checker's, so that a claim produced within a budget verifies within the same
one (a test compares them): besides the replayed enclosures of an expression, a root is charged the checker's Sturm
work (_sturm_allowance), and every answer the checker's fixed work (_CHECK_UNITS).
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from typing import Any

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Add, Mul, Pow, Rational, Sqrt
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of the numeric area (mathforge.areas.numeric.digits), extended by the objects of this family.
_SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets, "
          "decimal digits and best rational approximations")
# The largest max_denominator (the card); the checker keeps its own copy, which a test compares.
MAX_DENOMINATOR = 2 ** 64

SPECS = (
    OperationSpec("rational_approximation", "numeric",
                  (Argument("value", "Expression | int", None, "scalar"),
                   Argument("max_denominator", "int", None, "int")),
                  "Rational",
                  "Find the closest rational number with a bounded denominator to an exact real number.",
                  _SCOPE, version=6),
)

# The values are mathforge.model.Rational: the family defines no record class.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.numeric.approximation"

_OPERATION = "rational_approximation"
_ARGUMENTS = frozenset(("value", "max_denominator"))
_SCOPE_ERROR = ("rational_approximation requires an exact real number: an int, a Rational, a RealAlgebraicRoot or a "
                "closed expression of Rationals and square roots of Rationals with sums, products and powers >= 0")
_GUARD_BITS = 64  # Bits of the first enclosure beyond 4 * bits(N): the checker's first precision.
# Units charged on every answer for the checker's fixed work (reading the claim, its gcds and the midpoint), which
# the shortest walks, such as the value itself for a small denominator, do not reach.
_CHECK_UNITS = 16

Comparator = Callable[[int, int], int]  # sign(x - p/q) for q >= 1.
Answer = tuple[Rational, tuple]


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(Rational, evidence) of rational_approximation(value, max_denominator); the module states the algorithm and
    the evidence. InvalidInput for N outside 1 to MAX_DENOMINATOR and for a value outside the scope,
    ResourceLimitError past the budget."""
    if type(operation) is not str or operation != _OPERATION:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the approximation family")
    if type(args) is not dict or args.keys() != _ARGUMENTS:
        raise InvalidInput("rational_approximation takes exactly the arguments value and max_denominator")
    value, limit = args["value"], args["max_denominator"]
    if type(limit) is not int or not 1 <= limit <= MAX_DENOMINATOR:  # A bool, a float or a str is no exact int.
        raise InvalidInput(f"max_denominator of rational_approximation must be an exact int from 1 to 2**64 "
                           f"({MAX_DENOMINATOR:,})")
    from ...algebraic import RealAlgebraicRoot
    from . import digits

    with computation() as budget:
        budget.tick()  # The call itself.
        kind = type(value)
        if kind is int:
            budget.check_int(value)
            answer = _rational_answer(budget, value, 1, limit)[0]
        elif kind is Rational:
            answer = _rational_answer(budget, *digits._fraction(budget, value), limit)[0]
        elif kind is RealAlgebraicRoot:
            answer = _root_answer(budget, digits, value, limit)
        elif kind is Add or kind is Mul or kind is Pow or kind is Sqrt:
            answer = _expression_answer(budget, digits, digits._parse(budget, value), limit)
        else:
            raise InvalidInput(_SCOPE_ERROR)
        budget.tick(_CHECK_UNITS)
        return answer


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic

def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products or divisions of integers of the given bit lengths: a unit each and
    what their lengths cost beyond it (trusted.upoly.charge_product)."""
    from ...trusted.upoly import charge_product

    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _cross_sign(budget: Budget, a: int, b: int, p: int, q: int) -> int:
    """sign(a/b - p/q) for b, q >= 1, by the sign of a*q - p*b."""
    _charge(budget, max(a.bit_length(), p.bit_length()), max(b.bit_length(), q.bit_length()), 2)
    difference = a * q - p * b
    return (difference > 0) - (difference < 0)


def _rational(budget: Budget, numerator: int, denominator: int) -> Rational:
    """The Rational numerator/denominator (denominator >= 1, lowest terms); ResourceLimitError for a numerator past
    max_integer_bits, which the checker refuses in a claim."""
    budget.check_int(numerator)
    budget.check_int(denominator)
    return Rational(numerator, denominator)


# ---------------------------------------------------------------------------------------------------------------
# The walk (module docstring)

def _farthest(budget: Budget, compare: Comparator, start: tuple[int, int], toward: tuple[int, int], top: int,
              wanted: int) -> tuple[int, bool]:
    """(t, hit): the largest t in 0..top with start + t*toward on the wanted side of x (sign(x - step) == wanted);
    the steps move monotonically toward `toward`, so they are on that side for every t up to it. hit is True when
    the step t lands on x itself, which ends the search. Doubling, then bisection."""
    (sp, sq), (tp, tq) = start, toward

    def side(t: int) -> int:
        _charge(budget, t.bit_length(), max(tp.bit_length(), tq.bit_length()), 2)
        return compare(sp + t * tp, sq + t * tq)

    good, bad = 0, top + 1
    step = 1
    while step <= top:
        budget.tick()
        found = side(step)
        if found == 0:
            return step, True
        if found != wanted:
            bad = step
            break
        good = step
        step <<= 1
    while bad - good > 1:
        budget.tick()
        middle = (good + bad) >> 1
        found = side(middle)
        if found == 0:
            return middle, True
        if found == wanted:
            good = middle
        else:
            bad = middle
    return good, False


def _walk(budget: Budget, compare: Comparator, floor: int, limit: int) -> tuple[tuple[int, int], tuple]:
    """((p, q) of r, the Farey pair ((a, b), (c, d)) or ()) for x with floor < x < floor + 1, by the Stern-Brocot
    walk of the module docstring. The returned fractions are in lowest terms: each pair keeps c*b - a*d == 1."""
    a, b, c, d = floor, 1, floor + 1, 1  # l = a/b < x < u = c/d.
    while True:
        budget.tick()
        moved = False
        top = (limit - b) // d
        if top:
            t, hit = _farthest(budget, compare, (a, b), (c, d), top, 1)
            if hit:
                return (a + t * c, b + t * d), ()
            if t:
                a, b, moved = a + t * c, b + t * d, True
        top = (limit - d) // b
        if top:
            t, hit = _farthest(budget, compare, (c, d), (a, b), top, -1)
            if hit:
                return (c + t * a, d + t * b), ()
            if t:
                c, d, moved = c + t * a, d + t * b, True
        if not moved:  # b + d > limit: no step fits, l and u are the Farey neighbours of order limit.
            break
    _charge(budget, max(a.bit_length(), c.bit_length()), max(b.bit_length(), d.bit_length()), 3)
    numerator, denominator = a * d + c * b, 2 * b * d
    common = gcd(numerator, denominator)
    side = compare(numerator // common, denominator // common)
    if side < 0:
        chosen = (a, b)
    elif side > 0:
        chosen = (c, d)
    else:  # A tie: the smaller denominator, and of equal denominators the smaller rational, l.
        chosen = (c, d) if d < b else (a, b)
    return chosen, ((a, b), (c, d))


def _answer(budget: Budget, compare: Comparator, floor: int, limit: int) -> tuple[Answer, tuple]:
    """(the Rational answer and its evidence, the rationals that the checker compares with x) for x with
    floor < x < floor + 1."""
    (p, q), pair = _walk(budget, compare, floor, limit)
    value = _rational(budget, p, q)
    if not pair:
        return (value, ()), ((p, q),)
    (a, b), (c, d) = pair
    evidence = (_rational(budget, a, b), _rational(budget, c, d))
    numerator, denominator = a * d + c * b, 2 * b * d
    common = gcd(numerator, denominator)
    return (value, evidence), ((a, b), (c, d), (numerator // common, denominator // common))


def _rational_answer(budget: Budget, numerator: int, denominator: int, limit: int) -> tuple[Answer, tuple]:
    """(the answer, the checker's points) for the rational numerator/denominator (lowest terms, denominator >= 1)."""
    if denominator <= limit:
        return (_rational(budget, numerator, denominator), ()), ((numerator, denominator),)

    def compare(p: int, q: int) -> int:
        return _cross_sign(budget, numerator, denominator, p, q)

    return _answer(budget, compare, numerator // denominator, limit)


# ---------------------------------------------------------------------------------------------------------------
# Real algebraic roots

def _root_answer(budget: Budget, digits: Any, value: object, limit: int) -> Answer:
    """The answer for a RealAlgebraicRoot (module docstring), with the work of the checker's Sturm counts charged
    as well (_sturm_allowance)."""
    polynomial, answer, points = _root_search(budget, digits, value, limit)
    _sturm_allowance(budget, polynomial, points)
    return answer


def _root_search(budget: Budget, digits: Any, value: object,
                 limit: int) -> tuple[tuple[int, ...], Answer, tuple]:
    """(S, the answer, the checker's points) for a RealAlgebraicRoot (S, i)."""
    polynomial, (low, high) = digits._root(budget, value)
    if low == high:  # A rational root, met by a probe of the isolation.
        return (polynomial, *_rational_answer(budget, *low, limit))
    if len(polynomial) == 2:  # c0 + c1 x, primitive with c1 > 0: the root -c0/c1 in lowest terms.
        return (polynomial, *_rational_answer(budget, -polynomial[0], polynomial[1], limit))
    # alpha to within 2**-shift < 1/(4 N**2) first, by digits' Newton refinement: two fractions of denominators <= N
    # differ by at least 1/N**2, so at most one step of the walk, and the midpoint, fall strictly inside the bracket,
    # and every other comparison is decided by the bracket's ends alone. Comparing each step by the sign of S
    # instead cost a Horner evaluation per step on numerators as long as alpha: a root of 32,500 bits at N = 2**64
    # took 3.4 million units in about 130 evaluations, where the checker verifies its answer in 155,000.
    shift = 2 * limit.bit_length() + 2
    scaled, exact = digits._root_floor(budget, polynomial, low, high, 1 << shift)
    if exact:  # alpha = scaled / 2**shift, a dyadic rational root.
        _charge(budget, scaled.bit_length(), shift + 1, 3)
        common = gcd(scaled, 1 << shift)
        return (polynomial, *_rational_answer(budget, scaled // common, (1 << shift) // common, limit))
    floor = scaled >> shift  # scaled/2**shift < alpha < (scaled + 1)/2**shift, so both have this floor.
    right = digits._rational_sign(budget, polynomial, *high)  # The sign of S on (alpha, high]; high is no root.
    bracket = [low, high]  # alpha lies strictly between them, the only root of S there.
    if _cross_sign(budget, scaled, 1 << shift, *low) > 0:
        bracket[0] = (scaled, 1 << shift)
    if _cross_sign(budget, scaled + 1, 1 << shift, *high) < 0:
        bracket[1] = (scaled + 1, 1 << shift)

    def compare(p: int, q: int) -> int:
        (lp, lq), (hp, hq) = bracket
        if _cross_sign(budget, p, q, lp, lq) <= 0:
            return 1  # t <= low < alpha.
        if _cross_sign(budget, p, q, hp, hq) >= 0:
            return -1  # alpha < high <= t.
        side = right * digits._rational_sign(budget, polynomial, p, q)
        if side < 0:
            bracket[0] = (p, q)
            return 1
        if side > 0:
            bracket[1] = (p, q)
            return -1
        return 0  # S(t) == 0 inside the bracket: t is alpha, a rational root.

    return (polynomial, *_answer(budget, compare, floor, limit))


def _sturm_allowance(budget: Budget, polynomial: tuple[int, ...], points: tuple[tuple[int, int], ...]) -> None:
    """Charge the work of the checker's reading of the root and of its comparisons with the points. The checker
    tests that S is square-free (a gcd of S and S') and counts the roots of S on its own Sturm chain, whose members
    have the degrees of kernels.upoly's chain (all three are remainder sequences of S and S'): two sequences of
    pseudo-remainders, each charged as trusted.upoly.pseudo_remainder charges it on coefficients of twice the length
    of kernels.upoly's primitive members (a margin for the subresultant scalars), and the count of all the roots;
    then at each point the sign of S and of every member (trusted.upoly.sign_at's charge) and the unit per coefficient
    of count_roots. produce() finds its answer with far fewer evaluations than a Sturm count each, and without this a
    claim made within a budget could exceed the same budget in the checker."""
    from ...kernels import upoly as kernels
    from ...trusted.upoly import evaluation_factor, max_bits

    chain = kernels.sturm_chain(polynomial)
    budget.tick(len(chain))
    coefficients = sum(len(member) for member in chain)
    build = coefficients
    for f, g in zip(chain, chain[1:]):
        exponent, bits_f, bits_g = len(f) - len(g) + 1, 2 * max_bits(f), 2 * max_bits(g)
        build += exponent * len(f) * (1 + 2 * (((bits_f + exponent * bits_g) >> 6) * (bits_g >> 6) >> 8))
    budget.tick(2 * build)
    for p, q in points:
        budget.tick(coefficients + len(polynomial) * evaluation_factor(len(polynomial), p, q))
        for member in chain:
            budget.tick(len(member) * evaluation_factor(len(member), p, q))


# ---------------------------------------------------------------------------------------------------------------
# Expressions

class _Expression:
    """The exact comparison of a parsed closed expression with rationals (module docstring): the checker's
    enclosures, each computed once with its work recorded, and the exact decision of rationality, made once."""

    def __init__(self, budget: Budget, digits: Any, entries: list, limit: int) -> None:
        self.budget, self.digits, self.entries = budget, digits, entries
        self.top = budget.limits.max_integer_bits
        self.first = min(4 * limit.bit_length() + _GUARD_BITS, self.top)  # The checker's first precision.
        self.enclosures: dict[int, tuple[int, int, int]] = {}  # bits -> (L, H, the work that computing it took).
        self.exact: object = None  # None until decided: a Rational, or digits' marker for irrational or too many.

    def enclosure(self, bits: int) -> tuple[int, int, int]:
        """(L, H, work) with L/2**bits <= x <= H/2**bits, the checker's enclosure at that precision."""
        known = self.enclosures.get(bits)
        if known is None:
            before = self.budget.work
            low, high = self.digits._enclosure(self.budget, self.entries, bits)
            known = self.enclosures[bits] = (low, high, self.budget.work - before)
        return known

    def separated(self, bits: int, p: int, q: int) -> int:
        """sign(x - p/q) when the enclosure at that precision separates them, else 0: the checker's two tests."""
        low, high, _ = self.enclosure(bits)
        _charge(self.budget, max(low.bit_length(), high.bit_length(), p.bit_length() + bits), q.bit_length(), 2)
        target = p << bits
        if low * q > target:
            return 1
        if high * q < target:
            return -1
        return 0

    def decide(self) -> None:
        """Whether x is rational, decided once in the multiquadratic algebra."""
        if self.exact is None:
            self.exact = self.digits._exact_value(self.budget, self.entries)

    def rational(self) -> Rational | None:
        exact = self.exact
        return exact if type(exact) is Rational else None

    def precisions(self) -> list[int]:
        """The checker's precisions: the first, doubled up to max_integer_bits."""
        found = [self.first]
        while found[-1] < self.top:
            found.append(min(2 * found[-1], self.top))
        return found

    def compare(self, p: int, q: int) -> int:
        """sign(x - p/q) exactly; ResourceLimitError when no enclosure up to max_integer_bits separates them."""
        known = self.rational()
        if known is not None:
            return _cross_sign(self.budget, known.numerator, known.denominator, p, q)
        for bits in self.precisions():
            self.budget.tick()
            found = self.separated(bits, p, q)
            if found:
                return found
            self.decide()
            known = self.rational()
            if known is not None:
                return _cross_sign(self.budget, known.numerator, known.denominator, p, q)
        raise ResourceLimitError("Computation limit exceeded: no enclosure separates the value from a rational")

    def floor(self) -> int:
        """floor(x), from the first enclosure narrower than 1 and the exact comparisons with two integers."""
        for bits in self.precisions():
            self.budget.tick()
            low, high, _ = self.enclosure(bits)
            if high - low < 1 << bits:
                k = low >> bits  # k <= x < k + 2: x >= L/2**bits >= k and x <= H/2**bits < L/2**bits + 1.
                return k + 1 if self.compare(k + 1, 1) >= 0 else k
        # Wider than 1 at the last precision: only an exact rational value still has a floor here.
        self.decide()
        known = self.rational()
        if known is None:
            raise ResourceLimitError("Computation limit exceeded: no enclosure locates the value between integers")
        return known.numerator // known.denominator

    def replay(self, points: tuple[tuple[int, int], ...]) -> None:
        """The checker's comparisons of x with the points, charged as the checker charges them: every enclosure of
        its sequence once more, and ResourceLimitError where its formal value raises (more than MAX_RADICALS
        coprime non-square radicands when the first enclosure separates nothing)."""
        for p, q in points:
            for bits in self.precisions():
                self.budget.tick(self.enclosure(bits)[2] + 1)
                if self.separated(bits, p, q):
                    break
                self.decide()
                known = self.rational()
                if known is None and self.exact == self.digits._TOO_MANY:
                    raise ResourceLimitError(f"Computation limit exceeded: the value is not separated from a "
                                             f"rational at {bits} bits, and its more than {self.digits.MAX_RADICALS} "
                                             f"independent square roots leave no exact evaluation")
                if known is not None and (known.numerator, known.denominator) == (p, q):
                    break
            else:
                raise ResourceLimitError("Computation limit exceeded: no enclosure separates the value from a rational")


def _expression_answer(budget: Budget, digits: Any, entries: list, limit: int) -> Answer:
    """The answer for a parsed closed expression (module docstring)."""
    number = _Expression(budget, digits, entries, limit)
    floor = number.floor()
    if number.compare(floor, 1) == 0:
        answer, points = (_rational(budget, floor, 1), ()), ((floor, 1),)
    else:
        answer, points = _answer(budget, number.compare, floor, limit)
    number.replay(points)
    return answer
