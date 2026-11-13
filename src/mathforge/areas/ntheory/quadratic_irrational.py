"""Quadratic irrationals: their periodic continued fractions, the convergents of a rational, Pell's equation
(mathforge.ntheory).

quadratic_continued_fraction(p, q, d, r) takes exact integers with d >= 2 not a perfect square, q != 0 and r != 0,
and returns the simple continued fraction of the quadratic irrational (p + q*sqrt(d))/r as a
PeriodicContinuedFraction: the preperiod (a0, ..., a(s-1)) and the period (b0, ..., b(L-1)) that repeats forever
after it. Lagrange's theorem makes the expansion eventually periodic, and the canonical form of the record (below)
makes it unique: the period is the minimal one and the preperiod is the shortest. Evidence: ().

convergents(value) takes an exact rational, a Rational or an int, and returns the tuple of the convergents
p0/q0, ..., pn/qn of its canonical finite continued fraction (the one of continued_fraction, whose last of several
terms is at least 2): every convergent is a Rational, those with denominator 1 included. Evidence: ().

pell_fundamental(d) takes an exact integer d >= 2 that is not a perfect square and returns the tuple (x, y) of the
least solution of x^2 - d*y^2 = 1 in positive integers. Evidence: (period,), the minimal period of the continued
fraction of sqrt(d), a tuple of ints; the preperiod of sqrt(d) is (isqrt(d),), which the checker supplies itself.

The checker, mathforge.certify.ntheory.quadratic_irrational, states the theorems. It verifies an expansion (of
quadratic_continued_fraction, and that of sqrt(d) for pell_fundamental) only within its caps, beyond which it raises
ResourceLimitError: at most 99,000 terms in preperiod and period together, and a sum of the bit lengths of the
absolute values of all terms of at most max_integer_bits. The producer refuses the same expansions.

The producer, after docs/v09/cards/m1-ntheory.md:

- quadratic_continued_fraction. x = (p + q*sqrt(d))/r is a root of r^2*X^2 - 2*p*r*X + p^2 - d*q^2 and of the
  primitive equation a*X^2 - b*X + c = 0 that dividing the three coefficients by their gcd gives: the root
  (b + sign(q*r)*sqrt(b^2 - 4ac))/(2a). So x = (P + sqrt(D))/Q with P = sign(q*r)*b, Q = 2*sign(q*r)*a and
  D = b^2 - 4ac, or, for an even b, P = sign(q*r)*b/2, Q = sign(q*r)*a and D = (b/2)^2 - ac; Q divides D - P^2,
  which is -4ac or -ac. As b^2 - 4ac is the discriminant of the number itself, the integers of the expansion, and so
  the work of every step, do not depend on how the arguments write the number. Multiplying the arguments' fraction
  by k (P = p*k, Q = r*k, D = d*q^2*k^2 for q > 0), with the card's k = |r| or the least one, |r|/gcd(r, d*q^2 - p^2),
  also gives a state form, but a common factor of p, q and r, or a square factor of d that p and r absorb, then
  multiplies all its integers. Written with such a factor, the period of 47,000 ones that the test
  test_work_depends_on_the_number_only expands took twice the work with the least k: 1.8 million units of the
  default 2 million. Every state form gives the same terms. From a complete quotient (P + sqrt(D))/Q with
  Q | D - P^2, s = isqrt(D) gives the term a = floor((P + sqrt(D))/Q) exactly, sqrt(D) being irrational: (P + s)//Q
  for Q > 0 and -((P + s)//(-Q) + 1) for Q < 0. The next complete quotient is (P' + sqrt(D))/Q' with P' = a*Q - P
  and Q' = (D - P'^2)/Q, which the identity Q' = Q_before + a*(P - P') gives without a long division, Q_before being
  the Q of the complete quotient before (Q_before*Q = D - P^2). A complete quotient (P + sqrt(D))/Q is a pair of
  integers once D is fixed, since sqrt(D) is irrational. The tail of terms from a complete quotient on is purely
  periodic exactly when that quotient is reduced, greater than 1 with its conjugate in (-1, 0) (Galois); in
  integers: Q > 0, P <= s < P + Q and Q <= P + s. So the shortest preperiod ends at the first reduced complete
  quotient, and the minimal period at the first return to it: the card's "first repeated state", found without
  keeping every state, whose integers may be long.
- convergents. The terms of the Euclidean algorithm (floor division, as continued_fraction), then the forward
  recurrence p(k) = a(k)*p(k-1) + p(k-2) and q(k) = a(k)*q(k-1) + q(k-2) from p(-1) = 1, p(-2) = 0, q(-1) = 0,
  q(-2) = 1. Since p(k)*q(k-1) - p(k-1)*q(k) = (-1)^(k-1), every pair is coprime with q(k) >= 1, so each Rational is
  built in lowest terms without a gcd; |p(k)| <= |p| and q(k) <= q keep it within the bits of the input.
- pell_fundamental. The expansion of sqrt(d) = (0 + 1*sqrt(d))/1 above, whose preperiod is (isqrt(d),) and whose
  period of length L ends with 2*isqrt(d); then (h, k) = p(L-1)/q(L-1), the convergent of isqrt(d) followed by the
  period without its last term, is the value for even L, and (h^2 + d*k^2, 2*h*k) for odd L.

Limits. Every step charges the request budget before it runs: one unit per operation, the size of long products and
divisions with trusted.upoly.charge_product (the checker's measure), and for integers of thousands of bits their
linear work as well, so that max_work bounds the time of long state forms. The producer refuses with
ResourceLimitError, as soon as the expansion passes them, the expansions beyond the checker's caps (_MAX_TERMS terms,
max_integer_bits bits of terms), and every value whose Result the verifier would not admit under the request's
max_evidence_items. The verifier counts every item of the stored Result among its evidence items
(verification_v6.verify_result, Budget.inspect): the value twice (in the Result and in its step), the evidence once,
the request twice (in the Result and in the step's inputs), and the Result's own fields with its four stored checks
(_RESULT_ITEMS). So every Result that run() returns verifies again under the limits it was made with, as does the
Result decoded from its JSON text. At the default limits this leaves room for 49,954 terms of an expansion (whose
term bits mostly reach their cap first), 16,654 convergents of a Rational, and 99,926 period terms of sqrt(d), more
than the cap of 99,000 terms allows. The convergents of a rational may hold far more bits than the rational itself
(each up to its size), so their integers are charged one work unit per 128 bits as they are built: at the default
limits a value of up to about 250 million bits, which covers every ratio of Fibonacci numbers whose convergents the
verifier admits, and much more than a JSON document of the default decode limits holds (16 MB). A Pell solution whose
x or y exceeds max_integer_bits is refused as well, as the verifier would refuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Rational, trusted_rational
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

_SCOPE = "exact integers; certified primes, orders and residues; canonical integer solution families"

SPECS = (
    OperationSpec("quadratic_continued_fraction", "ntheory",
                  (Argument("p", "int", None, "int"), Argument("q", "int", None, "int"),
                   Argument("d", "int", None, "int"), Argument("r", "int", None, "int")),
                  "PeriodicContinuedFraction",
                  "Expand (p + q*sqrt(d))/r as an eventually periodic simple continued fraction.", _SCOPE,
                  version=6),
    OperationSpec("convergents", "ntheory",
                  (Argument("value", "Rational | int", None, "rational"),),
                  "tuple[Rational, ...]", "List the convergents of the continued fraction of an exact rational.",
                  _SCOPE, version=6),
    OperationSpec("pell_fundamental", "ntheory",
                  (Argument("d", "int", None, "int"),),
                  "tuple[int, int]", "Find the fundamental solution of the Pell equation x^2 - d*y^2 = 1.",
                  _SCOPE, version=6),
)

# The checker's cap on the terms of a verified expansion, preperiod and period together
# (mathforge.certify.ntheory.quadratic_irrational.MAX_TERMS, which a producer must not import; a test compares them).
_MAX_TERMS = 99_000

# What the verifier counts among its evidence items besides the value (twice) and the evidence (once): the Result's
# own fields with its four stored checks, and the request twice. A request counts _REQUEST_ITEMS and 3 items per
# argument (the pair, the name and an int), a Rational argument 2 more (its two integers). The verifier walks the whole
# Result (Budget.inspect), so these numbers follow the layout of contracts.Result; the boundary tests of
# tests/v09/ntheory/test_quadratic_irrational.py measure that walk and fail when the layout changes.
_RESULT_ITEMS = 54
_REQUEST_ITEMS = 3  # The request, its operation name and its tuple of arguments; the format version is not counted.


def _room(budget: Budget, arguments: int, rationals: int = 0) -> int:
    """The evidence items that the verifier leaves for the value, which it counts twice, and the evidence, which it
    counts once, in the Result of a request with the given number of arguments, rationals of them Rationals."""
    request = _REQUEST_ITEMS + 3 * arguments + 2 * rationals
    return budget.limits.max_evidence_items - _RESULT_ITEMS - 2 * request


def _repeats(word: tuple[int, ...], budget: Budget) -> bool:
    """Whether a nonempty word is a repetition of a shorter block. Testing the prime divisors k of its length
    suffices: a word u^m with m >= 2 is also the k-th power of u^(m/k) for every prime k that divides m."""
    length = rest = len(word)
    factor = 2
    while factor * factor <= rest:
        budget.tick()
        if rest % factor == 0:
            budget.tick(1 + (length >> 6))  # Building and comparing the k-th power is linear in the length.
            if word == word[:length // factor] * factor:
                return True
            while rest % factor == 0:
                budget.tick()
                rest //= factor
        factor += 1
    if rest > 1:
        budget.tick(1 + (length >> 6))
        return word == word[:length // rest] * rest
    return False


@dataclass(frozen=True)
class PeriodicContinuedFraction:
    """[a0; a1, ..., a(s-1), (b0, ..., b(L-1))]: the preperiod, then the period repeated forever.

    The canonical form, which each quadratic irrational has exactly once: tuples of exact ints; a nonempty period
    whose terms are all >= 1; preperiod terms after the first >= 1 (the first, a0, is any integer); a period word
    that is primitive, no repetition of a shorter block (the minimal period); and an empty preperiod or one whose
    last term differs from the last period term (the shortest preperiod)."""
    preperiod: tuple[int, ...]
    period: tuple[int, ...]

    def __post_init__(self):
        with computation() as budget:
            preperiod, period = self.preperiod, self.period
            if type(preperiod) is not tuple or type(period) is not tuple or not period:
                raise InvalidInput("A periodic continued fraction has a tuple preperiod and a nonempty tuple period")
            for index, term in enumerate(preperiod):
                budget.tick()
                if type(term) is not int:  # Neither bool nor float is an exact integer.
                    raise InvalidInput("Continued fraction terms are exact integers")
                budget.check_int(term)
                if index and term < 1:
                    raise InvalidInput("The preperiod terms after the first are positive")
            for term in period:
                budget.tick()
                if type(term) is not int:
                    raise InvalidInput("Continued fraction terms are exact integers")
                budget.check_int(term)
                if term < 1:
                    raise InvalidInput("The period terms are positive")
            if preperiod and preperiod[-1] == period[-1]:
                # Otherwise the last preperiod term would start the period, rotated by one: a shorter preperiod.
                raise InvalidInput("The last preperiod term differs from the last period term")
            if _repeats(period, budget):
                raise InvalidInput("The period repeats no shorter block")


TYPES = {"periodic_continued_fraction": PeriodicContinuedFraction}
CHECKER = "mathforge.certify.ntheory.quadratic_irrational"


def produce(operation, args):
    """(value, evidence) of an operation of the family, by the algorithms of the module docstring."""
    if operation == "quadratic_continued_fraction":
        return _produce_expansion(args)
    if operation == "convergents":
        return _produce_convergents(args)
    if operation == "pell_fundamental":
        return _produce_pell(args)
    raise UnsupportedOperation(f"{operation} is not an operation of the quadratic_irrational family")


def _irrational_root(budget: Budget, d: int, operation: str) -> int:
    """isqrt(d) for an exact int d that run() or the caller has checked; InvalidInput unless d >= 2 is no perfect
    square, so that sqrt(d) is a real irrational number."""
    from ...trusted.upoly import charge_product  # Loaded on first use: importing mathforge loads no trusted module.
    if d < 2:
        raise InvalidInput(f"{operation} requires d >= 2: sqrt(d) is rational or not real for d < 2")
    bits = d.bit_length()
    budget.tick(2)
    charge_product(bits, bits, 2)  # The square root, charged as a product of d with itself, and its square.
    root = isqrt(d)
    if root * root == d:
        raise InvalidInput(f"{operation} requires d that is no perfect square: sqrt({d}) is an integer"
                           if bits <= 64 else f"{operation} requires d that is no perfect square")
    return root


def _expansion(budget: Budget, p: int, q: int, d: int, r: int,
               admitted: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The canonical (preperiod, period) of the continued fraction of (p + q*sqrt(d))/r, for exact ints with d >= 2
    no perfect square, q != 0 and r != 0: the state form of the module docstring. ResourceLimitError as soon as the
    terms, preperiod and period together, pass the checker's caps or number more than admitted, the most that the
    verifier admits in the caller's value and evidence."""
    from ...trusted.upoly import charge_gcd, charge_product
    max_bits = budget.limits.max_integer_bits
    max_terms = min(_MAX_TERMS, admitted)
    bits = max(p.bit_length(), q.bit_length(), d.bit_length(), r.bit_length())
    budget.tick(6)
    charge_product(bits, 2 * bits + 1, 6)  # p*p, q*q, d*q^2, r*r, 2*p*r and |r|*gcd(r, 2p), before they run.
    norm = p * p - d * q * q  # Nonzero: p^2 = d*q^2 would make d a square.
    budget.tick()
    charge_gcd(bits + 1)
    multiple = abs(r) * gcd(r, 2 * p)  # gcd(r^2, 2*p*r) = |r|*gcd(r, 2p).
    budget.tick()
    charge_gcd(max(norm.bit_length(), multiple.bit_length()))
    common = gcd(multiple, norm)  # The gcd of the three coefficients of r^2*X^2 - 2*p*r*X + p^2 - d*q^2.
    square, twice = r * r, 2 * p * r
    widest = max(norm.bit_length(), square.bit_length(), twice.bit_length())
    budget.tick(3)
    charge_product(widest, widest - common.bit_length() + 1, 3)  # The exact divisions by common, before they run.
    a, b, c = square // common, twice // common, norm // common  # The primitive equation a*X^2 - b*X + c = 0.
    sign = 1 if (q > 0) == (r > 0) else -1  # x = (b + sign*sqrt(b^2 - 4ac))/(2a), the root that sign(q*r) selects.
    budget.tick(2)
    charge_product(max(a.bit_length(), b.bit_length()), max(b.bit_length(), c.bit_length()) + 2, 2)  # b^2, a*c.
    if b & 1:
        big_p, big_q, big_d, before = sign * b, 2 * sign * a, b * b - 4 * a * c, -2 * sign * c
    else:  # The discriminant over 4: x = (b/2 + sign*sqrt((b/2)^2 - ac))/a.
        half = b >> 1
        big_p, big_q, big_d, before = sign * half, sign * a, half * half - a * c, -sign * c
    square_bits = big_d.bit_length()
    budget.tick()
    charge_product(square_bits, square_bits)  # isqrt(D), charged like a product of D with itself.
    s = isqrt(big_d)
    terms: list[int] = []
    term_bits = 0
    start = -1  # The index of the first reduced complete quotient, once there is one.
    first = (0, 0)  # Its state (P, Q).
    while True:
        if start < 0 and big_q > 0 and big_p <= s < big_p + big_q and big_q <= big_p + s:
            start, first = len(terms), (big_p, big_q)
        top = big_p + s
        size = max(top.bit_length(), big_q.bit_length(), before.bit_length())
        # Before they run: the additions and comparisons of the step, linear in the size of its integers (a unit per
        # 2,048 bits), and the long division, whose quotient has at most bitlen(P + s) - bitlen(Q) + 1 bits.
        budget.tick(3 + (size >> 11))
        charge_product(top.bit_length(), max(1, top.bit_length() - big_q.bit_length() + 1))
        term = top // big_q if big_q > 0 else -(top // -big_q) - 1  # floor((P + sqrt(D))/Q)
        terms.append(term)
        term_bits += term.bit_length()  # bitlen(t) = ceil(log2(|t| + 1)), the checker's measure, also for a0 < 0.
        if len(terms) > max_terms:
            if max_terms < _MAX_TERMS:
                raise ResourceLimitError(f"Computation limit exceeded: continued fraction terms that the verifier "
                                         f"admits under max_evidence_items ({max_terms})")
            raise ResourceLimitError(f"Computation limit exceeded: continued fraction terms ({_MAX_TERMS})")
        if term_bits > max_bits:
            raise ResourceLimitError("Computation limit exceeded: continued fraction term bits")
        charge_product(term.bit_length(), size, 2)  # a*Q and a*(P - P'), before they run.
        following = term * big_q - big_p
        big_p, big_q, before = following, before + term * (big_p - following), big_q
        if start >= 0 and (big_p, big_q) == first:
            return tuple(terms[:start]), tuple(terms[start:])


def _produce_expansion(args: dict[str, object]) -> tuple[PeriodicContinuedFraction, tuple[()]]:
    p, q, d, r = args["p"], args["q"], args["d"], args["r"]
    for name, item in (("p", p), ("q", q), ("d", d), ("r", r)):
        if type(item) is not int:  # Neither bool nor float is an exact integer.
            raise InvalidInput(f"quadratic_continued_fraction requires an exact integer {name}")
    with computation() as budget:
        budget.tick()
        for item in (p, q, d, r):
            budget.check_int(item)  # run() refuses longer arguments already; a direct call is refused here.
        if q == 0:
            raise InvalidInput("quadratic_continued_fraction requires q != 0: (p + 0*sqrt(d))/r is rational")
        if r == 0:
            raise InvalidInput("quadratic_continued_fraction requires r != 0: the denominator of (p + q*sqrt(d))/r")
        _irrational_root(budget, d, "quadratic_continued_fraction")
        # The value counts the record, its two tuples and every term, twice; the evidence () counts one item.
        admitted = max(0, (_room(budget, 4) - 7) // 2)
        preperiod, period = _expansion(budget, p, q, d, r, admitted)
        return PeriodicContinuedFraction(preperiod, period), ()


def _produce_convergents(args: dict[str, object]) -> tuple[tuple[Rational, ...], tuple[()]]:
    value = args["value"]
    if type(value) is int:
        numerator, denominator = value, 1
    elif type(value) is Rational:
        # A Rational made by its constructor holds ints in lowest terms with a positive denominator; one made with
        # object.__new__ may hold anything, or lack a field.
        numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput("convergents requires a Rational of two exact integers with a positive denominator")
    else:  # A float, a bool or an expression is not an exact rational.
        raise InvalidInput("convergents requires an exact rational: a Rational or an int")
    from ...trusted.upoly import charge_gcd, charge_product
    with computation() as budget:
        budget.tick()
        budget.check_int(numerator)  # run() refuses longer arguments already; a direct call is refused here.
        budget.check_int(denominator)
        if denominator != 1:
            charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput("convergents requires a Rational in lowest terms")
        # The value counts its tuple and three items per Rational, twice; the evidence () counts one item. The request
        # holds the argument as it came: an int, or a Rational of three items.
        max_count = max(0, (_room(budget, 1, 1 if type(value) is Rational else 0) - 3) // 6)
        terms: list[int] = []
        p, q = numerator, denominator
        while True:
            # A floor division whose quotient is short in all but a few steps (their quotients multiply to at most
            # the denominator), and its remainder: linear in the size of p, a unit per 2,048 bits, before it runs.
            budget.tick(1 + (p.bit_length() >> 11))
            charge_product(p.bit_length(), max(1, p.bit_length() - q.bit_length() + 1))
            term, rest = divmod(p, q)  # q > 0: term = floor(p/q) and 0 <= rest < q.
            terms.append(term)
            if len(terms) > max_count:
                raise ResourceLimitError(f"Computation limit exceeded: convergents that the verifier admits "
                                         f"({max_count}, under max_evidence_items)")
            if not rest:
                break
            p, q = q, rest
        convergents: list[Rational] = []
        p, p_before, q, q_before = 1, 0, 0, 1  # p(k-1), p(k-2), q(k-1), q(k-2) for k = 0.
        for term in terms:
            charge_product(term.bit_length(), max(p.bit_length(), q.bit_length()), 2)  # Before the two products.
            p, p_before = term * p + p_before, p
            q, q_before = term * q + q_before, q
            # A unit per 128 bits of the convergent, which the value keeps (a bound on its memory), then the unit and
            # the bit check of every exact rational result.
            budget.tick((p.bit_length() + q.bit_length()) >> 7)
            budget.charge_rational(p, q)
            convergents.append(trusted_rational(p, q))  # In lowest terms with q >= 1 (module docstring).
        return tuple(convergents), ()


def _produce_pell(args: dict[str, object]) -> tuple[tuple[int, int], tuple[tuple[int, ...]]]:
    d = args["d"]
    if type(d) is not int:  # Neither bool nor float is an exact integer.
        raise InvalidInput("pell_fundamental requires an exact integer d")
    from ...trusted.upoly import charge_product
    with computation() as budget:
        budget.tick()
        budget.check_int(d)  # run() refuses a longer argument already; a direct call is refused here.
        root = _irrational_root(budget, d, "pell_fundamental")
        # The value (x, y) counts three items, twice; the evidence (period,) counts two tuples and the L period terms,
        # which are all the terms but isqrt(d): 8 + L items, so the 1 + L terms of the expansion leave 7 of the room.
        admitted = max(0, _room(budget, 1) - 7)
        preperiod, period = _expansion(budget, 0, 1, d, 1, admitted)  # sqrt(d) = [isqrt(d); (period)].
        if preperiod != (root,):  # The expansion of sqrt(d) cannot have another preperiod.
            raise ArithmeticError("The continued fraction of sqrt(d) has the preperiod (isqrt(d),)")
        h, h_before, k, k_before = 1, 0, 0, 1
        for term in (root, *period[:-1]):  # The convergent p(L-1)/q(L-1) = [isqrt(d); b0, ..., b(L-2)].
            budget.tick()
            charge_product(term.bit_length(), max(h.bit_length(), k.bit_length()), 2)
            h, h_before = term * h + h_before, h
            k, k_before = term * k + k_before, k
        if len(period) & 1:  # h^2 - d*k^2 = -1: the least solution of +1 is (h + k*sqrt(d))^2.
            size = max(h.bit_length(), k.bit_length())
            budget.tick(4)
            charge_product(size + d.bit_length(), size, 4)  # h*h, d*k, (d*k)*k and 2*h*k, before they run.
            h, k = h * h + d * k * k, 2 * h * k
        budget.check_int(h)  # The verifier refuses a longer x or y.
        budget.check_int(k)
        return (h, k), (period,)


def _render(fraction, renderer, child):
    r"""[1; (2)], [-2; 1, 1, (2)] or [(1)], the period in parentheses; in LaTeX [1; \overline{2}], [\overline{1}]."""
    period = renderer.join((child(term) for term in fraction.period), ", ")
    tail = r"\overline{" + period + "}" if renderer.latex else renderer.parens(period)
    if not fraction.preperiod:
        return "[" + tail + "]"
    head, *rest = (child(term) for term in fraction.preperiod)
    return "[" + head + "; " + renderer.join([*rest, tail], ", ") + "]"


RENDER = {PeriodicContinuedFraction: _render}
