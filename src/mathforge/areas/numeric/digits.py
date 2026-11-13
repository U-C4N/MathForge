"""Decimal digits of exact real numbers (mathforge.numeric): the decimal expansion truncated toward zero.

digits(value, places) takes an exact real number, value (MCP kind "scalar", annotation "Expression | int"), and an
exact int places with 0 <= places <= MAX_PLACES (kind "int"; any other places is invalid input). The value is an int,
a Rational, a RealAlgebraicRoot, or a closed legacy expression built from Rationals and square roots of Rationals:
a tree of the mathforge.model nodes Add, Mul, Pow (an exact int exponent >= 0) and Sqrt (a Rational radicand >= 0)
over Rationals, the scope of mf.algebraic without its RealAlgebraicRoot leaves. Anything else is invalid input: a
bool, a float, a str, a Symbol (no closed number), an expression with a RealAlgebraicRoot inside it, which the checker
does not evaluate, and a record made with object.__new__ outside its canonical form.

The value is the TruncatedDecimal record (sign, magnitude, places, exact) of the number x: sign the exact sign of x
(-1, 0 or 1), magnitude = floor(|x| * 10**places), places the argument, and exact whether |x| * 10**places is an
integer, so that x = sign * (magnitude + f) / 10**places with 0 <= f < 1 and f == 0 exactly when exact. The canonical
form, which each claim has exactly once: sign 0 exactly when the number is 0, and then magnitude 0 and exact True;
a nonzero number never has magnitude 0 with exact True.

The evidence, in the layout that the checker, mathforge.certify.numeric.digits, verifies:

- () for an int or a Rational value;
- ("root",) for a RealAlgebraicRoot value: the checker decides everything with Sturm counts of its own;
- ("bits", b) for an expression, b an exact int with 1 <= b <= max_integer_bits of the verifying budget (65,536 by
  default): the checker's outward-rounded enclosure of the expression at precision b (its module docstring defines
  it step by step) lies strictly inside the open decimal cell of the claim; it never proves a number on a cell
  boundary, so the claim has exact False and a nonzero sign;
- ("rational", r) for an expression, r a Rational in lowest terms (an integral one included, never an int): the
  expression equals r, which the checker re-verifies in the multiquadratic field of mathforge.trusted.surd over the
  natural coprime base of the expression's radicands; a number on a decimal boundary, such as
  Mul((Sqrt(2), Sqrt(2), Rational(1, 2))) at places 3 (1.000, exact), needs it. The checker raises
  ResourceLimitError when that base holds more than ten numbers that are no squares.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/numeric/test_digits.py is the definition of done.

How produce() computes (docs/v09/cards/m2-algebra-equations-sets-numeric.md). Nothing here is a proof: the checker
verifies every claim with methods of its own (integer comparisons, Sturm counts, its own enclosure, the field).

- An int or a Rational p/q: magnitude, rest = divmod(|p| * 10**places, q), exact when rest is 0.
- A RealAlgebraicRoot (S, i), validated in the canonical form that its constructor makes: Sturm counts on
  kernels.upoly's chain isolate alpha_i, the i-th real root of S, alone (_isolate: the geometric mean of a wide
  one-sided interval, else the midpoint), and a point interval or a linear S gives a rational root, written as above.
  A bracket whose ends differ greatly in magnitude is narrowed by bisecting the exponent (_magnitude_bracket).
  Otherwise alpha_i is a simple root of the square-free S, which changes its sign there and nowhere else in the
  interval: the sign of S at a point t of the interval tells whether t lies below, at or above
  alpha_i. The card's bisection on the isolating interval is accelerated by Newton's method on dyadic brackets: from
  the midpoint of the bracket, a Newton step at about twice the bracket's precision gives a candidate, and the exact
  signs of S at two points around it, 2**guard squared bracket widths apart, confirm a new bracket; when they do not,
  a bisection step halves the bracket and the guard widens by 8 bits. Every step at least halves the bracket, and from
  the time Newton's method takes hold each one about doubles its precision. Once the bracket is narrower than
  10**-places, at most one decimal point m/10**places lies inside it, and the exact sign of S there decides the cell
  and exactness. The Newton steps measure precision relative to the size of the root, and their radii are planned
  backwards from the last one (_radius), so that the last step lands on the final precision. Plain bisection to
  10,000 places would take some 33,000 exact evaluations on numbers of up to 100,000 bits, far past the default
  budget; the brackets take about a dozen.
- An expression, validated node by node (shared nodes once, no node inside itself): produce() computes the checker's
  enclosure, by the rules of the checker's module docstring, from b = ceil(10 * places / 3) + 64 bits (log2(10) is
  below 10/3) and doubling up to max_integer_bits, until it lies strictly inside an open decimal cell: that cell gives
  the sign and the magnitude, and ("bits", b) is the evidence. An enclosure narrower than one cell that still meets a
  decimal point (or 0) may hold a number on that point, which no enclosure separates, such as sqrt(2)*sqrt(2)/2 at
  places 3. produce() then decides, exactly and once, whether the expression is rational: the catalogue's way (2.2),
  in the multiquadratic algebra over the natural coprime base of its radicands. The products of the square roots of
  pairwise coprime non-squares over the subsets of them are linearly independent over Q (Besicovitch), so the
  expression's formal value is a constant exactly when the number is rational. A rational r gives its digits as above
  and the evidence ("rational", r); an irrational number lies on no decimal point, and the enclosures go on. The
  last precision, max_integer_bits, makes the same decision when no earlier enclosure was that narrow. A base
  with more than MAX_RADICALS non-squares leaves only the enclosures (the checker refuses such an algebra as well),
  and a number that max_integer_bits bits of enclosure do not separate from a decimal point raises
  ResourceLimitError.

  The card routes expressions through mf.algebraic to a RealAlgebraicRoot. That normalizer eliminates by resultants
  of degree up to 2**t and runs out of the default budget on sqrt(2) + sqrt(3) + sqrt(5) + sqrt(7) + sqrt(11), whose
  enclosure takes microseconds; the enclosure is what the evidence names anyway, so the producer reads the digits of
  an expression from it, and decides rationality by the field as catalogue 2.2 asks.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges a unit for the call, refuses an integer longer than max_integer_bits, an
expression of more than max_nodes distinct nodes or deeper than max_depth with ResourceLimitError, and charges every
further step before it runs, on the scale of trusted.upoly: a product, a division or a square root of long integers by
charge_product, a gcd by charge_gcd, an evaluation of a polynomial by size_factor per coefficient (one product per
step at a dyadic point, two at a rational one), and the operations of the algebra in batches by the checker's own
formulas, as 10**places is charged by the checker's chain of products: the producer's work is meant to bound the
checker's, so that a claim produced within a budget verifies within the same one; a test compares them,
test_claims_verify_within_the_producer_budget. An enclosure end longer than max_integer_bits + b bits raises
ResourceLimitError, where the checker raises it too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import gcd, isqrt
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

# The scope text of every operation of wave M2 (docs/v09/cards/m2-algebra-equations-sets-numeric.md).
_SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
          "and decimal digits")
# The most places that digits writes (the card); the checker keeps its own copy, which a test compares.
MAX_PLACES = 10_000
# The most coprime non-square radicands whose algebra decides rationality: the checker's ceiling on the
# ("rational", r) layout (mathforge.certify.numeric.digits.MAX_RADICALS), which a test compares.
MAX_RADICALS = 10
# Guard bits of the first enclosure beyond the bits of the places.
_GUARD_BITS = 64

SPECS = (
    OperationSpec("digits", "numeric",
                  (Argument("value", "Expression | int", None, "scalar"), Argument("places", "int", None, "int")),
                  "TruncatedDecimal",
                  "Write the decimal expansion of an exact real number truncated toward zero to a number of places.",
                  _SCOPE, version=6),
)


@dataclass(frozen=True)
class TruncatedDecimal:
    """sign * (magnitude + f) / 10**places with 0 <= f < 1; exact when f == 0.

    sign is the exact sign of the number (-1, 0 or 1), magnitude = floor(|value| * 10**places) and exact whether
    |value| * 10**places is an integer. The canonical form: sign, magnitude and places exact ints (a bool is none),
    magnitude >= 0 and places >= 0 within max_integer_bits, exact a bool, and sign == 0 exactly when magnitude == 0
    and exact (the number 0); a nonzero number with magnitude 0 is never exact."""
    sign: int
    magnitude: int
    places: int
    exact: bool

    def __post_init__(self) -> None:
        with computation() as budget:
            budget.tick()
            if type(self.sign) is not int or self.sign not in (-1, 0, 1):
                raise InvalidInput("The sign of a truncated decimal is the int -1, 0 or 1")
            if type(self.magnitude) is not int or self.magnitude < 0:
                raise InvalidInput("The magnitude of a truncated decimal is an exact int >= 0")
            budget.check_int(self.magnitude)
            if type(self.places) is not int or self.places < 0:
                raise InvalidInput("The places of a truncated decimal are an exact int >= 0")
            budget.check_int(self.places)
            if type(self.exact) is not bool:
                raise InvalidInput("Whether a truncated decimal is exact is a bool")
            if (self.sign == 0) != (self.magnitude == 0 and self.exact):
                raise InvalidInput("A truncated decimal has sign 0 exactly when it is exactly 0")


TYPES = {"truncated_decimal": TruncatedDecimal}
CHECKER = "mathforge.certify.numeric.digits"

_ARGUMENTS = frozenset(("value", "places"))
_SCOPE_ERROR = ("digits requires an exact real number: an int, a Rational, a RealAlgebraicRoot or a closed expression "
                "of Rationals and square roots of Rationals with sums, products and powers >= 0")
_MISSING = object()  # A field that a node made with object.__new__ lacks.
_IRRATIONAL = "irrational"  # _exact_value: the formal value is no constant.
_TOO_MANY = "too many radicals"  # _exact_value: more than MAX_RADICALS coprime non-square radicands.

Entry = tuple  # A node of a parsed expression (_parse): ("rational", p, q), ("sqrt", p, q), ("add", children), ...
Element = dict  # An element of the multiquadratic algebra: {subset mask: nonzero Rational coefficient}.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(TruncatedDecimal, evidence) of digits(value, places), for arguments that operations_v6.run has checked by name
    (and places by kind); the module states the algorithms and the evidence. InvalidInput for places outside 0 to
    MAX_PLACES and for a value outside the scope, ResourceLimitError past the budget."""
    if type(operation) is not str or operation != "digits":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the digits family")
    if type(args) is not dict or args.keys() != _ARGUMENTS:
        raise InvalidInput("digits takes exactly the arguments value and places")
    value, places = args["value"], args["places"]
    if type(places) is not int or not 0 <= places <= MAX_PLACES:  # A bool, a float or a str is no exact int.
        raise InvalidInput(f"places of digits must be an exact int from 0 to {MAX_PLACES:,}")
    from ...algebraic import RealAlgebraicRoot

    with computation() as budget:
        budget.tick()  # The call itself.
        kind = type(value)
        if kind is int:
            budget.check_int(value)
            return _rational_digits(budget, value, 1, places), ()
        if kind is Rational:
            return _rational_digits(budget, *_fraction(budget, value), places), ()
        if kind is RealAlgebraicRoot:
            polynomial, interval = _root(budget, value)
            return _root_digits(budget, polynomial, interval, places), ("root",)
        if kind is Add or kind is Mul or kind is Pow or kind is Sqrt:
            return _expression_digits(budget, _parse(budget, value), places)
        raise InvalidInput(_SCOPE_ERROR)


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic and rationals

def _charge(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, count products, divisions or square roots of integers of the given bit lengths: a
    unit each and what their lengths cost beyond it (trusted.upoly.charge_product)."""
    from ...trusted.upoly import charge_product

    budget.tick(count)
    charge_product(left_bits, right_bits, count)


def _gcd(budget: Budget, left: int, right: int) -> int:
    """gcd(left, right), charged before it runs by the length of the longer number (trusted.upoly.charge_gcd)."""
    from ...trusted.upoly import charge_gcd

    budget.tick()
    charge_gcd(max(left.bit_length(), right.bit_length()))
    return gcd(left, right)


def _ten_to(budget: Budget, places: int) -> int:
    """10**places, charged before it runs as the chain of products by squaring that the checker forms for it (each
    product by the lengths of its factors), so that the producer's work bounds the checker's."""
    budget.tick()
    result_bits, base_bits, exponent = 1, 4, places
    while exponent:
        budget.tick()
        if exponent & 1:
            _charge(budget, result_bits, base_bits)
            result_bits += base_bits
        exponent >>= 1
        if exponent:
            _charge(budget, base_bits, base_bits)
            base_bits *= 2
    return 10 ** places


def _fraction(budget: Budget, value: object) -> tuple[int, int]:
    """(numerator, denominator) of a Rational in the canonical form of its constructor: two exact ints in lowest terms
    with a positive denominator, each within max_integer_bits. InvalidInput for a Rational made with object.__new__
    outside that form, ResourceLimitError for an integer past the budget."""
    numerator, denominator = getattr(value, "numerator", _MISSING), getattr(value, "denominator", _MISSING)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput("A Rational holds two exact ints, the denominator positive")
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1 and _gcd(budget, numerator, denominator) != 1:
        raise InvalidInput("A Rational is in lowest terms")
    return numerator, denominator


def _rational_digits(budget: Budget, numerator: int, denominator: int, places: int) -> TruncatedDecimal:
    """The record of numerator/denominator (denominator >= 1): magnitude, rest = divmod(|p| * 10**places, q)."""
    scale = _ten_to(budget, places)
    _charge(budget, numerator.bit_length(), scale.bit_length())
    scaled = abs(numerator) * scale
    _charge(budget, scaled.bit_length(), denominator.bit_length())
    magnitude, rest = divmod(scaled, denominator)
    budget.check_int(magnitude)  # The record holds it; past max_integer_bits the budget refuses the claim.
    return TruncatedDecimal((numerator > 0) - (numerator < 0), magnitude, places, not rest)


# ---------------------------------------------------------------------------------------------------------------
# Real algebraic roots

def _root(budget: Budget, value: object) -> tuple[tuple[int, ...], tuple[tuple[int, int], tuple[int, int]]]:
    """(S, isolating interval of alpha_i) of a RealAlgebraicRoot (S, i) in the canonical form that its constructor
    makes: S a tuple of exact ints, lowest degree first, of degree >= 1, primitive and square-free with a positive
    leading coefficient, and i an exact int below the number of real roots of S. The interval is
    _isolate's ((num, den), (num, den)), a point for a root met by a probe. InvalidInput for a root
    made with object.__new__ outside that form, ResourceLimitError for a coefficient past the budget."""
    from ...kernels import upoly as kernels

    coefficients = getattr(value, "integer_coefficients", _MISSING)
    index = getattr(value, "real_index", _MISSING)
    if type(coefficients) is not tuple or len(coefficients) < 2 or type(index) is not int or index < 0:
        raise InvalidInput("A RealAlgebraicRoot holds a tuple of integer coefficients and an int index >= 0")
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not int:
            raise InvalidInput("The coefficients of a RealAlgebraicRoot are exact ints")
        budget.check_int(coefficient)
    if coefficients[-1] <= 0 or kernels.squarefree_part(coefficients) != coefficients:
        raise InvalidInput("A RealAlgebraicRoot holds the primitive square-free polynomial that its constructor makes")
    return coefficients, _isolate(budget, coefficients, index)


def _isolate(budget: Budget, polynomial: tuple[int, ...],
             index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """An interval (low, high] of dyadic rationals, ((num, den), (num, den)), that holds alpha_index, the index-th
    real root of the square-free polynomial, and no other root, with low no root; a point (t, t) when a probe meets
    alpha_index itself. InvalidInput when the polynomial has no root of that index.

    Sturm counts on kernels.upoly's chain, as kernels.upoly.isolate makes them, but for the one root asked for:
    (low, high] starts at (-bound, bound], bound = kernels.upoly.root_bound, and each probe keeps the half that holds
    alpha_index. isolate separates every root from every other, and its plain bisection from the bound needs one
    level per bit between the bound and the roots' separation: roots 2**-300 apart below a bound of 2**300 took it
    beyond the default budget. The probe is 0 for an interval around 0, the geometric mean 2**e of a one-sided
    interval whose ends differ by more than a factor of 8 (the lower root bound of _magnitude_bracket stands in for
    an end at 0), and the midpoint otherwise."""
    from ...kernels import upoly as kernels

    chain = kernels.sturm_chain(polynomial)
    bound = kernels.root_bound(polynomial)
    shift = 0
    low, high = -bound, bound  # The ends low/2**shift and high/2**shift; no root lies outside (-bound, bound).
    low_variations, low_zero = kernels.endpoint(chain, low, 1)
    high_variations, high_zero = kernels.endpoint(chain, high, 1)
    rank = index  # The rank of alpha_index among the roots in (low, high].
    if rank >= low_variations - high_variations:
        raise InvalidInput("The index of a RealAlgebraicRoot names one of the real roots of its polynomial")
    lower_bits = 0  # The lower root bound 2**-lower_bits once computed: 0 is no root here, as polynomial[0] != 0.
    while True:
        budget.tick()
        count = low_variations - high_variations  # Roots in (low, high].
        if count == 1 and not low_zero:
            if high_zero:  # The one root of (low, high] is high itself.
                return _lowest_dyadic(high, shift), _lowest_dyadic(high, shift)
            return _lowest_dyadic(low, shift), _lowest_dyadic(high, shift)
        if low < 0 < high:
            probe = 0
        else:
            near, far = (low, high) if low >= 0 else (-high, -low)
            probe = None
            if near == 0 and polynomial[0]:
                if not lower_bits:
                    budget.tick(len(polynomial))
                    lower_bits = (abs(polynomial[0]) + max(abs(c) for c in polynomial[1:])).bit_length()
                if far.bit_length() - shift > 3 - lower_bits:
                    if lower_bits > shift:
                        low, high, shift = low << lower_bits - shift, high << lower_bits - shift, lower_bits
                        near, far = (low, high) if low >= 0 else (-high, -low)
                    probe = 1 << (shift - lower_bits)
            elif near and far.bit_length() - near.bit_length() > 2:
                probe = 1 << ((near.bit_length() + far.bit_length()) >> 1)
            if probe is None:  # The midpoint, one bit finer.
                low, high, shift = low << 1, high << 1, shift + 1
                probe = (low + high) >> 1
            elif low < 0:
                probe = -probe
        variations, zero = kernels.endpoint(chain, probe, 1 << shift)
        left = low_variations - variations  # Roots in (low, probe].
        if rank < left:
            high, high_variations, high_zero = probe, variations, zero
        else:
            low, low_variations, low_zero = probe, variations, zero
            rank -= left


def _lowest_dyadic(numerator: int, shift: int) -> tuple[int, int]:
    """numerator/2**shift as (num, den) in lowest terms."""
    common = shift if not numerator else min(shift, (numerator & -numerator).bit_length() - 1)
    return numerator >> common, 1 << (shift - common)


def _root_digits(budget: Budget, polynomial: tuple[int, ...], interval: tuple[tuple[int, int], tuple[int, int]],
                 places: int) -> TruncatedDecimal:
    """The record of the root of the square-free polynomial in its isolating interval."""
    low, high = interval
    if low == high:  # A rational root, met at a midpoint of the isolation.
        return _rational_digits(budget, *low, places)
    if len(polynomial) == 2:  # c0 + c1 x with c1 > 0, primitive: the root -c0/c1 in lowest terms.
        return _rational_digits(budget, -polynomial[0], polynomial[1], places)
    scale = _ten_to(budget, places)
    floor, exact = _root_floor(budget, polynomial, low, high, scale)
    if floor > 0 or (floor == 0 and not exact):
        return TruncatedDecimal(1, floor, places, exact)
    if floor == 0:
        return TruncatedDecimal(0, 0, places, True)
    # alpha * 10**places lies in (floor, floor + 1) or is floor: |alpha| * 10**places in (-floor - 1, -floor) or -floor.
    return TruncatedDecimal(-1, -floor if exact else -floor - 1, places, exact)


def _evaluation(budget: Budget, length: int, point_bits: int, products: int) -> None:
    """Charge, before it runs, a Horner evaluation of a polynomial of that many coefficients at a point of that many
    bits (numerator and denominator), each step forming that many products of the running value, which grows to
    about length times the point's bits, by an integer of the point's bits (trusted.upoly.size_factor; the scale of
    trusted.upoly.evaluation_factor, which counts the two products of a rational point)."""
    from ...trusted.upoly import size_factor

    budget.tick(length * (1 if point_bits < 64 else size_factor(length * point_bits, point_bits, products)))


def _dyadic_value(budget: Budget, polynomial: tuple[int, ...], point: int, shift: int) -> int:
    """polynomial(point / 2**shift) * 2**(shift * degree), an integer with the sign of the polynomial at the point:
    Horner on the homogenized form, the powers of 2**shift as shifts. Each step forms one product, value * point;
    the shifted coefficient is added in linear time, so the evaluation is charged one product per step."""
    _evaluation(budget, len(polynomial), max(point.bit_length(), shift + 1), 1)
    degree = len(polynomial) - 1
    value = polynomial[degree]
    for i in range(degree - 1, -1, -1):
        value = value * point + (polynomial[i] << (shift * (degree - i)))
    return value


def _rational_sign(budget: Budget, polynomial: tuple[int, ...], numerator: int, denominator: int) -> int:
    """The sign of the polynomial at numerator/denominator (denominator >= 1): Horner on the homogenized form, two
    products of long integers per step (value * numerator and power * denominator)."""
    _evaluation(budget, len(polynomial), max(numerator.bit_length(), denominator.bit_length()), 2)
    value, power = 0, 1
    for coefficient in reversed(polynomial):
        value = value * numerator + coefficient * power
        power *= denominator
    return (value > 0) - (value < 0)


def _root_floor(budget: Budget, polynomial: tuple[int, ...], low_end: tuple[int, int], high_end: tuple[int, int],
                scale: int) -> tuple[int, bool]:
    """(floor(alpha * scale), whether alpha * scale is an integer) for the one root alpha of the square-free integer
    polynomial in the open interval between low_end and high_end, (numerator, denominator) pairs that are no roots.

    right is the sign of the polynomial on the part of the interval above alpha; it has the sign -right below alpha,
    since the simple root alpha is its only root there: side(t) = sign(polynomial(t)) * right is -1, 0 or 1 as t lies
    below, at or above alpha, for t in the interval. The bracket (low/2**shift, high/2**shift) keeps side -1 at low
    and 1 at high; a probe at side 0 is alpha itself, a dyadic rational."""
    right = _rational_sign(budget, polynomial, *high_end)
    if not right:  # _isolate returns a root met by a probe as a point, never as an end.
        raise ArithmeticError("An end of an isolating interval is a root")
    bracket = _dyadic_bracket(budget, polynomial, low_end, high_end, right)
    if len(bracket) == 3:
        bracket = _magnitude_bracket(budget, polynomial, *bracket, right)
    if len(bracket) == 2:  # A dyadic root: (point, shift).
        return _dyadic_floor(budget, *bracket, scale)
    low, high, shift = bracket
    derivative = tuple(i * coefficient for i, coefficient in enumerate(polynomial))[1:]
    final = scale.bit_length() + 2  # A bracket narrower than 2**(1 - final) is narrower than 1/scale.
    guard = 8
    while (high - low) * scale >= 1 << shift:
        budget.tick()
        precision = shift - (high - low).bit_length()  # The bracket is narrower than 2**-precision.
        # Newton's method squares the relative error: for |alpha| near 2**size, an error of 2**-precision becomes
        # about 2**(-2 * precision - size) (x = 2**size * y scales the root to 1). An absolute test would make a
        # root of 10,000 bits bisect 10,000 times before Newton's method starts; roots below 1 keep the absolute
        # test, which is the more cautious one for them.
        size = max(0, max(-low, high).bit_length() - shift)
        if precision + size > guard + 2:  # A Newton step whose bracket is narrower than half of this one.
            radius = _radius(guard - 2 * precision - size, final, guard)  # log2 of the half-width to confirm.
            target = max(shift + 1, 2 - radius)  # Its precision: the half-width is 2**(radius + target) >= 4 units.
            middle = (low + high) << (target - shift - 1)
            value = _dyadic_value(budget, polynomial, middle, target)
            if not value:
                return _dyadic_floor(budget, middle, target, scale)
            slope = _dyadic_value(budget, derivative, middle, target)  # polynomial' * 2**(target * (degree - 1)).
            if slope:
                if slope < 0:
                    value, slope = -value, -slope
                _charge(budget, value.bit_length(), slope.bit_length())
                candidate = middle - (2 * value + slope) // (2 * slope)  # x - p(x)/p'(x), to the nearest unit.
                half = 1 << (radius + target)
                old_low, old_high = low << (target - shift), high << (target - shift)
                first, last = max(candidate - half, old_low), min(candidate + half, old_high)
                if first < last:
                    first_side = -1 if first == old_low else right * _sign(_dyadic_value(budget, polynomial, first,
                                                                                          target))
                    if not first_side:
                        return _dyadic_floor(budget, first, target, scale)
                    if first_side < 0:
                        last_side = 1 if last == old_high else right * _sign(_dyadic_value(budget, polynomial, last,
                                                                                            target))
                        if not last_side:
                            return _dyadic_floor(budget, last, target, scale)
                        if last_side > 0:
                            low, high, shift = first, last, target
                            continue
            guard = min(guard + 8, 4096)  # Newton's step missed: bisect, and ask for more precision next time.
        low, high, shift = low << 1, high << 1, shift + 1
        middle = (low + high) >> 1
        side = right * _sign(_dyadic_value(budget, polynomial, middle, shift))
        if not side:
            return _dyadic_floor(budget, middle, shift, scale)
        if side < 0:
            low = middle
        else:
            high = middle
    # The bracket is narrower than 1/scale: at most one decimal point m/scale lies inside it.
    _charge(budget, low.bit_length(), scale.bit_length(), 2)
    point = ((low * scale) >> shift) + 1  # The least m with m/scale above low/2**shift.
    if point << shift >= high * scale:  # No decimal point inside: alpha lies in the open cell of point - 1.
        return point - 1, False
    side = right * _rational_sign(budget, polynomial, point, scale)
    if not side:
        return point, True
    return (point, False) if side < 0 else (point - 1, False)


def _dyadic_bracket(budget: Budget, polynomial: tuple[int, ...], low_end: tuple[int, int],
                    high_end: tuple[int, int], right: int) -> tuple[int, ...]:
    """(low, high, shift) with low_end <= low/2**shift < alpha < high/2**shift <= high_end, or (point, shift) when a
    probe met alpha itself. _isolate probes dyadic points, so its ends are dyadic and the first pass returns them as
    they are; other ends are rounded inward, finer each pass, until the signs confirm the bracket."""
    (low_numerator, low_denominator), (high_numerator, high_denominator) = low_end, high_end
    shift = max(low_denominator.bit_length(), high_denominator.bit_length())
    while True:
        budget.tick()
        low = -(-(low_numerator << shift) // low_denominator)  # The least dyadic point at or above low_end.
        high = (high_numerator << shift) // high_denominator  # The greatest at or below high_end.
        if low < high:
            low_side = -1 if low * low_denominator == low_numerator << shift else right * _sign(
                _dyadic_value(budget, polynomial, low, shift))
            if not low_side:
                return low, shift
            high_side = 1 if high * high_denominator == high_numerator << shift else right * _sign(
                _dyadic_value(budget, polynomial, high, shift))
            if not high_side:
                return high, shift
            if low_side < 0 < high_side:
                return low, high, shift
        shift += 8


def _magnitude_bracket(budget: Budget, polynomial: tuple[int, ...], low: int, high: int, shift: int,
                       right: int) -> tuple[int, ...]:
    """The bracket (low, high, shift) of alpha, narrowed until its ends differ by at most a factor of 8 in magnitude,
    or (point, shift) when a probe meets alpha itself.

    _isolate starts from the root bound, 2 + max |c_i / lc|, which can exceed the root by far: the root
    2**10000 of x**2 - 2**20000 - 1 has the isolating interval (0, 2**20001). Newton's method from such a bracket
    gains one bit per step, and bisection one bit per probe: some 20,000 evaluations of numbers of 20,000 bits. Here
    the bracket is first put on one side of 0 (by the sign at 0), then the exponent is bisected: the probe
    +-2**e/2**shift, e the mean of the bit lengths of the ends' magnitudes, lies strictly between them once their
    lengths differ by more than 2, and each probe halves that difference; a bracket that ends at 0 starts from the
    lower root bound |alpha| > |c_0| / (|c_0| + max_(i >= 1) |c_i|) > 2**-bits(|c_0| + max |c_i|) (Cauchy's bound on
    the reciprocal polynomial; 0 is no root here, as a root on an end of the bracket would have been met already).
    Every new end is confirmed by the exact sign of the polynomial there, as in _dyadic_bracket; a sign that does
    not confirm it leaves the bracket as it is."""

    def side(point: int) -> int:
        return right * _sign(_dyadic_value(budget, polynomial, point, shift))

    if low < 0 < high:
        middle_side = side(0)
        if not middle_side:
            return 0, shift
        if middle_side < 0:
            low = 0
        else:
            high = 0
    negative = high <= 0
    near, far = (-high, -low) if negative else (low, high)  # The magnitudes of the ends: 0 <= near < far.
    if near == 0 and far > 8 << shift:  # A root above 1 needs no lower bound and no finer shift: try the point 1.
        found = side(-(1 << shift) if negative else 1 << shift)
        if not found:
            return -(1 << shift) if negative else 1 << shift, shift
        if (found > 0) == negative:
            near = 1 << shift
        else:
            far = 1 << shift
    if near == 0:
        if not polynomial[0]:
            return low, high, shift
        budget.tick(len(polynomial))
        bound_bits = (abs(polynomial[0]) + max(abs(c) for c in polynomial[1:])).bit_length()
        if far.bit_length() - shift <= 3 - bound_bits:  # Already within a factor 8 of the bound.
            return low, high, shift
        if bound_bits > shift:
            far <<= bound_bits - shift
            shift = bound_bits
        candidate = 1 << (shift - bound_bits)
        found = side(-candidate if negative else candidate)
        if not found:
            return -candidate if negative else candidate, shift
        if found != (1 if negative else -1):  # Not below |alpha|: the bound did not hold, leave the bracket.
            return (-far, 0, shift) if negative else (0, far, shift)
        near = candidate
    while far.bit_length() - near.bit_length() > 2:
        budget.tick()
        probe = 1 << ((near.bit_length() + far.bit_length()) >> 1)
        found = side(-probe if negative else probe)
        if not found:
            return -probe if negative else probe, shift
        if (found > 0) == negative:  # The probe lies below alpha in magnitude.
            near = probe
        else:
            far = probe
    # The same two rationals at the least shift that holds them exactly: the lower root bound may have asked for a
    # shift of 60,000 bits that a root of 30,000 bits does not need, and every later evaluation pays for it.
    common = min(shift, (near & -near).bit_length() - 1, (far & -far).bit_length() - 1)  # near > 0 here.
    if common > 0:
        near, far, shift = near >> common, far >> common, shift - common
    return (-far, -near, shift) if negative else (near, far, shift)


def _radius(allowed: int, final: int, guard: int) -> int:
    """The radius, log2 of the half-width, of the bracket that the next Newton step confirms: the most negative of
    the radii r_0 = -final, r_(j+1) = floor((r_j - guard - 4) / 2) that is at least allowed, the least radius that the
    step admits. A bracket of radius r has a precision of at least -r - 2, from which the next step admits the radius
    guard + 2r + 4 <= r_j for r = r_(j+1): each radius of the chain leads to the one before it, so the last step lands
    on -final instead of just short of it, which would cost one more step at full precision."""
    radius = -final
    while radius < allowed:
        following = (radius - guard - 4) >> 1
        if following <= radius:  # The chain has reached its fixed point near -guard - 4.
            return max(allowed, -final)
        radius = following
    return radius


def _dyadic_floor(budget: Budget, point: int, shift: int, scale: int) -> tuple[int, bool]:
    """(floor(t * scale), whether t * scale is an integer) for t = point/2**shift."""
    _charge(budget, point.bit_length(), scale.bit_length())
    scaled = point * scale
    floor = scaled >> shift
    return floor, floor << shift == scaled


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


# ---------------------------------------------------------------------------------------------------------------
# Expressions: validation and the checker's enclosure

def _parse(budget: Budget, root: object) -> list[Entry]:
    """The distinct nodes of a closed expression, children before parents, as entries ("rational", p, q),
    ("sqrt", p, q), ("add", children), ("mul", children) and ("pow", child, exponent), children being positions in the
    list; the root's entry is the last. InvalidInput for anything outside the scope of the module docstring and for a
    node inside itself; ResourceLimitError past max_nodes distinct nodes, max_depth levels or max_integer_bits."""
    from ...algebraic import RealAlgebraicRoot
    from ...model import Symbol

    entries: list[Entry] = []
    position: dict[int, int] = {}  # id(node) -> its entry. Every node stays reachable from the root: no id is reused.
    walking: set[int] = set()  # The ids on the path from the root to the node being walked.
    stack: list[tuple[object, int, tuple | None]] = [(root, 1, None)]
    max_depth = budget.limits.max_depth
    while stack:
        node, depth, pending = stack.pop()
        key = id(node)
        if pending is not None:  # Every child has its entry: the node's follows.
            walking.discard(key)
            kind, children, exponent = pending
            budget.tick(len(children))
            if kind is Pow:
                entries.append(("pow", position[id(children[0])], exponent))
            else:
                entries.append(("add" if kind is Add else "mul", tuple(position[id(child)] for child in children)))
            position[key] = len(entries) - 1
            continue
        if key in position:
            continue  # A shared node, evaluated once.
        if key in walking:
            raise InvalidInput("An expression of digits does not contain itself")
        if depth > max_depth:
            raise ResourceLimitError("Computation limit exceeded: expression depth")
        budget.nodes()
        budget.tick()
        kind = type(node)
        if kind is Rational:
            position[key] = len(entries)
            entries.append(("rational", *_fraction(budget, node)))
            continue
        if kind is Sqrt:
            radicand = getattr(node, "radicand", _MISSING)
            if type(radicand) is not Rational:
                raise InvalidInput("digits takes square roots of Rationals only")
            numerator, denominator = _fraction(budget, radicand)
            if numerator < 0:
                raise InvalidInput("digits takes square roots of Rationals >= 0 only")
            position[key] = len(entries)
            entries.append(("sqrt", numerator, denominator))
            continue
        exponent = None
        if kind is Add or kind is Mul:
            children = getattr(node, "terms" if kind is Add else "factors", _MISSING)
            if type(children) is not tuple or len(children) < 2:
                raise InvalidInput("A sum or a product of digits holds a tuple of at least two expressions")
        elif kind is Pow:
            exponent = getattr(node, "exponent", _MISSING)
            if type(exponent) is not int or exponent < 0:
                raise InvalidInput("digits takes powers with exact int exponents >= 0 only")
            budget.check_int(exponent)
            children = (getattr(node, "base", _MISSING),)
        elif kind is Symbol:
            raise InvalidInput("digits requires a closed number: the value holds a Symbol")
        elif kind is RealAlgebraicRoot:
            raise InvalidInput("digits takes a RealAlgebraicRoot as the whole value, never inside an expression")
        else:
            raise InvalidInput(_SCOPE_ERROR)
        walking.add(key)
        budget.tick(len(children))  # Every child pushed now is popped once, shared or not.
        stack.append((node, depth, (kind, children, exponent)))
        stack.extend((child, depth + 1, None) for child in reversed(children))
    return entries


def _bounded(low: int, high: int, limit: int) -> tuple[int, int]:
    """The ends, when both have at most limit bits; ResourceLimitError otherwise, as the checker refuses them."""
    if max(-low, low, -high, high).bit_length() > limit:
        raise ResourceLimitError("Computation limit exceeded: integer bits")
    return low, high


def _product(budget: Budget, left: tuple[int, int], right: tuple[int, int], bits: int,
             limit: int) -> tuple[int, int]:
    """The checker's enclosure of a product: the least and the greatest of the four products, rounded outward to the
    scale 2**bits."""
    (a, c), (g, h) = left, right
    left_size, right_size = max(-a, a, -c, c).bit_length(), max(-g, g, -h, h).bit_length()
    _charge(budget, left_size, right_size, 4)
    budget.tick((left_size + right_size) >> 9)  # The negations, comparisons and shifts, linear: a unit per 512 bits.
    products = (a * g, a * h, c * g, c * h)
    return _bounded(min(products) >> bits, -(-max(products) >> bits), limit)


def _square(budget: Budget, low: int, high: int, bits: int, limit: int) -> tuple[int, int]:
    """The checker's enclosure of a square: the image of [low, high] under t -> t*t, rounded outward."""
    size = max(-low, low, -high, high).bit_length()
    _charge(budget, size, size, 2)
    budget.tick(size >> 8)  # The negations, comparisons and shifts, linear: a unit per 256 bits of the ends.
    if low >= 0:
        small, large = low * low, high * high
    elif high <= 0:
        small, large = high * high, low * low
    else:
        small, large = 0, max(low * low, high * high)
    return _bounded(small >> bits, -(-large >> bits), limit)


def _enclosure(budget: Budget, entries: list[Entry], bits: int) -> tuple[int, int]:
    """(L, H) with L/2**bits <= x <= H/2**bits, the very enclosure of the checker's module docstring: Rationals and
    square roots rounded outward, sums of the ends, products left to right, powers by squaring from the high bit.
    ResourceLimitError when an end passes max_integer_bits + bits bits."""
    limit = budget.limits.max_integer_bits + bits
    one = 1 << bits
    ends: list[tuple[int, int]] = []
    for entry in entries:
        budget.tick()
        kind = entry[0]
        if kind == "rational":
            _, p, q = entry
            _charge(budget, p.bit_length() + bits, q.bit_length(), 2)
            shifted = p << bits
            low, high = shifted // q, -(-shifted // q)
        elif kind == "sqrt":
            _, p, q = entry
            _charge(budget, p.bit_length() + 2 * bits, q.bit_length())
            square = (p << 2 * bits) // q
            _charge(budget, square.bit_length(), square.bit_length())  # isqrt, charged like a product of its size.
            low = isqrt(square)
            high = low + 1
        elif kind == "add":
            low = high = 0
            for child in entry[1]:
                budget.tick()
                term_low, term_high = ends[child]
                low, high = low + term_low, high + term_high
        elif kind == "mul":
            children = entry[1]
            low, high = ends[children[0]]
            for child in children[1:]:
                low, high = _product(budget, (low, high), ends[child], bits, limit)
        else:
            _, child, exponent = entry
            if exponent == 0:
                low = high = one
            else:
                base = ends[child]
                low, high = base
                for digit in bin(exponent)[3:]:  # The bits after the leading one, from the high end.
                    budget.tick()
                    low, high = _square(budget, low, high, bits, limit)
                    if digit == "1":
                        low, high = _product(budget, (low, high), base, bits, limit)
        ends.append(_bounded(low, high, limit))
    return ends[-1]


def _cell(budget: Budget, low: int, high: int, bits: int, scale: int) -> tuple[int, int] | None:
    """(sign, magnitude) when the enclosure [low, high] / 2**bits lies strictly inside one open decimal cell on one
    side of 0, as the checker's ("bits", b) test asks; None otherwise."""
    if low > 0:
        sign, bottom, top = 1, low, high
    elif high < 0:
        sign, bottom, top = -1, -high, -low  # |x| lies in [-high, -low] / 2**bits.
    else:
        return None
    _charge(budget, top.bit_length(), scale.bit_length(), 2)
    bottom, top = bottom * scale, top * scale
    magnitude = bottom >> bits
    if magnitude << bits < bottom and top < (magnitude + 1) << bits:
        return sign, magnitude
    return None


def _expression_digits(budget: Budget, entries: list[Entry], places: int) -> tuple[TruncatedDecimal, tuple]:
    """The record and the evidence of an expression: the first enclosure inside an open cell, or the exact rational
    value when an enclosure narrower than a cell meets a decimal point (module docstring)."""
    scale = _ten_to(budget, places)
    limit = budget.limits.max_integer_bits
    bits = min(limit, -(-10 * places // 3) + _GUARD_BITS)
    decided: object = None  # The exact value once _exact_value has run: a Rational, _IRRATIONAL or _TOO_MANY.
    while True:
        budget.tick()
        low, high = _enclosure(budget, entries, bits)
        found = _cell(budget, low, high, bits, scale)
        if found is not None:
            return TruncatedDecimal(found[0], found[1], places, False), ("bits", bits)
        # Narrower than a cell, and meets a decimal point; or the last precision, where a rational value still has
        # its exact digits however wide its enclosure is.
        if decided is None and ((high - low) * scale < 1 << bits or bits >= limit):
            decided = _exact_value(budget, entries)
            if type(decided) is Rational:
                return _rational_digits(budget, decided.numerator, decided.denominator, places), ("rational", decided)
        if bits >= limit:
            if decided is _TOO_MANY:
                raise ResourceLimitError(f"Computation limit exceeded: the value meets a decimal point at "
                                         f"{limit} bits, and its more than {MAX_RADICALS} independent square roots "
                                         f"leave no exact evaluation")
            raise ResourceLimitError(f"Computation limit exceeded: {limit} bits do not separate the value from a "
                                     f"decimal point")
        bits = min(2 * bits, limit)


# ---------------------------------------------------------------------------------------------------------------
# Expressions: the exact decision in the multiquadratic algebra

def _square_root(budget: Budget, n: int) -> int | None:
    """The integer square root of n >= 0 when n is a perfect square; None otherwise."""
    _charge(budget, n.bit_length(), n.bit_length(), 2)
    root = isqrt(n)
    return root if root * root == n else None


def _strip(budget: Budget, number: int, factor: int) -> tuple[int, int]:
    """(rest, e) with number == factor**e * rest and factor no divisor of rest, for number >= 1 and factor >= 2: the
    powers factor**(2**j) divide out while they do, then each of them once more from the highest down where it
    divides (what remains then holds factor to a power below 2**j), a few divisions instead of one per factor."""
    powers = [factor]
    exponent = 0
    while True:
        _charge(budget, number.bit_length(), powers[-1].bit_length())
        quotient, remainder = divmod(number, powers[-1])
        if remainder:
            break
        number = quotient
        exponent += 1 << (len(powers) - 1)
        _charge(budget, powers[-1].bit_length(), powers[-1].bit_length())
        powers.append(powers[-1] * powers[-1])
    for j in range(len(powers) - 2, -1, -1):
        _charge(budget, number.bit_length(), powers[j].bit_length())
        quotient, remainder = divmod(number, powers[j])
        if not remainder:
            number = quotient
            exponent += 1 << j
    return number, exponent


def _coprime_base(budget: Budget, numbers: list[int]) -> list[int]:
    """The natural coprime base of integers > 1, by factor refinement: a number that shares g = gcd > 1 with a member
    replaces that member by g and the two parts left once every factor g is divided out of each, until the members
    are pairwise coprime. Every number in play is a gcd or an exact quotient of the inputs, so the base is the unique
    one that the checker reaches too, and each input is a product of powers of its members."""
    base: list[int] = []
    for number in numbers:
        pending = [number]
        while pending:
            budget.tick()
            m = pending.pop()
            if m == 1:
                continue
            for j, member in enumerate(base):
                g = _gcd(budget, m, member)
                if g > 1:
                    del base[j]
                    pending += [g, _strip(budget, member, g)[0], _strip(budget, m, g)[0]]
                    break
            else:
                base.append(m)
    return base


# The work of the algebra is charged in batches, one per operation, by the formulas of the checker's own sparse
# algebra (mathforge.certify.numeric.digits: _lengths, _charge_sums, _charge_products): the producer then never
# makes a ("rational", r) claim whose evaluation the checker cannot afford within the same limits.

def _gcd_extra(budget: Budget, left_bits: int, right_bits: int, count: int = 1) -> None:
    """Charge, before they run, what count gcds of integers of the given bit lengths cost beyond the unit of the
    operation that forms them: the long division of the longer by the shorter, (long words - short words + 1) *
    (short words + 1) word steps on charge_gcd's scale of 64 per unit, and Lehmer's algorithm on the shorter
    (trusted.upoly.charge_gcd); nothing for integers of a few words."""
    from ...trusted.upoly import charge_gcd

    long_words, short_words = max(left_bits, right_bits) >> 6, min(left_bits, right_bits) >> 6
    extra = count * ((long_words - short_words + 1) * (short_words + 1) >> 6)
    if extra:
        budget.tick(extra)
    charge_gcd(min(left_bits, right_bits), count)


def _lengths(budget: Budget, element: Element) -> tuple[int, int]:
    """(the longest numerator of the coordinates of an element, the length of the least common multiple of their
    denominators, at most max_integer_bits), in bits: every denominator of a sum or a product of the coordinates of
    two elements divides the product of their common multiples."""
    from ...trusted.upoly import charge_product

    budget.tick(len(element) + 1)
    limit = budget.limits.max_integer_bits
    numerator, common = 0, 1
    for c in element.values():
        numerator = max(numerator, c.numerator.bit_length())
        denominator = c.denominator
        if denominator != 1 and common.bit_length() <= limit:
            _gcd_extra(budget, common.bit_length(), denominator.bit_length())
            divisor = gcd(common, denominator)
            charge_product(common.bit_length(), denominator.bit_length(), 2)
            common = common // divisor * denominator
    return numerator, min(common.bit_length(), limit)


def _charge_sums(budget: Budget, count: int, numerator: int, denominator: int) -> None:
    """Charge, before they run, count sums of Rationals with integers of at most those lengths: three products, two
    gcds and two divisions each, by the lengths of the integers they take."""
    from ...trusted.upoly import charge_product

    total, common = numerator + denominator + 1, 2 * denominator
    budget.tick(count)  # The loop; each Rational operation charges its own unit as well.
    charge_product(numerator, denominator, 3 * count)
    _gcd_extra(budget, denominator, denominator, count)
    _gcd_extra(budget, total, common, count)
    charge_product(total, common, 2 * count)


def _charge_products(budget: Budget, count: int, left: tuple[int, int], right: tuple[int, int],
                     weight_bits: int) -> None:
    """Charge, before they run, count terms a * b * weight of a product of elements with the lengths left and right
    (_lengths) and a weight of weight_bits bits, and their sums into the product."""
    from ...trusted.upoly import charge_product

    (na, da), (nb, db) = left, right
    budget.tick(count)  # The loop; each Rational operation charges its own unit as well.
    charge_product(na, nb, count)
    charge_product(da, db, count)
    charge_product(na, db, 2 * count)
    charge_product(nb, da, 2 * count)
    _gcd_extra(budget, na, db, count)
    _gcd_extra(budget, nb, da, count)
    if weight_bits > 1:
        charge_product(na + nb, weight_bits, count)
        _gcd_extra(budget, weight_bits, da + db, count)
        charge_product(na + nb + weight_bits, da + db, 2 * count)
    limit = budget.limits.max_integer_bits
    _charge_sums(budget, count, min(na + nb + weight_bits + da + db + count.bit_length(), limit), min(da + db, limit))


def _accumulate(budget: Budget, total: Element, right: Element) -> None:
    """total += right in A, in place, coordinate by coordinate (the rule of trusted.surd.Field.add on the nonzero
    coordinates only: a zero coordinate adds nothing); each sum of two coordinates charged before it runs by the
    lengths of its own integers, so that a sum of many terms costs the sizes of the terms, not the size of the sum
    so far once per term."""
    budget.tick(len(right) + 1)
    for mask, b in right.items():
        a = total.get(mask)
        if a is None:
            total[mask] = b
            continue
        _charge_sums(budget, 1, max(a.numerator.bit_length(), b.numerator.bit_length()),
                     max(a.denominator.bit_length(), b.denominator.bit_length()))
        c = a + b
        if c.numerator:
            total[mask] = c
        else:
            del total[mask]


def _times(budget: Budget, left: Element, right: Element, weight: Callable[[int], int], weight_bits: int) -> Element:
    """The product of two elements of the algebra: e_s * e_u = weight(s & u) * e_(s ^ u), weight(m) the product of
    the generators in m (weight_bits bounds its length); every pair of coordinates charged before the products run."""
    pairs = len(left) * len(right)
    _charge_products(budget, pairs, _lengths(budget, left), _lengths(budget, right), weight_bits)
    product: Element = {}
    for s, a in left.items():
        for u, b in right.items():
            term = a * b
            common = s & u
            if common:
                term = term * weight(common)
            mask = s ^ u
            total = product.get(mask)
            product[mask] = term if total is None else total + term
    budget.tick(len(product))
    return {mask: c for mask, c in product.items() if c.numerator}


def _exact_value(budget: Budget, entries: list[Entry]) -> object:
    """The value of the expression as a Rational when it is rational; _IRRATIONAL when it is not, and _TOO_MANY when
    more than MAX_RADICALS coprime non-square radicands leave the question open.

    sqrt(p/q) = sqrt(n)/q with n = p*q. 0 and perfect squares give rationals; the other n refine into their natural
    coprime base, whose non-square members b_1, ..., b_t generate A = Q[Y_1, ..., Y_t]/(Y_j**2 - b_j), and
    sqrt(n) = c * Y_S with n = c**2 * prod_{j in S} b_j. The elements of A are sparse: {subset mask: coefficient}."""
    from ...trusted.upoly import charge_power, charge_product

    radicands: dict[int, int | None] = {}  # n -> its square root, None for a non-square; in order of appearance.
    for entry in entries:
        budget.tick()
        if entry[0] == "sqrt":
            _charge(budget, entry[1].bit_length(), entry[2].bit_length())
            n = entry[1] * entry[2]
            if n not in radicands:
                radicands[n] = _square_root(budget, n)
    base = _coprime_base(budget, [n for n, root in radicands.items() if root is None])
    member_roots = {member: _square_root(budget, member) for member in base}
    generators = [member for member in base if member_roots[member] is None]
    if len(generators) > MAX_RADICALS:
        return _TOO_MANY
    slot = {member: j for j, member in enumerate(generators)}
    weight_bits = sum(member.bit_length() for member in generators)  # At least the length of every weight.
    # The checker builds trusted.surd.Field over the generators (4 * 2**t units, its weights charged by product) and
    # checks the record at the end: charged here as well, with a margin, so that the producer's work bounds the
    # checker's for every ("rational", r) claim it makes.
    budget.tick(4 * (1 << len(generators)) + 64)
    charge_product(weight_bits, max((member.bit_length() for member in generators), default=0),
                   1 << len(generators))
    weights: dict[int, int] = {0: 1}

    def weight(mask: int) -> int:
        """The product of the generators in the subset mask."""
        known = weights.get(mask)
        if known is None:
            low = mask & -mask
            rest = weight(mask ^ low)
            generator = generators[low.bit_length() - 1]
            _charge(budget, rest.bit_length(), generator.bit_length())
            known = weights[mask] = rest * generator
        return known

    surds: dict[int, tuple[int, int]] = {}  # n -> (c, S) with sqrt(n) = c * Y_S.
    for n, root in radicands.items():
        if root is not None:
            continue
        rest, coefficient, mask = n, 1, 0
        for member in base:
            rest, exponent = _strip(budget, rest, member)
            if exponent:
                charge_power(member.bit_length(), exponent >> 1)
                coefficient *= member ** (exponent >> 1)
                if exponent & 1:
                    if member_roots[member] is None:
                        mask |= 1 << slot[member]
                    else:
                        coefficient *= member_roots[member]
        if rest != 1:  # Every input is a product of powers of its coprime base.
            raise ArithmeticError("A radicand is no product of its coprime base")
        surds[n] = (coefficient, mask)
    values: list[Element] = []
    for entry in entries:
        budget.tick()
        kind = entry[0]
        if kind == "rational":
            value: Element = {0: Rational(entry[1], entry[2])} if entry[1] else {}
        elif kind == "sqrt":
            n = entry[1] * entry[2]
            root = radicands[n]
            if root is None:
                coefficient, mask = surds[n]
                value = {mask: Rational(coefficient, entry[2])}
            else:
                value = {0: Rational(root, entry[2])} if root else {}
        elif kind == "add":
            value = dict(values[entry[1][0]])
            budget.tick(len(value))
            for child in entry[1][1:]:
                _accumulate(budget, value, values[child])
        elif kind == "mul":
            value = values[entry[1][0]]
            for child in entry[1][1:]:
                value = _times(budget, value, values[child], weight, weight_bits)
        else:
            _, child, exponent = entry
            value = {0: Rational(1)}
            if exponent:
                base = value = values[child]
                for digit in bin(exponent)[3:]:  # From the high bit, as the checker evaluates powers.
                    budget.tick()
                    value = _times(budget, value, value, weight, weight_bits)
                    if digit == "1":
                        value = _times(budget, value, base, weight, weight_bits)
        values.append(value)
    result = values[-1]
    if any(result_mask for result_mask in result):
        return _IRRATIONAL
    constant = result.get(0, Rational(0))
    budget.check_int(constant.numerator)  # The evidence holds it; the checker refuses a longer one.
    budget.check_int(constant.denominator)
    return constant


# ---------------------------------------------------------------------------------------------------------------
# Rendering

def _padded(count: int, digits: str) -> Iterator[str]:
    """count zeros in runs of at most 4,096, then the digits: renderer.join consumes the runs lazily and stops at its
    character limit, so places far beyond any text that the render limits admit build no long string."""
    while count > 0:
        run = min(count, 4096)
        yield "0" * run
        count -= run
    yield digits


def _render(decimal: TruncatedDecimal, renderer: Any, child: Callable[..., str]) -> str:
    r"""The digits with the decimal point before the last places of them, a leading - for a negative number and ...
    after a truncated one: 1.41421356..., 0.250 (exact), -0.12..., 0.0000 (exact), 2... at places 0; in LaTeX \ldots
    instead of ... ."""
    digits = renderer.integer(decimal.magnitude)
    places = decimal.places
    if places:
        missing = places + 1 - len(digits)  # Leading zeros, so that one digit stands before the point.
        if missing > 0:
            digits = renderer.join(_padded(missing, digits))
        digits = renderer.join((digits[:-places], ".", digits[-places:]))
    tail = "" if decimal.exact else r"\ldots" if renderer.latex else "..."
    return renderer.join(("-" if decimal.sign < 0 else "", digits, tail))


RENDER = {TruncatedDecimal: _render}
