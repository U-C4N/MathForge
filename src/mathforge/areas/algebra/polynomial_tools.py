"""Polynomial tools over the exact rationals (mathforge.algebra): interpolate, the polynomial of least degree through
points with distinct rational abscissas.

interpolate(xs, ys, variable) -> Polynomial
    Domain: xs and ys hold exact rationals, ints or Rationals (MCP kind "rational_list": operations_v6.run freezes a
    list into a tuple and already rejects a bool, a float, a str or None in it as invalid input), with the same
    length n >= 1; the abscissas xs are pairwise distinct as numbers (2 and Rational(2) are one abscissa). variable
    is a Symbol (kind "symbol"). Different lengths, no point and a repeated abscissa are invalid input.
    Value: the unique polynomial P of degree < n with P(x_i) = y_i for every i, the zero polynomial when every
    ordinate is 0, as the v0.7 record mathforge.polynomial.Polynomial: canonical Rationals, constant term first,
    without trailing zeros ((Rational(0),) for zero), in the request's variable itself. A polynomial of degree < n
    through the points is unique (two of them differ by a polynomial of degree < n with n roots), so it is the one
    of least degree through them. The record bounds the degree by MAX_POLYNOMIAL_DEGREE = 1024.
    Evidence: (). The checker, mathforge.certify.algebra.polynomial_tools, evaluates the claimed polynomial exactly at
    every node, after a cheap test of the same identities modulo the prime 2**61 - 1 that refuses most wrong claims.

The family defines no record class: its only value is a Polynomial, which the shared renderer presents (text
"x^2 + x + 1" and "2/3*x - 1/3", LaTeX "x^{2} + x + 1" and "\\frac{2}{3}\\, x - \\frac{1}{3}"), so TYPES is empty and
there is no RENDER.

How produce() computes: Newton's divided differences (docs/v09/cards/m2-algebra-equations-sets-numeric.md), on integers
only, with no Fraction and no gcd per step of the table. Nothing here is a proof: the checker evaluates the claim at
every node with an algorithm of its own.

- Validation first: an exact int, or a Rational of exact ints in lowest terms with a positive denominator (one forged
  with object.__new__ may hold anything), an integer over max_integer_bits being a ResourceLimitError; a Symbol whose
  fields have their exact types; then the card's cases. A list is accepted as well as a tuple, as the kind admits it.
- Integer nodes. With Q the least common multiple of the denominators of the abscissas, the nodes t_i = Q x_i are
  distinct integers, and P(x) = R(Q x) for the polynomial R of degree < n with R(t_i) = y_i. The nodes are sorted:
  divided differences are symmetric in their nodes, so the order changes no value, and equally spaced nodes show as
  equal steps.
- Divided differences, one level at a time, each level over one common denominator: level j holds the divided
  differences R[t_m, ..., t_(m+j)] = N_j[m] / D_j for m = 0, ..., n - 1 - j, where D_0 = V, the least common multiple
  of the denominators of the ordinates, and N_0[m] = V y_m. Since
      R[t_m, ..., t_(m+j)] = (N_(j-1)[m+1] - N_(j-1)[m]) / (D_(j-1) * (t_(m+j) - t_m)),
  level j takes D_j = D_(j-1) * s_j and N_j[m] = (N_(j-1)[m+1] - N_(j-1)[m]) * s_j / (t_(m+j) - t_m), with an s_j >= 1
  that makes every N_j[m] an integer. For equally spaced nodes t_m = t_0 + m*h every node difference of the level is
  j*h, and s_j = j*h leaves N_j the plain differences of N_(j-1), the forward differences, with no division at all.
  Otherwise s_j is the least that works: the least common multiple, over the differences that their node difference
  does not divide, of |t_(m+j) - t_m| / gcd(t_(m+j) - t_m, remainder), so that the division is exact, and s_j = 1
  wherever the divided differences are integers, as all of them are for a polynomial with integer coefficients at
  integer nodes (the divided differences of t**k are the complete homogeneous symmetric polynomials of the nodes).
- Rational abscissas over themselves. Where Q grows with many different denominators, the integer nodes carry it into
  every level (the divided differences of R are those of P divided by Q**j), so when the abscissas are not equally
  spaced and Q is longer than twice their denominators on average, the table runs over the abscissas x_m = a_m / b_m
  instead, with the same common denominators: x_(m+j) - x_m = delta_m / (b_m b_(m+j)) with the short integer
  delta_m = a_(m+j) b_m - a_m b_(m+j) > 0, so level j multiplies the differences by b_m b_(m+j) and divides them by
  delta_m where it divided them by t_(m+j) - t_m, s_j chosen as above. Horner's rule then expands Newton's form in x
  directly, each factor x - a_k/b_k as (b_k x - a_k) / b_k: the running denominator takes the factor b_k before step
  k, and the row is multiplied by b_k x - a_k, so that P = A_0 / L_0 without Q. Equally spaced abscissas and those of
  one common denominator keep the integer nodes, where no division or a short one remains.
- A level of zeros ends the table. R agrees with the data at every node, so the data's divided differences are R's,
  and those of order j vanish exactly when deg R < j: the degree d is the last level that is not all zero, and n
  points on a polynomial of low degree cost about n*(d + 2) steps rather than n**2/2. A degree above
  MAX_POLYNOMIAL_DEGREE, which only more than 1,025 points can give, has no Polynomial record: UnsupportedOperation,
  as the record's constructor raises for it.
- Newton's coefficients c_k = N_k[0] / D_k (D_k = s_0 * ... * s_k with s_0 = D_0), each reduced to lowest terms p_k /
  q_k: by one long division when it leaves no remainder (an integer), by a gcd otherwise.
- Newton's form R = sum_k c_k * prod_(m < k) (t - t_m) is expanded by Horner's rule over the running least common
  multiple L_k = lcm(q_k, ..., q_d): A_d = p_d, and for k = d - 1, ..., 0
      A_k = (L_k / L_(k+1)) * A_(k+1) * (t - t_k) + p_k * (L_k / q_k),
  which keeps A_k = L_k * T_k for the tail T_k = c_k + (t - t_k) * T_(k+1) of Newton's form, so R = A_0 / L_0 with
  integer coefficients A = A_0. Where Newton's coefficients are integers, as for a polynomial with integer
  coefficients at integer nodes, L stays 1 and every row of Horner's rule is a tail itself, no longer than it must
  be: over the single denominator D_d the rows of 1,025 such points grew by up to log2(1024!) = 8,769 bits.
- The coefficient of x**i in P is A_i * Q**i / L_0, reduced to lowest terms with one gcd each (none when L_0 = 1),
  of A_i and the part of L_0 that is coprime to Q**i / gcd(Q**i, L_0), which builds up with a short gcd per
  coefficient (_coefficients states why that is lowest terms).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one per number, so that every valid input costs
at least three, and charges every step before it runs: a pass over a level of the table or a row of Horner's rule at
the total length of its numbers, which a cheap pass charged with it measures after each step, and any other operation
at the lengths of its operands. The cost model is this family's own, on the scale of trusted.integers (a unit per
step, products and long divisions by pairs of 64-bit words, a gcd quadratic in its shorter operand as
trusted.upoly.charge_gcd charges it), with the linear terms and the products and divisions of a long by a short
integer measured, since they decide the cost of a table: measured with CPython on one machine, a unit takes at most
about 0.25 microseconds at every length up to 65,536 bits, mostly 0.15 to 0.2. These requests fit the default max_work
of 2,000,000, and the checker verifies their values in a budget of its own: 1,025 consecutive integer abscissas on a
polynomial of degree 1,024 with 20-bit integer coefficients, the record's full size, about 1.41 million units (0.28 s;
the checker about 1.6 million), 600 consecutive integer abscissas with one-digit ordinates (coefficients of about 4,600
bits) about 360,000 (the checker about 616,000), 2,000 points on a cubic about 14,000, and 300 random integer
abscissas in [-3000, 3000] with one-digit ordinates (coefficients of about 13,800 bits) about 1.04 million (the checker
about 540,000). At 1,025 consecutive integer abscissas with one-digit ordinates (coefficients of about 8,800 bits) this
producer needs about 1.52 million units and the checker more than its 2 million, so run() reports
resource_limit_exceeded. Unequally spaced nodes divide every entry of the table, about twice the work of the checker's
evaluation: 700 random integer abscissas in [-3000, 3000] on a polynomial of degree 699 with one-digit coefficients
take about 1.66 million units (the checker about 750,000), and 800 run out of the default budget here, where the
checker would still verify the claim with about 980,000.

Rational abscissas with many different denominators run the table over themselves, which leaves Q out of every
level: 200 abscissas k/p_k with distinct 20-bit primes p_k, a value whose coefficients have under 8,000 bits, take about
311,000 to 437,000 units (the checker about 577,000; over the integer nodes they took 300 to 650 million), 50 random
abscissas a/b with |a|, b < 2**16 about 400,000 (the checker about 180,000) and 40 with 30-bit ones about 630,000
(the checker about 285,000), at 0.11 to 0.14 microseconds a unit. The remaining gap is the cost of long coefficients
over long common denominators: the table's per-level common denominator holds every delta_m that a level divides by,
and Newton's coefficients and the final coefficients are reduced by gcds quadratic in their length. 75 random abscissas
with 16-bit denominators take about 2.4 million units here (7.3 million over the integer nodes) for a value of
52,000-bit coefficients that the checker verifies with about 1.2 million, so run() still reports
resource_limit_exceeded for it, about a third of the time in reducing the coefficients to lowest terms; such requests
fit the default budget up to about 70 abscissas with 16-bit denominators (65: about 1.2 million units) or 50 with
30-bit ones (about 1.8 million).
"""

from __future__ import annotations

from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, computation
from ...model import Assumption, Domain, Rational, Symbol, Truth, trusted_rational
from ...polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
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

# The scope text of every spec of wave M2 (docs/v09/cards/m2-algebra-equations-sets-numeric.md).
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
         "and decimal digits")

# xs and ys are unannotated, as v0.7's rational_list arguments are (mean(data), covariance(x, y)): a list is
# accepted as well as a tuple.
SPECS = (
    OperationSpec("interpolate", "algebra",
                  (Argument("xs", "", None, "rational_list"), Argument("ys", "", None, "rational_list"),
                   Argument("variable", "Symbol", None, "symbol")),
                  "Polynomial", "Find the polynomial of least degree through given points with distinct rational "
                  "abscissas.", SCOPE, version=6),
)

TYPES: dict[str, type] = {}  # The value is the v0.7 record mathforge.polynomial.Polynomial.
CHECKER = "mathforge.certify.algebra.polynomial_tools"

_ARGUMENTS = frozenset(argument.name for argument in SPECS[0].arguments)
_MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
_ZERO = trusted_rational(0, 1)
_DIGIT_BITS = 30  # CPython divides by an integer below 2**30, one digit, about four times faster per word.

Pair = tuple[int, int]  # (numerator, denominator) of a validated exact rational: lowest terms, denominator >= 1.


def produce(operation: str, args: dict[str, object]) -> tuple[Polynomial, tuple[()]]:
    """(value, evidence) of interpolate for arguments that operations_v6.run has checked by name and kind; the module
    states the algorithm. InvalidInput for an argument outside the card's domain, UnsupportedOperation for another
    operation and for a degree above MAX_POLYNOMIAL_DEGREE, ResourceLimitError past the budget."""
    if type(operation) is not str or operation != "interpolate":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the polynomial_tools family")
    if type(args) is not dict or set(args) != _ARGUMENTS:
        raise InvalidInput("interpolate takes exactly the arguments xs, ys and variable")
    xs, ys, variable = args["xs"], args["ys"], args["variable"]
    if not _plain_symbol(variable):
        raise InvalidInput("interpolate requires a Symbol variable")
    for name, numbers in (("xs", xs), ("ys", ys)):
        if type(numbers) is not tuple and type(numbers) is not list:
            raise InvalidInput(f"interpolate requires {name} as a tuple or list of exact rationals")
    with computation() as budget:
        budget.tick()  # The call itself.
        nodes, values = _pairs(budget, xs, "xs"), _pairs(budget, ys, "ys")
        if len(nodes) != len(values):
            raise InvalidInput("interpolate requires xs and ys of the same length: one ordinate per abscissa")
        if not nodes:
            raise InvalidInput("interpolate requires at least one point")
        _require_distinct(budget, nodes)
        return Polynomial(_interpolate(budget, nodes, values), variable), ()


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _plain_symbol(variable: object) -> bool:
    """Whether variable is a Symbol whose fields have their exact types. A Symbol forged with object.__new__ may hold
    anything; the checker refuses such a variable, so produce() refuses it as input."""
    if type(variable) is not Symbol:
        return False
    name, domain = getattr(variable, "name", None), getattr(variable, "domain", None)
    assumptions, uid = getattr(variable, "assumptions", None), getattr(variable, "uid", None)
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > _MAX_ASSUMPTIONS):
        return False
    return all(type(item) is Assumption and type(getattr(item, "name", None)) is str
               and type(getattr(item, "truth", None)) is Truth for item in assumptions)


def _pairs(budget: Budget, numbers: tuple[object, ...] | list[object], name: str) -> list[Pair]:
    """The exact rationals of xs or ys as (numerator, denominator) pairs. InvalidInput for anything but an int or a
    canonical Rational, ResourceLimitError for an integer over max_integer_bits."""
    pairs = []
    for number in numbers:
        budget.tick()
        kind = type(number)
        if kind is int:  # Neither a bool nor a float nor a str is an exact number.
            budget.check_int(number)
            pairs.append((number, 1))
            continue
        if kind is not Rational:
            raise InvalidInput(f"interpolate requires exact rationals in {name}: ints or Rationals")
        numerator = getattr(number, "numerator", None)  # A forged Rational may lack a slot.
        denominator = getattr(number, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput(f"interpolate requires canonical Rationals in {name}")
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator != 1:
            budget.tick(_gcd_work(numerator.bit_length(), denominator.bit_length()))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput(f"interpolate requires canonical Rationals in {name}: lowest terms")
        pairs.append((numerator, denominator))
    return pairs


def _require_distinct(budget: Budget, nodes: list[Pair]) -> None:
    """InvalidInput unless the abscissas are pairwise distinct: lowest terms make equal numbers equal pairs."""
    seen: set[Pair] = set()
    for node in nodes:
        budget.tick(1 + ((node[0].bit_length() + node[1].bit_length()) >> 14))  # Hashing reads both integers.
        if node in seen:
            raise InvalidInput("interpolate requires pairwise distinct abscissas xs")
        seen.add(node)


# ---------------------------------------------------------------------------------------------------------------
# Costs, measured, on the scale of trusted.integers

def _sums_work(count: int, total: int) -> int:
    """The units of a pass of count additions or subtractions of neighbours in a list whose integers hold total bits
    in all: about 0.03 microseconds each, and 0.022 more per 1,000 bits of the longer operand."""
    return 1 + (count >> 3) + (total >> 13)


def _products_work(count: int, total: int, factor_bits: int) -> int:
    """The units of a pass of count products, each with an addition, of the integers of a list, which hold total bits
    in all, by integers of at most factor_bits bits: about 0.07 microseconds each, 0.05 more per 1,000 bits of the
    long operand, and the product itself, schoolbook below CPython's Karatsuba cutoff and charged so above it too."""
    return 1 + (count >> 2) + (total >> 12) + (count * factor_bits >> 12) + (total * factor_bits >> 18)


def _product_work(left_bits: int, right_bits: int) -> int:
    """The units of one product of integers of these lengths, with an addition."""
    return 1 + ((left_bits + right_bits) >> 12) + (left_bits * right_bits >> 18)


def _divisions_work(count: int, total: int, divisor_bits: int) -> int:
    """The units of a pass of count long divisions of integers that hold total bits in all by integers of at most
    divisor_bits bits, by the length of each quotient (at most its dividend's and 64 bits): a pass over the dividend,
    and per quotient word a pass over the divisor, faster for a divisor of one digit."""
    quotients = total + 64 * count
    per_word = 11 if divisor_bits <= _DIGIT_BITS else 9
    return 1 + (count >> 2) + (total >> 12) + (quotients >> per_word) + (quotients * divisor_bits >> 18)


def _division_work(dividend_bits: int, divisor_bits: int) -> int:
    """The units of one long division, by the length of its quotient: an exact quotient may be far shorter than the
    dividend."""
    quotient = max(dividend_bits - divisor_bits, 0) + 64
    per_word = 11 if divisor_bits <= _DIGIT_BITS else 9
    return 2 + (dividend_bits >> 12) + (quotient >> per_word) + (quotient * divisor_bits >> 18)


def _gcd_work(left_bits: int, right_bits: int) -> int:
    """The units of math.gcd of integers of these lengths: a long division of the longer by the shorter, then Lehmer's
    algorithm on the shorter, a pass per 64 bits and one unit per 64 squared 64-bit words, the quadratic scale of
    trusted.upoly.charge_gcd (about 4 microseconds at 1,000 bits, 64 at 9,000 and 2,400 at 65,536)."""
    shorter, longer = min(left_bits, right_bits), max(left_bits, right_bits)
    words = shorter >> 6
    return _division_work(longer, shorter) + words + (words * words >> 6)


def _sort_work(count: int, bits: int) -> int:
    """The units of sorting count integers of at most bits bits: count * log2(count) comparisons, each about 0.01
    microseconds for short integers and, for long ones that share a long leading part, a read of both up to 0.5
    microseconds at 60,000 bits; charged a sixteenth of a unit each plus a unit per 8,192 bits of their length, at
    least twice the time measured (sorting 100,000 integers took 2 to 18 milliseconds; a count * log2(count) charge,
    some 1.7 million units for them, kept 100,000 points on a line, which the checker verifies with 1.3 million, from
    the default budget)."""
    return 1 + (count * count.bit_length() * (1 + (bits >> 9)) >> 4)


def _total_bits(budget: Budget, numbers: list[int]) -> int:
    """The bit lengths of the numbers added up, after a charge for the pass."""
    budget.tick(1 + (len(numbers) >> 4))
    return sum(map(int.bit_length, numbers))


# ---------------------------------------------------------------------------------------------------------------
# Newton's divided differences on integers

def _interpolate(budget: Budget, nodes: list[Pair], values: list[Pair]) -> tuple[Rational, ...]:
    """The coefficients of the interpolating polynomial, constant first, as canonical Rationals without trailing zeros;
    the module states the algorithm."""
    scale, points = _integers(budget, nodes)  # Q and the integer nodes t_i = Q x_i.
    common, level = _integers(budget, values)  # D_0 = V and N_0 = V y.
    count = len(points)
    budget.tick(1 + (count >> 3))  # Whether every ordinate is 0, and the lengths of the nodes.
    if not any(level):
        return (_ZERO,)
    node_bits = max(map(int.bit_length, points))  # Every node difference has at most node_bits + 1 bits.
    budget.tick(_sort_work(count, node_bits))
    ranks = sorted(range(count), key=points.__getitem__)
    points, level = [points[rank] for rank in ranks], [level[rank] for rank in ranks]
    budget.tick(2 * _sums_work(count, 2 * count * node_bits))  # The steps, and whether they are all equal.
    steps = [upper - lower for lower, upper in zip(points, points[1:])]
    spacing = steps[0] if steps and steps.count(steps[0]) == len(steps) else None
    rational = None  # The sorted abscissas as pairs, for the table over the abscissas themselves.
    budget.tick(1 + (count >> 4))  # The lengths of the denominators.
    if (scale != 1 and spacing is None
            and scale.bit_length() * count > 2 * sum(denominator.bit_length() for _, denominator in nodes)):
        rational = [nodes[rank] for rank in ranks]
        node_bits = max(max(numerator.bit_length(), denominator.bit_length()) for numerator, denominator in rational)
    heads, factors = [level[0]], [common]  # N_k[0] and s_k for each level k, with s_0 = D_0.
    total = _total_bits(budget, level)
    for order in range(1, count):
        found = _next_level(budget, level, total, points, order, node_bits, spacing, rational)
        if found is None:
            break  # Every divided difference of this order is 0: the degree is order - 1.
        if order > MAX_POLYNOMIAL_DEGREE:
            raise UnsupportedOperation(f"interpolate: the interpolating polynomial has a degree above "
                                       f"{MAX_POLYNOMIAL_DEGREE}, which no Polynomial record holds")
        level, factor = found
        total = _total_bits(budget, level)
        heads.append(level[0])
        factors.append(factor)
    if rational is not None:
        numerators, denominator = _expand(budget, _newton(budget, heads, factors),
                                          [numerator for numerator, _ in rational],
                                          [denominator for _, denominator in rational])
        return _coefficients(budget, numerators, denominator, 1)
    numerators, denominator = _expand(budget, _newton(budget, heads, factors), points)
    return _coefficients(budget, numerators, denominator, scale)


def _integers(budget: Budget, pairs: list[Pair]) -> tuple[int, list[int]]:
    """(L, [p * (L / q) for each pair (p, q)]): L, the least common multiple of the denominators, and the numbers
    times L, all integers."""
    common = 1
    for _, denominator in pairs:
        budget.tick()
        if denominator == 1:
            continue
        common_bits, bits = common.bit_length(), denominator.bit_length()
        budget.tick(_division_work(common_bits, bits))
        remainder = common % denominator
        if remainder:  # L becomes L * (q / gcd(q, L)), and gcd(q, L) = gcd(q, L mod q).
            budget.tick(_gcd_work(bits, remainder.bit_length()))
            divisor = gcd(denominator, remainder)
            budget.tick(_division_work(bits, divisor.bit_length()))
            factor = denominator // divisor
            budget.tick(_product_work(common_bits, factor.bit_length()))
            common *= factor
    if common == 1:
        return 1, [numerator for numerator, _ in pairs]
    common_bits = common.bit_length()
    integers = []
    for numerator, denominator in pairs:
        bits = denominator.bit_length()
        budget.tick(_division_work(common_bits, bits) + _product_work(numerator.bit_length(), common_bits - bits + 1))
        integers.append(numerator * (common // denominator))
    return common, integers


def _next_level(budget: Budget, level: list[int], total: int, points: list[int], order: int, node_bits: int,
                spacing: int | None, rational: list[Pair] | None = None) -> tuple[list[int], int] | None:
    """(N_j, s_j) for the level j = order from the numerators of level j - 1, whose integers hold total bits in all
    (module docstring); None when every divided difference of order j is 0. spacing is the step h of equally spaced
    increasing nodes, else None. With rational, the increasing abscissas a_m / b_m as pairs, the table runs over them
    rather than over the integer nodes (module docstring), and node_bits bounds their integers."""
    count = len(level) - 1
    budget.tick(_sums_work(count, total) + (count >> 4))  # The differences, then whether they are all 0.
    differences = [upper - lower for lower, upper in zip(level, level[1:])]
    if not any(differences):
        return None
    if spacing is not None:  # Every node difference of the level is j*h: s_j = j*h, and no division at all.
        budget.tick(_product_work(order.bit_length(), spacing.bit_length()))
        return differences, order * spacing
    delta_bits = node_bits + 1  # Every node difference has at most this many bits.
    if rational is None:
        budget.tick(_sums_work(count, 2 * count * node_bits))
        deltas = [upper - lower for lower, upper in zip(points, points[order:])]  # t_(m+j) - t_m > 0.
    else:
        # x_(m+j) - x_m = delta_m / (b_m b_(m+j)) with delta_m = a_(m+j) b_m - a_m b_(m+j) > 0, so the divided
        # difference is (N[m+1] - N[m]) b_m b_(m+j) / (D_(j-1) delta_m): the differences take the factor b_m b_(m+j)
        # and are divided by delta_m, of at most 2 node_bits + 1 bits. Three products of short integers each.
        short = count * node_bits
        budget.tick(3 * _products_work(count, short, node_bits) + _sums_work(count, 2 * short))
        deltas = [upper * lower_scale - lower * upper_scale for (lower, lower_scale), (upper, upper_scale)
                  in zip(rational, rational[order:])]
        scales = [lower_scale * upper_scale for (_, lower_scale), (_, upper_scale) in zip(rational, rational[order:])]
        delta_bits = 2 * node_bits + 1
        budget.tick(_products_work(count, _total_bits(budget, differences), 2 * node_bits))
        differences = [difference * scale for difference, scale in zip(differences, scales)]
    dividends = _total_bits(budget, differences)
    budget.tick(_divisions_work(count, dividends, delta_bits) + (count >> 3))  # Then the remainders' list.
    divided = list(map(divmod, differences, deltas))
    inexact = [(remainder, delta) for (_, remainder), delta in zip(divided, deltas) if remainder]
    if not inexact:
        return [quotient for quotient, _ in divided], 1
    # For each quotient that is no integer, what s_j must hold to make it one: delta / gcd(delta, remainder), where
    # 0 < remainder < delta; then s_j is the least common multiple of these. A pass of gcds and divisions of numbers
    # of at most delta_bits bits: about 0.1 microseconds each for short ones, more as _gcd_work counts for long.
    inexact_count = len(inexact)
    budget.tick(1 + (inexact_count >> 1)
                + inexact_count * (_gcd_work(delta_bits, delta_bits) + _division_work(delta_bits, delta_bits) - 4))
    needs = {delta // gcd(delta, remainder) for remainder, delta in inexact}
    budget.tick(_sort_work(len(needs), delta_bits))  # Sorting them.
    factor = 1
    for needed in sorted(needs):  # In increasing order, so that the charges do not depend on a set's order.
        factor_bits, needed_bits = factor.bit_length(), needed.bit_length()
        budget.tick(_gcd_work(factor_bits, needed_bits))
        shared = gcd(factor, needed)
        if shared != needed:
            budget.tick(_division_work(needed_bits, shared.bit_length()) + _product_work(factor_bits, needed_bits))
            factor *= needed // shared
    # factor is a multiple of delta / gcd(delta, remainder), so delta divides remainder * factor exactly.
    factor_bits, short = factor.bit_length(), inexact_count * delta_bits
    budget.tick(_products_work(count, dividends, factor_bits) + _products_work(inexact_count, short, factor_bits)
                + _divisions_work(inexact_count, short + inexact_count * factor_bits, delta_bits))
    return [quotient * factor + remainder * factor // delta if remainder else quotient * factor
            for (quotient, remainder), delta in zip(divided, deltas)], factor


def _newton(budget: Budget, heads: list[int], factors: list[int]) -> list[Pair]:
    """Newton's coefficients heads[k] / D_k in lowest terms, with D_k = factors[0] * ... * factors[k]."""
    coefficients = []
    denominator = 1
    for head, factor in zip(heads, factors):
        budget.tick()
        if factor != 1:
            budget.tick(_product_work(denominator.bit_length(), factor.bit_length()))
            denominator *= factor
        coefficients.append(_lowest_terms(budget, head, denominator))
    return coefficients


def _lowest_terms(budget: Budget, numerator: int, denominator: int) -> Pair:
    """numerator / denominator, for a denominator >= 1, as a pair in lowest terms."""
    if denominator == 1 or not numerator:
        return numerator, 1
    numerator_bits, denominator_bits = numerator.bit_length(), denominator.bit_length()
    budget.tick(_division_work(numerator_bits, denominator_bits))
    quotient, remainder = divmod(numerator, denominator)
    if not remainder:
        return quotient, 1
    budget.tick(_gcd_work(denominator_bits, remainder.bit_length()))
    divisor = gcd(denominator, remainder)  # gcd(denominator, numerator), as remainder = numerator mod denominator.
    if divisor == 1:
        return numerator, denominator
    divisor_bits = divisor.bit_length()
    budget.tick(_division_work(numerator_bits, divisor_bits) + _division_work(denominator_bits, divisor_bits))
    return numerator // divisor, denominator // divisor


def _expand(budget: Budget, newton: list[Pair], points: list[int],
            scales: list[int] | None = None) -> tuple[list[int], int]:
    """(A, L_0): Newton's form sum_k (p_k / q_k) * prod_(m < k) (t - points[m]) for newton = [(p_k, q_k), ...] as
    A / L_0 with integer coefficients A, constant first, by Horner's rule over the running least common multiple of
    the denominators (module docstring). With scales, the nodes are the rationals points[m] / scales[m]: each factor
    t - a/b is (b t - a) / b, so the running denominator takes the factor b before the step."""
    degree = len(newton) - 1
    numerator, common = newton[degree]
    row = [numerator]
    total = numerator.bit_length()
    for k in range(degree - 1, -1, -1):
        numerator, denominator = newton[k]
        bits = denominator.bit_length()
        scale = 1 if scales is None else scales[k]
        if scale != 1:  # T_k = c_k + (b t - a) A_(k+1) / (L_(k+1) b): the row's denominator is L_(k+1) b.
            budget.tick(_product_work(common.bit_length(), scale.bit_length()))
            common *= scale
        if denominator != 1:
            common_bits = common.bit_length()
            budget.tick(_division_work(common_bits, bits))
            remainder = common % denominator
            if remainder:  # L_k = L_(k+1) * (q_k / gcd(q_k, L_(k+1))): every coefficient of the row takes that factor.
                budget.tick(_gcd_work(bits, remainder.bit_length()))
                divisor = gcd(denominator, remainder)
                budget.tick(_division_work(bits, divisor.bit_length()))
                factor = denominator // divisor
                factor_bits = factor.bit_length()
                budget.tick(_product_work(common_bits, factor_bits) + _products_work(len(row), total, factor_bits))
                common *= factor
                row = [coefficient * factor for coefficient in row]
                total += len(row) * factor_bits  # At least the length of each product.
        if denominator == common:  # The constant term p_k * (L_k / q_k).
            constant = numerator
        else:
            if denominator == 1:
                multiplier = common
            else:
                budget.tick(_division_work(common.bit_length(), bits))
                multiplier = common // denominator
            budget.tick(_product_work(numerator.bit_length(), multiplier.bit_length()))
            constant = numerator * multiplier
        node = points[k]
        node_bits = node.bit_length()
        # The row times t - node, a product and a difference per coefficient, then the new constant term.
        budget.tick(_products_work(len(row), total, node_bits) + _product_work(constant.bit_length(), node_bits))
        if scale != 1:  # The row times b t - a.
            budget.tick(_products_work(len(row), total, scale.bit_length()))
            row = ([constant - node * row[0]] + [scale * lower - node * upper for lower, upper in zip(row, row[1:])]
                   + [scale * row[-1]])
        elif node:
            row = ([constant - node * row[0]] + [lower - node * upper for lower, upper in zip(row, row[1:])]
                   + [row[-1]])
        else:
            row = [constant] + row
        total = _total_bits(budget, row)
    return row, common


def _coefficients(budget: Budget, numerators: list[int], denominator: int, scale: int) -> tuple[Rational, ...]:
    """The canonical Rationals numerators[i] * scale**i / denominator, constant first; ResourceLimitError for one
    whose numerator or denominator is longer than max_integer_bits.

    With g_i = gcd(scale**i, denominator), the integers lift_i = scale**i / g_i and rest_i = denominator / g_i are
    coprime, so numerators[i] * scale**i / denominator = numerators[i] * lift_i / rest_i is in lowest terms once
    numerators[i] and rest_i are divided by their gcd: only that gcd is long. g_i builds up one cheap gcd at a time,
    g_(i+1) = g_i * gcd(scale, rest_i), since for every prime p with a = v_p(scale) and l = v_p(denominator)
    min(i*a, l) + min(a, l - min(i*a, l)) = min((i + 1)*a, l). Multiplying by scale**i first and reducing against the
    whole denominator took a gcd of integers up to (degree * bits(scale)) bits longer at every coefficient: about
    5.8 of the 10.2 million units of 75 abscissas with 16-bit denominators."""
    scale_bits = scale.bit_length()
    coefficients = []
    rest, lift = denominator, 1  # rest_i and lift_i.
    for index, numerator in enumerate(numerators):
        budget.tick()
        if index and scale != 1:
            factor = scale  # lift_i / lift_(i-1) = scale / gcd(scale, rest_(i-1)).
            if rest != 1:
                rest_bits = rest.bit_length()
                budget.tick(_gcd_work(scale_bits, rest_bits))
                shared = gcd(scale, rest)
                if shared != 1:
                    shared_bits = shared.bit_length()
                    budget.tick(_division_work(rest_bits, shared_bits) + _division_work(scale_bits, shared_bits))
                    rest //= shared
                    factor = scale // shared
            budget.tick(_product_work(lift.bit_length(), factor.bit_length()))
            lift *= factor
        if not numerator:
            coefficients.append(_ZERO)
            continue
        reduced = rest
        if rest != 1:
            numerator_bits, rest_bits = numerator.bit_length(), rest.bit_length()
            budget.tick(_gcd_work(numerator_bits, rest_bits))
            divisor = gcd(numerator, rest)
            if divisor != 1:
                divisor_bits = divisor.bit_length()
                budget.tick(_division_work(numerator_bits, divisor_bits) + _division_work(rest_bits, divisor_bits))
                numerator //= divisor
                reduced = rest // divisor
        if lift != 1:
            budget.tick(_product_work(numerator.bit_length(), lift.bit_length()))
            numerator *= lift
        budget.check_int(numerator)
        budget.check_int(reduced)
        coefficients.append(trusted_rational(numerator, reduced))  # Lowest terms, a positive denominator.
    return tuple(coefficients)
