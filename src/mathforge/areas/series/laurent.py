"""Residues of rational functions at rational points (mathforge.series): residue.

residue(function, variable, point) takes a RationalFunction (MCP kind "rational_function") in its variable, a Symbol
(kind "symbol"), and an exact rational point a (kind "rational": an int, or a Rational in lowest terms with a positive
denominator); no argument has a default. Write f = N/D with N and D the numerator and the denominator that the record
holds: coprime, D monic. The value is the coefficient of (x - a)^(-1) in the Laurent expansion of f at a, a Rational
(never an int, also for an integral residue: Rational(0), Rational(1)); a point that is no pole of f (D(a) != 0) gives
Rational(0). The excluded polynomials of the record mark points where f is undefined but not infinite; they leave the
Laurent expansion of N/D, and so the residue, unchanged, and are not read.

Evidence (k, R, c), exactly this layout (docs/v09/cards/m7-extended.md), which the checker
mathforge.certify.series.laurent verifies:
- k, an int: the pole order, the multiplicity of a as a root of D, 0 <= k <= deg D (k = 0 when D(a) != 0);
- R, a tuple of Rationals (never ints), lowest degree first, without a trailing zero and exactly deg D - k + 1 of them:
  the cofactor with D = (x - a)^k R and R(a) != 0 (R is monic, as D is; for k = 0, R holds the coefficients of D);
- c, a tuple of exactly k Rationals: the principal part (c_{-k}, ..., c_{-1}), so that
  f = c_{-k} (x - a)^(-k) + ... + c_{-1} (x - a)^(-1) + (a function holomorphic at a); () for k = 0.
The value equals c_{-1} for k >= 1 and Rational(0) for k = 0.

Known values: 1/(x^2 - 1) at 1 -> 1/2; 1/x^3 at 0 -> 0; (x + 1)/x^2 at 0 -> 1; 1/(x - 2) at 3 -> 0;
x^2/(x - 1)^3 at 1 -> 1.

Invalid input: a function that is no RationalFunction (a Polynomial, an Expression, a str, ...), or one whose
numerator or denominator is no canonical Polynomial in the variable (1 to 1025 exact Rationals in lowest terms, lowest
degree first, without a trailing zero) or whose denominator is not monic (records forged with object.__new__); a
variable that is no Symbol with fields of their exact types, or not the function's variable (another Symbol of the same
name included); a point that is neither an exact int nor a Rational in lowest terms with a positive denominator (a
bool, a float, a str, a Fraction, a Rational subclass). A record forged outside lowest terms is read as the quotient
N/D, as the checker reads it: k is then the multiplicity of a in D, and leading entries of c may be 0. An integer
longer than max_integer_bits in an argument, a common denominator of N or D that long (the checker refuses such an
argument), and a value or evidence holding an integer that long (the checker could not read it) are resource limits.
The family defers nothing: every valid input within the budget has its value.

How produce() computes (nothing here is a proof; the checker verifies the claim by multiplying it out). With a = p/q in
lowest terms, N = N'/G and D = D'/F for integer vectors N', D' and their least common denominators G, F:
- k and R: for a = 0, k is the number of vanishing lowest coefficients of D' and S the rest. Otherwise k is guessed
  as the index g of the first Taylor coefficient of D' at a that does not vanish modulo the prime 2^61 - 1 (all of
  them from one product of integers, by Kronecker substitution), so g >= k; one exact long division of D' by the
  primitive integer polynomial (q x - p)^g (by Gauss's lemma an exact quotient has integer coefficients, so a division
  by q^g that leaves a remainder ends it) proves g = k when it succeeds, and a binary search with exact divisions
  finds k when it fails (R(a) divisible by the prime). Then D' = (q x - p)^k S, and R = S/lead(S), monic since D is.
  Dividing by q x - p again and again, k passes over the whole vector, would exceed the budget on 1/(x - 100)^1024,
  whose claim the checker verifies with a few thousand products.
- c: the Taylor coefficients of N and R at a, in z = q (x - a), are the integer polynomials T_A(z) = q^n A'((z + p)/q)
  (n = deg A), whose first k coefficients come from Horner's rule in Z[z]/(z^k): T <- T (z + p) + A'_i q^(n-i). The
  power series quotient U = T_N/T_R mod z^k follows from U_m = (T_N[m] - sum_{j=1..m} T_R[j] U_(m-j))/T_R[0], each
  U_m reduced, the last deg R of them kept over a running common denominator (the only ones the recursion reads).
  Then N/R = (E q^(deg R)/(G q^(deg N))) U(q (x - a)), with R = S/E, so c_{-(k-m)} = U_m E q^(deg R + m - deg N)/G.
Every product, quotient and gcd is charged before it runs, on the scale of mathforge.trusted.integers; a step of
Horner's rule is charged before it runs by a bound of its integers. The work is about one product of two integers of
17 (deg D + 1) bytes, (k + 1)(deg R + 1) products for the division (as many as the checker's condition (1)),
k (deg N + deg R + 2) Horner steps, and for the quotient 2 k (deg R + 1) products and 2 k gcds: on every input measured
in the m7b review the producer's work stays within 1.3 times the checker's. Under the default budget, 1/(x - a)^1024
at a for a = 1, 100 or 7/5, 1/x^1024 at 0 and 1/((x - 1)^512 (x + 1)^512) at 1 complete. The producer does not
estimate the check's work: operations_v6.run verifies every claim at once, within the request's limits, and reports a
check beyond them as resource_limit_exceeded.

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/series/test_laurent.py is the definition of done.
"""

from __future__ import annotations

from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...model import Assumption, Domain, Rational, Symbol, Truth
from ...polynomial import MAX_POLYNOMIAL_DEGREE, Polynomial
from ...rational_function import RationalFunction
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

# The scope text of the series area's v0.9 families (areas/series/power_series.py), extended by the objects of
# residue: rational functions and their residues.
_SCOPE = ("exact rationals, real functions and rational functions; certified closed forms of sums, series, residues "
          "and trigonometric integrals")

SPECS = (
    OperationSpec("residue", "series",
                  (Argument("function", "RationalFunction", None, "rational_function"),
                   Argument("variable", "Symbol", None, "symbol"),
                   Argument("point", "Rational | int", None, "rational")),
                  "Rational", "Compute the residue of a rational function at a rational point.", _SCOPE, version=6),
)

# No record classes: the value is a Rational, which serializes and renders without one, so there is no RENDER.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.series.laurent"

_ARGUMENTS = ("function", "variable", "point")  # The argument names of residue, in the order of its spec.
_MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
_NOT_A_FUNCTION = "residue requires a RationalFunction as its function"
_NOT_CANONICAL = ("residue requires a canonical RationalFunction: a numerator and a monic denominator, each a "
                  "Polynomial of 1 to 1025 exact Rationals in lowest terms without a trailing zero, in the variable")
_NOT_A_SYMBOL = "residue requires a Symbol as its variable"
_OTHER_VARIABLE = "residue requires the function to be a rational function in the variable"
_NOT_RATIONAL = "residue requires an exact rational point: an int or a Rational"
_NOT_CANONICAL_RATIONAL = "residue requires a point Rational in lowest terms with a positive denominator"
_LONG_DENOMINATOR = "Computation limit exceeded: integer bits of the common denominator of the function"

Pair = tuple[int, int]  # (numerator, denominator) of a canonical rational.
Key = tuple[object, ...]  # The fields of a Symbol that Symbol equality compares.


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of residue, for arguments that operations_v6.run has checked by name; the module states the
    value, the evidence and the algorithm. InvalidInput for an argument of the wrong type or form,
    UnsupportedOperation for another operation, ResourceLimitError past the budget."""
    if type(operation) is not str or operation != "residue":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the laurent family")
    if type(args) is not dict or any(type(name) is not str for name in args) or set(args) != set(_ARGUMENTS):
        raise InvalidInput(f"residue takes exactly the arguments {', '.join(_ARGUMENTS)}")
    with computation() as budget:
        budget.tick()  # The call itself; validating an argument charges at least one unit more.
        return _residue(budget, args["function"], args["variable"], args["point"])


# ---------------------------------------------------------------------------------------------------------------
# The operation

def _residue(budget: Budget, function: object, variable: object, point: object) -> tuple[Rational, tuple]:
    key = _symbol_key(variable)
    if key is None:
        raise InvalidInput(_NOT_A_SYMBOL)
    n_pairs, d_pairs, denominator = _function(budget, function, key)
    p, q = _rational(budget, point)
    n_integers, g = _integer_form(budget, n_pairs)  # N = N'/G.
    # D = D'/F; D is monic, so lead(S) carries F and R = S/lead(S) needs no F. _integer_form still refuses an F
    # beyond max_integer_bits, as the checker refuses that argument.
    d_integers, _ = _integer_form(budget, d_pairs)

    k, cofactor = _split(budget, d_integers, p, q)  # D' = (q x - p)^k S.
    if k == 0:  # D(a) != 0: N/D is holomorphic at a, R = D.
        return Rational(0), (0, denominator.coefficients, ())
    e = cofactor[-1]
    if e < 0:  # R = S/e with e > 0.
        cofactor, e = [-item for item in cofactor], -e
    r_pairs = [_reduced(budget, item, e) for item in cofactor]

    n_degree, r_degree = len(n_integers) - 1, len(cofactor) - 1
    qpowers = _powers(budget, q, max(n_degree, r_degree + k))
    t_n = _taylor(budget, n_integers, p, qpowers, k)
    t_r = _taylor(budget, cofactor, p, qpowers, k)
    if t_r[0] == 0:  # R(a) != 0 by the loop of _split: a zero here is a producer bug, never an input error.
        raise ArithmeticError("The cofactor vanishes at the point")
    principal = []
    for m, (numerator, part) in enumerate(_series_quotient(budget, t_n, t_r, k)):
        # c_{-(k-m)} = U_m E q^(deg R + m - deg N)/G.
        exponent = r_degree + m - n_degree
        numerator = _times(budget, numerator, e)
        part = _times(budget, part, g)
        if exponent > 0:
            numerator = _times(budget, numerator, qpowers[exponent])
        elif exponent < 0:
            part = _times(budget, part, qpowers[-exponent])
        principal.append(_reduced(budget, numerator, part))
    cofactor_value = tuple(_rational_of(budget, pair) for pair in r_pairs)
    c = tuple(_rational_of(budget, pair) for pair in principal)
    return c[-1], (k, cofactor_value, c)


def _split(budget: Budget, integers: list[int], p: int, q: int) -> tuple[int, list[int]]:
    """(k, S) with integers = (q x - p)^k S as integer polynomials and (q x - p) not dividing S.

    Dividing by q x - p again and again costs k passes over the whole vector, with integers of about deg D bits(p)
    bits: past the budget for 1/(x - 100)^1024, whose claim the checker verifies with a few thousand products. So k
    is guessed modulo a prime and confirmed by one exact division: the guess g is the index of the first Taylor
    coefficient of D' at a that does not vanish modulo _PRIME. A coefficient that vanishes vanishes modulo _PRIME too,
    so g >= k; (q x - p)^g divides D' exactly when g <= k, so a division that succeeds proves g = k. When it fails
    (R(a) is then divisible by _PRIME, as for (x - 1)^2 (x - 1 - _PRIME) at 1), a binary search over
    [0, g) with exact divisions finds k: divisibility by (q x - p)^j holds exactly for j <= k."""
    if p == 0:  # a = 0, q = 1: x^k divides exactly when the k lowest coefficients vanish.
        budget.tick(len(integers))
        k = next(i for i, item in enumerate(integers) if item)  # integers is not zero: D is monic.
        return k, integers[k:]
    guess = _modular_order(budget, integers, p, q)
    if guess == 0:  # T_0 = D'(a) q^n != 0: no pole, S = D'.
        return 0, integers
    quotient = _exact_quotient(budget, integers, p, q, guess)
    if quotient is not None:
        return guess, quotient
    low, high, found = 0, guess, integers  # (q x - p)^low divides with quotient found; (q x - p)^high does not.
    while high - low > 1:
        budget.tick()
        middle = (low + high) // 2
        quotient = _exact_quotient(budget, integers, p, q, middle)
        if quotient is None:
            high = middle
        else:
            low, found = middle, quotient
    return low, found


# A prime above MAX_POLYNOMIAL_DEGREE (so that no factorial up to the degree vanishes modulo it), below 2^61 so that
# its residues fit a machine word: the Mersenne prime 2^61 - 1.
_PRIME = (1 << 61) - 1


def _modular_order(budget: Budget, integers: list[int], p: int, q: int) -> int:
    """The least j with T_j != 0 (mod _PRIME), or the degree n when there is none, for the integer Taylor polynomial
    T(z) = sum_i A_i q^(n-i) (z + p)^i = sum_j T_j z^j of A = integers at a = p/q (z = q x - p).

    T_j = sum_{i >= j} A_i q^(n-i) binomial(i, j) p^(i-j), so n! j! T_j = sum_{l=0..n-j} u_(j+l) v_l with
    u_i = A_i q^(n-i) i! and v_l = p^l n!/l!, all integers: a correlation, which one product of integers computes by
    Kronecker substitution. Modulo the prime, every u_i and v_l is below 2^61 and every sum of products below
    (n + 1) 2^122, so slots of `width` bytes hold them without carries. As _PRIME > n, n! j! is invertible modulo it
    and T_j vanishes modulo _PRIME exactly when slot n - j of the product does. Work: O(n) word operations, the
    reductions of the n + 1 coefficients and one product of two integers of about 17 (n + 1) bytes."""
    from ...trusted.integers import division_work, product_work

    n = len(integers) - 1
    reduced = []
    for coefficient in integers:
        budget.tick(1 + division_work(coefficient.bit_length(), 61))
        reduced.append(coefficient % _PRIME)
    budget.tick(1 + division_work(q.bit_length(), 61) + division_work(p.bit_length(), 61) + 6 * (n + 1))
    q_reduced, p_reduced = q % _PRIME, p % _PRIME
    factorials = [1] * (n + 1)  # i! modulo the prime.
    for i in range(1, n + 1):
        factorials[i] = factorials[i - 1] * i % _PRIME
    u = [0] * (n + 1)
    q_power = 1  # q^(n-i), for i from n down.
    for i in range(n, -1, -1):
        u[i] = reduced[i] * q_power % _PRIME * factorials[i] % _PRIME
        q_power = q_power * q_reduced % _PRIME
    v = [0] * (n + 1)
    tail = [1] * (n + 1)  # tail[l] = n!/l! = (l + 1) (l + 2) ... n, modulo the prime.
    for l in range(n - 1, -1, -1):
        tail[l] = tail[l + 1] * (l + 1) % _PRIME
    p_power = 1  # p^l, for l from 0 up.
    for l in range(n + 1):
        v[l] = p_power * tail[l] % _PRIME
        p_power = p_power * p_reduced % _PRIME
    width = (2 * 61 + (n + 1).bit_length() + 8) // 8  # Bytes per slot: room for (n + 1) 2^122 without a carry.
    # The correlation sum_l u_(j+l) v_l is slot n - j of (u reversed) * v.
    left = int.from_bytes(b"".join(item.to_bytes(width, "little") for item in reversed(u)), "little")
    right = int.from_bytes(b"".join(item.to_bytes(width, "little") for item in v), "little")
    budget.tick(1 + product_work(left.bit_length(), right.bit_length()) + 2 * (n + 1))
    data = (left * right).to_bytes(2 * (n + 1) * width, "little")
    for j in range(n + 1):
        slot = n - j
        if int.from_bytes(data[slot * width:(slot + 1) * width], "little") % _PRIME:
            return j
    return n


def _exact_quotient(budget: Budget, integers: list[int], p: int, q: int, order: int) -> list[int] | None:
    """The integer quotient S of W = integers by (q x - p)^order (1 <= order), when it divides W; None otherwise.

    Long division from the top by t = (q x - p)^order, t_j = binomial(order, j) q^j (-p)^(order-j): with m = n - order,
    S_i = (W_(i+order) - sum_{j<order} t_j S_(i+order-j))/q^order for i from m down, then the remainder, the order
    lowest coefficients W_i - sum_j t_j S_(i-j), must vanish. q x - p is primitive, so by Gauss's lemma S is an
    integer polynomial when the division is exact, and a division by q^order that leaves a remainder answers None at
    once. Every product is charged before it runs: about (m + 1)(order + 1) of them, the products that the checker's
    condition (1) costs."""
    from ...trusted.integers import division_work

    n = len(integers) - 1
    m = n - order
    if m < 0:
        return None
    t = [0] * (order + 1)
    negative = _powers(budget, -p, order)
    qpowers = _powers(budget, q, order)
    binomial = 1
    for j in range(order + 1):
        budget.tick()
        t[j] = _times(budget, _times(budget, binomial, qpowers[j]), negative[order - j])
        if j < order:
            binomial = _quotient(budget, _times(budget, binomial, order - j), j + 1)
    lead = t[order]  # q^order >= 1.
    quotient = [0] * (m + 1)
    for i in range(m, -1, -1):
        budget.tick()
        total = integers[i + order]
        for j in range(max(0, i + order - m), order):
            total -= _times(budget, t[j], quotient[i + order - j])
        if lead != 1:
            budget.tick(1 + division_work(total.bit_length(), lead.bit_length()))
            total, remainder = divmod(total, lead)
            if remainder:
                return None
        quotient[i] = total
    for i in range(order):
        budget.tick()
        total = integers[i]
        for j in range(max(0, i - m), i + 1):
            total -= _times(budget, t[j], quotient[i - j])
        if total:
            return None
    return quotient



def _taylor(budget: Budget, integers: list[int], p: int, qpowers: list[int], count: int) -> list[int]:
    """The first count coefficients (zeros beyond the degree) of T(z) = q^n A((z + p)/q) = sum_i A_i q^(n-i) (z + p)^i
    for the integer vector A of degree n = len(integers) - 1, qpowers holding q^0 .. q^n at least: Horner's rule in
    Z[z]/(z^count), T <- T (z + p) + A_i q^(n-i) for i from n down to 0. After the step of A_i, T has degree n - i, so
    that step updates min(count, n - i + 1) entries. Every entry stays below (n + 1) max|A_i q^(n-i)| (1 + |p|)^n in
    absolute value, the bound by which each step is charged before it runs."""
    from ...trusted.integers import product_work

    n = len(integers) - 1
    weighted = []
    for i, coefficient in enumerate(integers):
        budget.tick()
        weighted.append(coefficient if qpowers[n - i] == 1 or not coefficient
                        else _times(budget, coefficient, qpowers[n - i]))
    if p == 0:  # T(z) = sum_i A_i q^(n-i) z^i.
        values = weighted[:count]
        return values + [0] * (count - len(values))
    bound = _bits(weighted) + (n + 1).bit_length() + n * (abs(p) + 1).bit_length()
    step = 1 + product_work(p.bit_length(), bound) + (bound >> 11)
    values = [0] * count
    for i in range(n, -1, -1):
        active = min(count, n - i + 1)
        budget.tick(active * step)
        for j in range(active - 1, 0, -1):
            values[j] = p * values[j] + values[j - 1]
        values[0] = p * values[0] + weighted[i]
    return values


def _series_quotient(budget: Budget, numerator: list[int], divisor: list[int], count: int) -> list[Pair]:
    """The reduced pairs of U_0, ..., U_(count-1) with U = numerator/divisor mod z^count, divisor[0] != 0:
    U_m = (numerator[m] - sum_{j=1..m} divisor[j] U_(m-j))/divisor[0]. U_m reads only the U_(m-j) with
    divisor[j] != 0, so only the last `reach` of them (reach = the largest such j) are kept, as integers over a common
    denominator L of every U_i so far, which grows only by a factor that divides divisor[0] in a step.

    Regression (m7b review): every earlier U_i was once kept and rescaled whenever L grew, about count^2/2 products;
    for 1/((x - 1/2)^1010 R) with deg R = 14 that cost 1,990,000 units of the default 2,000,000, against the checker's
    830,000. Keeping the U_m unreduced over divisor[0]^(m+1) instead avoids the rescaling but inflates them when the
    true denominators grow more slowly: for 1/((x - 1)^512 (x + 1)^512) at 1, divisor[0] = 2^512 while the
    denominators of the U_m grow by 2 a step, beyond the budget. The window keeps the reduced sizes, and costs
    count reach rescaling products at most."""
    head = divisor[0]
    nonzero = [j for j in range(1, count) if divisor[j]]  # The sum runs over the nonzero divisor[j] alone.
    reach = nonzero[-1] if nonzero else 0
    budget.tick(count)
    common, recent, pairs = 1, [], []  # recent[-j] = U_(m-j) common for j = 1 .. len(recent) <= reach.
    for m in range(count):
        budget.tick()
        total = 0
        for j in nonzero:
            if j > m:
                break
            budget.tick()
            if recent[-j]:
                total += _times(budget, divisor[j], recent[-j])
        top = _times(budget, numerator[m], common) - total
        bottom = _times(budget, head, common)
        top, bottom = _reduced(budget, top, bottom)
        factor = _quotient(budget, bottom, _gcd(budget, common, bottom))  # lcm(L, bottom) = L factor.
        if factor != 1:
            recent = [item if not item else _times(budget, item, factor) for item in recent]
            budget.tick(len(recent))
            common = _times(budget, common, factor)
        if reach:
            budget.tick(1 + len(recent) // 64)
            recent.append(_times(budget, top, _quotient(budget, common, bottom)))
            if len(recent) > reach:
                del recent[0]
        pairs.append((top, bottom))
    return pairs


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _symbol_key(symbol: object) -> Key | None:
    """(name, domain, assumptions, uid) of a Symbol whose fields have their exact types, as the checker requires them:
    what Symbol equality compares. None for anything else, a Symbol forged with fields of other types included."""
    if type(symbol) is not Symbol:
        return None
    name, domain = getattr(symbol, "name", None), getattr(symbol, "domain", None)
    assumptions, uid = getattr(symbol, "assumptions", None), getattr(symbol, "uid", None)
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > _MAX_ASSUMPTIONS):
        return None
    facts = []
    for assumption in assumptions:
        if type(assumption) is not Assumption:
            return None
        fact = getattr(assumption, "name", None), getattr(assumption, "truth", None)
        if type(fact[0]) is not str or type(fact[1]) is not Truth:
            return None
        facts.append(fact)
    return name, domain, tuple(facts), uid


def _function(budget: Budget, function: object, key: Key) -> tuple[list[Pair], list[Pair], Polynomial]:
    """(pairs of N, pairs of D, the denominator record) of a RationalFunction argument in the variable of that key,
    with canonical polynomials and D monic; InvalidInput otherwise."""
    if type(function) is not RationalFunction:
        raise InvalidInput(_NOT_A_FUNCTION)
    numerator, denominator = getattr(function, "numerator", None), getattr(function, "denominator", None)
    if type(numerator) is not Polynomial or type(denominator) is not Polynomial:
        raise InvalidInput(_NOT_CANONICAL)
    n_variable, d_variable = getattr(numerator, "variable", None), getattr(denominator, "variable", None)
    n_key, d_key = _symbol_key(n_variable), _symbol_key(d_variable)
    if n_key is None or d_key is None:
        raise InvalidInput(_NOT_CANONICAL)
    if n_key != key or d_key != key:
        raise InvalidInput(_OTHER_VARIABLE)
    n_pairs = _coefficients(budget, numerator)
    d_pairs = _coefficients(budget, denominator)
    if d_pairs[-1] != (1, 1):  # Monic, hence not zero.
        raise InvalidInput(_NOT_CANONICAL)
    return n_pairs, d_pairs, denominator


def _coefficients(budget: Budget, polynomial: Polynomial) -> list[Pair]:
    """The pairs of a canonical coefficient tuple: 1 to 1025 exact Rationals in lowest terms, lowest degree first,
    without a trailing zero; InvalidInput otherwise. One unit for each coefficient."""
    coefficients = getattr(polynomial, "coefficients", None)
    if type(coefficients) is not tuple or not 0 < len(coefficients) <= MAX_POLYNOMIAL_DEGREE + 1:
        raise InvalidInput(_NOT_CANONICAL)
    pairs = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:  # Neither an int nor a float nor a Rational subclass.
            raise InvalidInput(_NOT_CANONICAL)
        pairs.append(_integers(budget, coefficient, _NOT_CANONICAL))
    if len(pairs) > 1 and not pairs[-1][0]:  # A trailing zero.
        raise InvalidInput(_NOT_CANONICAL)
    return pairs


def _integers(budget: Budget, value: Rational, message: str) -> Pair:
    """(numerator, denominator) of a Rational, which must hold two exact ints in lowest terms with a positive
    denominator: one made with object.__new__ may lack a field or break that form, InvalidInput(message) then.
    ResourceLimitError for an integer longer than max_integer_bits, before the gcd that its length bounds."""
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(message)
    budget.check_int(numerator)
    budget.check_int(denominator)
    if denominator != 1 and _gcd(budget, numerator, denominator) != 1:
        raise InvalidInput(message)
    return numerator, denominator


def _rational(budget: Budget, value: object) -> Pair:
    """(p, q) of the point: an exact int, or a canonical Rational; InvalidInput for anything else."""
    budget.tick()
    if type(value) is int:  # Neither a bool nor a float is an exact rational.
        budget.check_int(value)
        return value, 1
    if type(value) is not Rational:  # A float, a str, None, a Fraction, a Rational subclass, a container.
        raise InvalidInput(_NOT_RATIONAL)
    return _integers(budget, value, _NOT_CANONICAL_RATIONAL)


def _integer_form(budget: Budget, pairs: list[Pair]) -> tuple[list[int], int]:
    """(P, L): the least common denominator L of the pairs and the integers P[i] = pairs[i] L. ResourceLimitError when
    L has more than max_integer_bits bits, as the checker refuses such an argument."""
    common = 1
    limit = budget.limits.max_integer_bits
    for _, part in pairs:
        budget.tick()
        if part != 1:
            common = _times(budget, _quotient(budget, common, _gcd(budget, common, part)), part)
            if common.bit_length() > limit:
                raise ResourceLimitError(_LONG_DENOMINATOR)
    integers = []
    for numerator, part in pairs:
        budget.tick()
        integers.append(numerator if common == 1 else _times(budget, numerator, _quotient(budget, common, part)))
    return integers, common


def _reduced(budget: Budget, numerator: int, denominator: int) -> Pair:
    """numerator/denominator in lowest terms with a positive denominator, for a denominator != 0."""
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if denominator != 1:
        common = _gcd(budget, numerator, denominator)
        if common != 1:
            numerator, denominator = _quotient(budget, numerator, common), _quotient(budget, denominator, common)
    return numerator, denominator


def _rational_of(budget: Budget, pair: Pair) -> Rational:
    """The Rational of a reduced pair; ResourceLimitError for an integer longer than max_integer_bits, which the
    checker could not read in a value or evidence."""
    budget.check_int(pair[0])
    budget.check_int(pair[1])
    return Rational(*pair)


# ---------------------------------------------------------------------------------------------------------------
# Charged integer arithmetic, on the cost scale of mathforge.trusted.integers

def _times(budget: Budget, left: int, right: int) -> int:
    """left * right, charged before it runs: one unit and what the lengths cost beyond it (product_work)."""
    from ...trusted.integers import product_work

    budget.tick(1 + product_work(left.bit_length(), right.bit_length()))
    return left * right


def _quotient(budget: Budget, dividend: int, divisor: int) -> int:
    """dividend // divisor for a divisor != 0, its long division charged before it runs (division_work)."""
    from ...trusted.integers import division_work

    budget.tick(1 + division_work(dividend.bit_length(), divisor.bit_length()))
    return dividend // divisor


def _gcd(budget: Budget, left: int, right: int) -> int:
    """math.gcd, charged before it runs: the long division of the longer operand by the shorter (division_work) and
    one unit per 64 squared 64-bit words of the shorter."""
    from ...trusted.integers import division_work

    longer, shorter = max(left.bit_length(), right.bit_length()), min(left.bit_length(), right.bit_length())
    words = shorter >> 6
    budget.tick(1 + (words * words >> 6) + division_work(longer - shorter + 64, shorter))
    return gcd(left, right)


def _powers(budget: Budget, base: int, count: int) -> list[int]:
    """[base^0, base^1, ..., base^count], each a charged product; base = 1 costs one unit per entry."""
    powers = [1]
    for _ in range(count):
        budget.tick()
        powers.append(powers[-1] if base == 1 else _times(budget, powers[-1], base))
    return powers


def _bits(values: list[int]) -> int:
    """The largest bit length among the integers (0 for none)."""
    return max((value.bit_length() for value in values), default=0)
