"""Arithmetic functions of exact integers (mathforge.ntheory): the Moebius function, the sums of powers of the divisors
and the integer part of a k-th root.

- moebius(n), n >= 1: mu(n), an int in {-1, 0, 1}. Evidence ("square", s) with an int s > 1 and s*s | n when mu(n) = 0;
  otherwise ("factorization", factors), the certified prime factorization of n with every exponent 1
  (("factorization", ()) for n = 1).
- divisor_sigma(n, k), n >= 1, k >= 0 and k * n.bit_length() <= max_integer_bits (else the budget's resource limit):
  sigma_k(n), the sum of d**k over the positive divisors d of n, a positive int. Evidence (factors,), the certified
  prime factorization of n (((),) for n = 1).
- integer_nth_root(n, k), n >= 0, k >= 1: floor(n**(1/k)), the int r >= 0 with r**k <= n < (r + 1)**k. Evidence ().

A certified factorization is ((p, e, certificate), ...) with the primes strictly increasing, every e >= 1 and every
certificate as mathforge.kernels.certificates.prime_evidence(p) builds it (() below psi_13);
mathforge.kernels.certificates.factorization_evidence(n) builds the whole tuple. Every argument has kind "int", so
operations_v6.run rejects a bool, a float, a str or None before produce() runs; produce() tests the types again and
rejects the rest of an input outside the domain with InvalidInput. The values are plain ints, so the family defines no
record class and no RENDER. Its checker, mathforge.certify.ntheory.arithmetic_functions, states the theorems that it
verifies. mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md
is the checklist, and tests/v09/ntheory/test_arithmetic_functions.py is the definition of done.

How produce() computes:

- moebius divides n by the primes below 2**10 that divide it (a gcd with their product names them) and stops at the
  first of them whose square divides n: mu(n) = 0 with ("square", p). The cofactor, which has no prime factor below
  2**10, goes to the integer kernel's factor(), and its least prime with an exponent above 1 gives ("square", p) too.
  So s is always the least prime whose square divides n, and mu(n) = 0 never waits for a prime certificate. A
  squarefree n with r primes has mu(n) = (-1)**r, and each prime takes its certificate from prime_evidence(p), so
  that the cofactor is factored once (factorization_evidence(n) would factor it again).
- divisor_sigma refuses k * bits(n) > max_integer_bits before any work, then takes factorization_evidence(n) and
  multiplies the closed forms sigma_k(p**e) = (p**(k*(e + 1)) - 1) // (p**k - 1), and e + 1 for k = 0. The checker
  sums the same geometric series by Horner's rule instead.
- integer_nth_root answers n itself for n <= 1 or k = 1, and 1 for k >= bits(n) (then 1 <= n < 2**bits(n) <= 2**k).
  Otherwise the root r lies below 2**m, m = (bits(n) - 1) // k + 1. Bisection finds, bit by bit, the root top of
  n >> (k * shift), where shift = max(0, m - head) and head = 2 * bits(k) + 2; with shift = 0, top is r. Newton's
  iteration x -> ((k - 1) * x + n // x**(k - 1)) // k then runs from x = (top + 1) << shift until x stops decreasing,
  and the last x is r. The start is above r: n < ((n >> (k * shift)) + 1) * 2**(k * shift) <= x**k. A step never
  falls below r: the arithmetic mean of k - 1 copies of x and n / x**(k - 1) is at least their geometric mean,
  n**(1/k), and the floors keep that. A step from x > r decreases, as x**k > n. With shift > 0, top has exactly head
  bits, so the start is within a factor 1 + 2**(1 - head) of r, and Newton's error contracts quadratically from the
  first step (the constant (k - 1) / 2 times that relative error is below 2**(-bits(k) - 2)): about
  log2(m / head) + 2 steps. From 2**m, up to twice r, the first steps would shrink x by little more than a factor
  (k - 1) / k each.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. An argument longer than max_integer_bits raises ResourceLimitError first. The integer
kernel and kernels.certificates charge their own work, and every power, product and long division here is charged
before the builtin computes it, with the cost model of trusted.integers (product_work, division_work). A certified
factorization is refused with ResourceLimitError when its check would need more than max_work, the rule of
kernels.certificates: the verifier checks in a fresh budget of the same limits, so the check may take all of max_work
whatever the production spent, and producer and checker run out at the same point. moebius reserves all that its
checker charges for the factorization, trusted.integers.factorization_work and one unit per entry for the checker's
exponent screen. divisor_sigma reserves factorization_work and the checker's Horner evaluation with its powers p**k, on
the checker's cost model (_horner_work): at most about 131,000 units under the cap, at divisor_sigma(3, 32768), against
70,149 for the producer's closed form, so that the producer refuses that claim under max_work=100_000 rather than
leaving verification to run out. The inner Horner sums, which the closed form never forms, are charged by a bound on
their lengths, which may reserve a few units more than the checker uses. The divisor_sigma cap is the checker's as well.
"""

from __future__ import annotations

from math import gcd, prod

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
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

# The scope text of every operation of wave M1 (docs/v09/cards/m1-ntheory.md).
SCOPE = "exact integers; certified primes, orders and residues; canonical integer solution families"

SPECS = (
    OperationSpec("moebius", "ntheory", (Argument("n", "int", None, "int"),), "int",
                  "Compute the Moebius function of a positive integer.", SCOPE, version=6),
    OperationSpec("divisor_sigma", "ntheory", (Argument("n", "int", None, "int"), Argument("k", "int", None, "int")),
                  "int", "Compute the sum of the k-th powers of the positive divisors of a positive integer.", SCOPE,
                  version=6),
    OperationSpec("integer_nth_root", "ntheory",
                  (Argument("n", "int", None, "int"), Argument("k", "int", None, "int")), "int",
                  "Compute the integer part of the k-th root of a nonnegative integer.", SCOPE, version=6),
)

# Every value is a plain int, which the shared renderer prints: no record class, and so no RENDER.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.ntheory.arithmetic_functions"

SQUARE, FACTORIZATION = "square", "factorization"  # The two tags of the evidence of moebius.
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the evidence"
_SIGMA_CAP = "Computation limit exceeded: divisor_sigma takes k * bits(n) <= max_integer_bits"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name; the
    module states the algorithms and the evidence. InvalidInput for an argument outside the operation's domain."""
    if operation == "moebius":
        n = args["n"]
        if type(n) is not int or n < 1:
            raise InvalidInput("moebius requires an exact integer n >= 1")
        with computation() as budget:
            budget.check_int(n)
            return _moebius(n, budget)
    if operation == "divisor_sigma":
        n, k = args["n"], args["k"]
        if type(n) is not int or type(k) is not int or n < 1 or k < 0:
            raise InvalidInput("divisor_sigma requires exact integers n >= 1 and k >= 0")
        with computation() as budget:
            budget.check_int(n)
            budget.check_int(k)
            return _divisor_sigma(n, k, budget)
    if operation == "integer_nth_root":
        n, k = args["n"], args["k"]
        if type(n) is not int or type(k) is not int or n < 0 or k < 1:
            raise InvalidInput("integer_nth_root requires exact integers n >= 0 and k >= 1")
        with computation() as budget:
            budget.check_int(n)
            budget.check_int(k)
            return _integer_nth_root(n, k, budget), ()
    raise UnsupportedOperation(f"{operation} is not an operation of the arithmetic_functions family")


def _moebius(n: int, budget: Budget) -> tuple[int, tuple]:
    """mu(n) for n >= 1 and its evidence: ("square", p) for the least prime p with p*p | n, else the certified
    factorization."""
    from ...kernels.integers import SMALL_PRIMES, factor
    from ...trusted.integers import division_work

    primes: list[int] = []  # The primes of n found so far, in increasing order; each divides n exactly once.
    rest = n
    small = prod(SMALL_PRIMES)  # The 172 primes below 2**10: 1,419 bits.
    # One long division for the gcd, and a unit for each small prime that may divide it.
    budget.tick(1 + division_work(rest.bit_length(), small.bit_length()) + len(SMALL_PRIMES))
    common = gcd(rest, small)  # The product of the primes below 2**10 that divide n.
    for p in SMALL_PRIMES:
        if common == 1:
            break
        if common % p:
            continue
        common //= p
        budget.tick(2 * (1 + division_work(rest.bit_length(), p.bit_length())))  # Two long divisions by p.
        rest //= p
        if rest % p == 0:
            return 0, (SQUARE, p)
        primes.append(p)
    if rest > 1:
        # No prime below 2**10 divides rest, so the kernel's trial division finds nothing to remove one at a time.
        for p, exponent in factor(rest):  # Increasing primes; the kernel charges the budget.
            if exponent > 1:
                return 0, (SQUARE, p)
            primes.append(p)
    factors = _certified(primes, budget)
    return (1 if len(factors) % 2 == 0 else -1), (FACTORIZATION, factors)


def _certified(primes: list[int], budget: Budget) -> tuple[tuple[int, int, tuple], ...]:
    """((p, 1, certificate), ...) for the increasing primes of a squarefree integer, each certificate from
    prime_evidence; ResourceLimitError when the checker would need more than max_work for it, the rule of
    factorization_evidence: the checker has a fresh budget of the same limits to itself, whatever the production
    spent. The checker of moebius charges one unit per entry (its exponent screen) before check_factorization charges
    trusted.integers.factorization_work, and kernels.certificates asks a family whose checker does more than
    check_factorization to leave room for it."""
    from ...kernels.certificates import prime_evidence
    from ...trusted.integers import factorization_work

    entries = []
    for p in primes:
        try:
            certificate = prime_evidence(p)
        except InvalidInput as error:  # factor() took a composite for a prime: a producer bug, never an input error.
            raise ArithmeticError("Internal factorization produced a composite factor") from error
        entries.append((p, 1, certificate))
    factors = tuple(entries)
    spare = budget.limits.max_work - len(factors)  # What check_factorization may charge after the checker's screen.
    if spare < 0 or factorization_work(factors, spare) > spare:
        raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
    return factors


def _divisor_sigma(n: int, k: int, budget: Budget) -> tuple[int, tuple]:
    """sigma_k(n) for n >= 1 and k >= 0 and its evidence, (certified factorization of n,). ResourceLimitError when the
    checker would need more than max_work: check_factorization's work and its Horner evaluation's together."""
    from ...kernels.certificates import factorization_evidence
    from ...trusted.integers import division_work, factorization_work, product_work

    # p**k may have up to k * bits(n) bits: the checker raises at this same cap once the evidence has passed.
    if k * n.bit_length() > budget.limits.max_integer_bits:
        raise ResourceLimitError(_SIGMA_CAP)
    factors = factorization_evidence(n)
    total = 1
    horner = 0  # What the checker charges beyond check_factorization, prime by prime (_horner_work).
    for prime, exponent, _ in factors:
        if k:
            step = _power(budget, prime, k)  # p**k >= 2, so the quotient below divides exactly.
            top = _power(budget, step, exponent + 1)
            budget.tick(1 + division_work(top.bit_length(), step.bit_length()))
            part = (top - 1) // (step - 1)  # 1 + p**k + ... + p**(k*e)
        else:
            step, part = 1, exponent + 1  # The number of divisors of p**e.
        horner += _horner_work(prime, exponent, k, step.bit_length(), part.bit_length(), total.bit_length())
        budget.tick(1 + product_work(total.bit_length(), part.bit_length()))
        total *= part
    # The rule of kernels.certificates, which factorization_evidence applied to check_factorization alone: the whole
    # check fits in max_work, which the checker has to itself in a fresh budget.
    spare = budget.limits.max_work - horner  # What check_factorization may charge beside the Horner evaluation.
    if spare < 0 or factorization_work(factors, spare) > spare:
        raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
    return total, (factors,)


def _horner_work(prime: int, exponent: int, k: int, step_bits: int, part_bits: int, total_bits: int) -> int:
    """The work that the checker of divisor_sigma charges for the prime power p**e of the factorization, after its cost
    model (certify.ntheory.arithmetic_functions, where a product of integers of a and b bits costs
    1 + product_work(a, b)): a unit for the prime; p**k, charged as two products of k * bits(p) bits per bit of k (none
    for k = 0); the e Horner steps s -> s * p**k + 1, a product each; and the product of the sums so far, of
    total_bits, with the whole sum, of part_bits. The sums 1 + p**k + ... + p**(k*j) for 0 < j < e, which the closed
    form never forms, are bounded by their lengths, below j * bits(p**k) + 1 and at most part_bits. So the result is
    the checker's charge, or a little more where such a bound passes a 64-bit word that the sum itself does not."""
    from ...trusted.integers import product_work

    work = 1 + exponent + 1 + product_work(total_bits, part_bits)  # The prime's unit and the units of the products.
    if k:
        power_bits = prime.bit_length() * k
        work += 2 * k.bit_length() * (1 + product_work(power_bits, power_bits))
    if step_bits >> 6:  # product_work(..., step_bits) is 0 below 64 bits, whatever the length of the sum.
        for j in range(1, exponent):  # The sum before step j; before step 0 it is 1, whose product costs the unit.
            work += product_work(min(j * step_bits + 1, part_bits), step_bits)
    return work


def _integer_nth_root(n: int, k: int, budget: Budget) -> int:
    """floor(n**(1/k)) for n >= 0 and k >= 1: bisection for the leading bits, then Newton's iteration (see the
    module)."""
    from ...trusted.integers import division_work, product_work

    if n < 2 or k == 1:
        return n
    bits = n.bit_length()
    if k >= bits:  # 1 <= n < 2**bits <= 2**k, so 1 <= root < 2.
        return 1
    head = 2 * k.bit_length() + 2  # The leading bits of the root that the bisection finds.
    shift = max(0, (bits - 1) // k + 1 - head)
    budget.tick()
    top = _bisected_root(n >> (k * shift), k, budget)
    if not shift:
        return top
    x = (top + 1) << shift  # Above the root, within a factor 1 + 2**(1 - head) of it.
    while True:
        power = _power(budget, x, k - 1)
        # n // power, (k - 1) * x and the division of their sum by k.
        budget.tick(3 + division_work(bits, power.bit_length()) + product_work(k.bit_length(), x.bit_length())
                    + division_work(bits + k.bit_length(), k.bit_length()))
        following = ((k - 1) * x + n // power) // k
        if following >= x:
            return x
        x = following


def _bisected_root(n: int, k: int, budget: Budget) -> int:
    """floor(n**(1/k)) for n >= 1 and k >= 2, one bit at a time from the top: the root lies below
    2**((bits(n) - 1) // k + 1), and a bit stays set when the k-th power of the root so far with that bit is <= n."""
    root = 0
    for position in range((n.bit_length() - 1) // k, -1, -1):
        candidate = root | 1 << position
        if _power(budget, candidate, k) <= n:
            root = candidate
    return root


def _power(budget: Budget, base: int, exponent: int) -> int:
    """base**exponent for ints base >= 0 and exponent >= 1, charged before the builtin runs. The power has fewer than
    length = bits(base) * exponent bits; binary exponentiation squares numbers of at most length / 2 bits, which
    together cost at most a third of a product of two length-bit numbers, and multiplies by the base once per bit of
    the exponent, which costs at most such a product more."""
    from ...trusted.integers import product_work

    if base < 2 or exponent == 1:
        return base
    length = base.bit_length() * exponent
    budget.tick(1 + exponent.bit_length() + 2 * product_work(length, length))
    return base ** exponent
