"""The nearest primes above and below an integer: next_prime and prev_prime (mathforge.ntheory).

next_prime(n) is the smallest prime greater than the exact integer n, 2 for every n < 2. prev_prime(n) is the largest
prime smaller than n, defined for n >= 3: n <= 2 is invalid input, since no prime lies below 2. Both values are plain
ints, so the family has no record class and no RENDER entry.

The evidence of both operations is (certificate, witnesses):

- certificate proves the value r prime, as mathforge.kernels.certificates.prime_evidence(r) builds it: () below
  psi_13 = 3,317,044,064,679,887,385,961,981, a Pocklington certificate from psi_13 on. That needs the complete
  factorization of r - 1; when the budget cannot deliver it, the result is a resource error (the documented cliff).
- witnesses holds one entry for each skipped integer m (n < m < r for next_prime, r < m < n for prev_prime) that the
  checker cannot show composite on its own, in increasing order of m and for no other m: those with m >= 10,000,
  gcd(m, 2*3*...*97) == 1 and pow(2, m - 1, m) == 1. The entry is ("fermat", m, w) with 1 < w < m and
  pow(w, m - 1, m) != 1, or else ("factor", m, d) with a divisor 1 < d < m of m, which a Carmichael number needs.

For n < 2, next_prime gives 2 with the evidence ((), ()). The checker, mathforge.certify.ntheory.prime_search,
enumerates the skipped integers itself and states the theorems and the work it charges.

The producer walks from n towards the result one integer at a time, up for next_prime and down for prev_prime, and
sorts each candidate m as the checker will:

- m < 10,000: the integer kernel's table of smallest prime factors decides, exactly.
- m >= 10,000 with a prime factor up to 97 (a gcd with their product): composite, and the checker sees it too.
- any other m takes the strong probable-prime test to base 2. When 2**(m - 1) is not 1 modulo m, m is composite and
  the checker's own Fermat test to base 2 shows it. Otherwise m is a prime or needs a witness. Below psi_13 the strong
  tests go on to the prime bases 3 to 41, which decide every m there. The witness of a composite m is
  ("fermat", m, w) for the first base w with w**(m - 1) not 1 modulo m, as the card prefers; when no base is one, as
  for a Carmichael number, it is ("factor", m, d) with d = gcd(x - 1, m) for the first square root x of 1 other than
  1 and -1 that a strong test met: m divides (x - 1)(x + 1) but neither factor. From psi_13 on the strong tests go on
  to base 3 only, and the bases 5 to 97 are tried only to choose the witness of an m that needs one, the same way.
- the first candidate that no test shows composite is the value r, and prime_evidence(r) proves it prime. From psi_13
  on the strong tests prove nothing and each costs a multiplication modulo m per bit of m, so 2 and 3 screen there
  and prime_evidence decides. Should it find the candidate composite (a strong pseudoprime to 2 and 3, such as
  psi_13 itself), the strong tests to the primes 5 to 97 give its witness as above, or, for a strong pseudoprime to
  all of them, which can be constructed, the integer kernel's prime_certificate does, and the walk goes on. Base 3
  matters: every composite 2**p - 1 with p prime is a strong pseudoprime to base 2 (its m - 1 = 2 * (2**(p-1) - 1)
  has an odd part divisible by p), and it is the first candidate of prev_prime(2**p); the composite Fermat numbers
  2**(2**k) + 1, which next_prime(2**(2**k)) meets first, pass base 2 as well. Base 3 is a Fermat witness for each
  composite 2**p - 1 with a prime p from 83 to 1,300 and for 2**(2**k) + 1 with k from 7 to 11, at the price of one
  exponentiation, where prime_evidence would have to factor m - 1. The 23 bases from 5 on, tried on every candidate
  that passes 2 and 3, would cost 23 exponentiations modulo the value r, most of the budget for an r of 2,000 bits:
  they made the producer refuse next_prime(3 * 2**2208), whose claim fits in the budget.

Work. Each step is charged to the active budget before it runs, on the scale of mathforge.trusted.integers: one unit
plus one per 64-bit word of a candidate for its table lookup or gcd, and for each strong test what charged_pow would
charge for pow(base, m - 1, m) (a modular multiplication per bit of m - 1), with a long division for the gcd of a
factor witness; the integer kernel charges the certificate. Alongside, the producer adds up what the checker will
charge for the claim, by the checker's cost model: its scan, paid in advance at one unit plus one per 64-bit word of
the larger end of the range for each skipped integer; the proof of r (trusted.integers.prime_proof_work); and
certify.ntheory.factorization.charged_pow's price for every base-2 test and Fermat witness, a multiplication for a
factor witness. Like kernels.certificates, it refuses the claim with ResourceLimitError as soon as the check alone
would exceed max_work, whatever the production spent and whether or not a budget was active: the verifier checks in a
fresh budget of the request's limits (the lean runner in one of the default limits), so production and check need not
fit in max_work together, and the checker never runs out of budget on a claim that the producer made under the same
limits. With the default limits the checker's scan alone fills the budget at about 1,000 to 1,500 bits for a prime gap
of average length, and the producer, which pays for its own scan in its own budget, runs out of it at about the same
size; from psi_13 on, the factorization of r - 1 that the certificate needs usually stops it long before.
"""

from __future__ import annotations

from math import gcd, prod
from types import ModuleType

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

SPECS = (
    OperationSpec("next_prime", "ntheory", (Argument("n", "int", None, "int"),), "int",
                  "Find the smallest prime greater than an integer.",
                  "exact integers; certified primes, orders and residues; canonical integer solution families",
                  version=6),
    OperationSpec("prev_prime", "ntheory", (Argument("n", "int", None, "int"),), "int",
                  "Find the largest prime smaller than an integer.",
                  "exact integers; certified primes, orders and residues; canonical integer solution families",
                  version=6),
)

TYPES: dict[str, type] = {}  # Both values are plain ints.
CHECKER = "mathforge.certify.ntheory.prime_search"

_OPERATIONS = ("next_prime", "prev_prime")
# The primes up to 97. Their product is the checker's gcd screen, and which skipped integers need a witness depends
# on it, so the producer screens with exactly the same product. They are also the bases of the strong tests: all of
# them are prime to a candidate that passed the gcd screen.
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
_PRIMORIAL = prod(_SMALL_PRIMES)
_BASES = _SMALL_PRIMES[:13]  # 2 to 41: their strong tests decide every integer below psi_13 (OEIS A014233).
_LARGE_BASES = _SMALL_PRIMES[:2]  # From psi_13 on 2 and 3 screen, and prime_evidence decides (see the module).
_WITNESS_BASES = _SMALL_PRIMES[2:]  # 5 to 97: they choose a witness from psi_13 on, after 2 and 3.
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the evidence"


def produce(operation: str, args: dict[str, object]) -> tuple[int, tuple[tuple, tuple]]:
    """(r, (certificate, witnesses)) of next_prime(n) or prev_prime(n), found by the walk of the module docstring.

    InvalidInput for an n that is not an exact int, and for prev_prime with n <= 2. ResourceLimitError for an n or a
    candidate over max_integer_bits, for work past max_work, for a certificate that the budget cannot build, and for a
    claim whose check would need more than max_work (see the module)."""
    if operation not in _OPERATIONS:
        raise UnsupportedOperation(f"{operation} is not an operation of the prime_search family")
    n = args["n"]
    if type(n) is not int:  # Neither bool nor float is an exact integer.
        raise InvalidInput(f"{operation} requires an exact integer n")
    if operation == "prev_prime" and n <= 2:
        raise InvalidInput("prev_prime requires n >= 3: no prime is smaller than 2")
    from ...kernels import integers as kernel  # Kernels load on first use: importing mathforge loads none.
    from ...kernels.certificates import prime_evidence
    from ...trusted import integers as costs
    from ...trusted.primes import DETERMINISTIC_BELOW
    with computation() as budget:
        budget.check_int(n)
        if n < 2:  # Only next_prime gets here: no integer below 2 is prime, so 2 is the smallest prime above n.
            return 2, ((), ())
        upward, max_work = operation == "next_prime", budget.limits.max_work
        table = kernel.smallest_factor_table()
        witnesses: list[tuple[str, int, int]] = []
        skipped = check = 0  # The integers passed over, and what the checker pays for them beyond its scan units.

        def refuse_unless_checkable(top: int) -> None:
            """ResourceLimitError once the check of the integers passed over cannot fit in max_work, the checker's
            own fresh budget (as in kernels.certificates, the production is not counted): the checker's scan, at
            1 + (bits(top) >> 6) units each for top = r (next_prime) or n (prev_prime), then their base-2 tests and
            witnesses."""
            if skipped * (1 + (top.bit_length() >> 6)) + check > max_work:
                raise ResourceLimitError(_UNCHECKABLE.format(max_work))

        m = n
        while True:
            m += 1 if upward else -1
            refuse_unless_checkable(m if upward else n)  # For next_prime, r >= m: the units are at least these.
            budget.check_int(m)
            budget.tick(1 + (m.bit_length() >> 6))  # The table lookup, or the gcd with 2*3*...*97.
            if m < costs.TRIAL_BELOW:
                if table[m]:  # The smallest prime factor of the composite m: the checker's trial test agrees.
                    skipped += 1
                    continue
            elif gcd(m, _PRIMORIAL) > 1:  # A prime up to 97 < m divides m, as the checker's own gcd shows.
                skipped += 1
                continue
            else:
                # Below psi_13 the thirteen bases decide m. From psi_13 on 2 and 3 screen, and prime_evidence decides:
                # further bases would cost an exponentiation modulo m each on every value.
                large = m >= DETERMINISTIC_BELOW
                composite, witness = _screen(budget, costs, m, _LARGE_BASES if large else _BASES)
                if composite:
                    if large and witness is not None and witness[0] == "factor":
                        # A square root of 1 and no Fermat base yet: as below psi_13, a Fermat base comes first.
                        witness = _large_witness(budget, costs, m, witness)
                    skipped += 1
                    check += _pow_work(costs, 2, m - 1, m)  # The checker's own base-2 test of m.
                    if witness is not None:  # 2**(m - 1) = 1 modulo m: the checker needs the witness.
                        witnesses.append(witness)
                        check += _witness_work(costs, witness)
                    continue
            # No test showed m composite: below psi_13 it is prime, from psi_13 on prime_evidence decides.
            refuse_unless_checkable(m if upward else n)  # Before the certificate, the costly step from psi_13 on.
            try:
                certificate = prime_evidence(m)
            except InvalidInput:
                if m < DETERMINISTIC_BELOW:  # The table and the thirteen strong tests are exact there.
                    raise ArithmeticError("The integer kernel and the strong tests disagree about a prime") from None
                # A strong pseudoprime to 2 and 3, composite after all: the strong tests to 5 ... 97 give its witness,
                # else the integer kernel's certificate does.
                witness = _large_witness(budget, costs, m, None)
                if witness is None:  # A strong pseudoprime to every prime base up to 97.
                    witness = _kernel_witness(kernel, m)
                witnesses.append(witness)
                skipped += 1
                check += _pow_work(costs, 2, m - 1, m) + _witness_work(costs, witness)
                continue
            break
        work = skipped * (1 + ((m if upward else n).bit_length() >> 6)) + check  # The check, in a budget of its own.
        work += costs.prime_proof_work(m, certificate, max(0, max_work - work))
        if work > max_work:
            raise ResourceLimitError(_UNCHECKABLE.format(max_work))
        if not upward:
            witnesses.reverse()  # Met from n downwards; the checker takes them in increasing order of m.
        return m, (certificate, tuple(witnesses))


def _screen(budget: Budget, costs: ModuleType, m: int, bases: tuple[int, ...]) -> tuple[bool, tuple | None]:
    """How the strong probable-prime tests to bases, 2 first, sort an m >= 10,000 prime to 2*3*...*97, each charged
    before it runs: (True, None) when 2**(m - 1) is not 1 modulo m, which the checker shows itself; (True, witness)
    for another composite; (False, None) when m passes every test.

    The witness is ("fermat", m, w) for the first base w with w**(m - 1) not 1 modulo m, the kind that the card names
    first; when no base is one, ("factor", m, d) from the first test that met a square root of 1 other than 1 and
    -1. A composite m that passes the base-2 Fermat test fails some test unless it is a strong pseudoprime to every
    base, which below psi_13 no composite is for the bases 2 to 41."""
    exponent = m - 1
    shift = (exponent & -exponent).bit_length() - 1  # m - 1 = odd * 2**shift with shift >= 1, since m is odd.
    odd = exponent >> shift
    factor = None
    for base in bases:
        budget.tick(_pow_work(costs, base, exponent, m))
        failure = _strong_failure(budget, costs, m, odd, shift, base)
        if failure is None:
            continue
        if failure[0] == "fermat":
            return True, (None if base == 2 else failure)
        if factor is None:
            factor = failure
    return (True, factor) if factor is not None else (False, None)


def _large_witness(budget: Budget, costs: ModuleType, m: int, found: tuple | None) -> tuple | None:
    """The witness of a composite m >= psi_13 with 2**(m - 1) = 1 modulo m and 3**(m - 1) = 1 modulo m (the strong
    tests to 2 and 3 found no Fermat witness), chosen as the thirteen bases choose it below psi_13: ("fermat", m, w)
    for the first prime base w from 5 to 97 with w**(m - 1) not 1 modulo m; else found, the factor witness of the
    strong test to 2 or 3 when one met a square root of 1, else the first factor witness of the bases 5 to 97; None
    when m passes the strong test to each of them. Each test is charged before it runs."""
    _, witness = _screen(budget, costs, m, _WITNESS_BASES)  # 2 is not among the bases: no (True, None) verdict.
    if witness is not None and witness[0] == "fermat":
        return witness
    return found if found is not None else witness


def _strong_failure(budget: Budget, costs: ModuleType, m: int, odd: int, shift: int, base: int) -> tuple | None:
    """None when the odd m passes the strong probable-prime test to base, with m - 1 = odd * 2**shift (the caller
    has charged the exponentiation); otherwise the witness that the failure gives: ("fermat", m, base) when
    base**(m - 1) is not 1 modulo m, else ("factor", m, gcd(x - 1, m)) for the square root x of 1 modulo m, other
    than 1 and m - 1, that the test met. m divides (x - 1)(x + 1) and neither factor, so the gcd is a proper divisor."""
    x = pow(base, odd, m)
    if x == 1 or x == m - 1:
        return None
    for _ in range(shift - 1):
        y = x * x % m
        if y == m - 1:
            return None
        if y == 1:
            break  # x is a square root of 1 other than 1 and m - 1.
        x = y
    else:
        if x * x % m != 1:  # x * x = base**(m - 1).
            return ("fermat", m, base)
    bits = m.bit_length()
    budget.tick(1 + costs.division_work(bits, bits))  # The gcd, about a long division of two integers of m's size.
    return ("factor", m, gcd(x - 1, m))


def _pow_work(costs: ModuleType, base: int, exponent: int, modulus: int) -> int:
    """The work of pow(base, exponent, modulus): one unit, a multiplication modulo the modulus per bit of the
    exponent, and the reduction of the base. certify.ntheory.factorization.charged_pow charges exactly this, so it is
    also the checker's price of a base-2 test or of a Fermat witness; the producer charges its own strong tests so."""
    bits = modulus.bit_length()
    return 1 + exponent.bit_length() * costs.multiplication_work(bits) + costs.division_work(base.bit_length(), bits)


def _witness_work(costs: ModuleType, witness: tuple) -> int:
    """What the checker charges for a witness: the exponentiation of a Fermat witness through charged_pow, or the
    multiplication_work it pays before the division of a factor witness."""
    kind, m, number = witness
    if kind == "fermat":
        return _pow_work(costs, number, m - 1, m)
    return costs.multiplication_work(m.bit_length())


def _kernel_witness(kernel: ModuleType, m: int) -> tuple:
    """The witness of a composite m that passed the strong test to every base: the integer kernel's v0.7 certificate
    of m, ("fermat", m, w) or ("factor", m, d), which has the shape of a witness entry. The kernel charges the budget
    for it; ArithmeticError when it proves m prime after all, or gives anything else."""
    proven, certificate = kernel.prime_certificate(m)
    if (proven or len(certificate) != 3 or certificate[0] not in ("fermat", "factor") or certificate[1] != m
            or type(certificate[2]) is not int or not 1 < certificate[2] < m):
        raise ArithmeticError("The integer kernel gave no witness for a composite")
    return certificate
