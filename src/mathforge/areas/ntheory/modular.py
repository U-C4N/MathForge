"""Modular arithmetic of exact integers (mathforge.ntheory): modular powers, multiplicative orders, primitive roots,
the Legendre and Jacobi symbols and square roots modulo a prime.

Every argument is an exact int (MCP kind and annotation "int", so operations_v6.run already rejects a bool, a float,
a str and None as invalid input), and no argument has a default. The values are plain ints and tuples of ints: the
family defines no record class, so TYPES is empty and there is no RENDER (the shared renderer presents ints and
tuples). For each operation, its domain, its value and the evidence that its checker, mathforge.certify.ntheory.modular,
verifies:

- powmod(base, exponent, modulus): base**exponent modulo a modulus >= 1, the residue in [0, modulus); 0**0 = 1, and
  modulus 1 gives 0. A negative exponent raises the inverse of base, which must exist: gcd(base, modulus) = 1, else
  invalid input. Evidence ().
- multiplicative_order(a, n): the least d >= 1 with a**d = 1 modulo n >= 1, for gcd(a, n) = 1; n = 1 gives 1.
  Evidence (factors,): the certified factorization of d, () for d = 1, so ((),) then.
- primitive_root(p): the smallest generator g of (Z/p)^* for a prime p, 1 <= g < p; p = 2 gives 1. A p that is not
  prime is invalid input. Evidence (factors, refutations): the certified factorization of p - 1, and for each
  h = 2, ..., g - 1 in this order a prime q of p - 1 with h**((p - 1)/q) = 1 modulo p; ((), ()) for p = 2.
- legendre_symbol(a, p): (a/p), one of -1, 0 and 1, for an odd prime p (p = 2, p < 3 and composites are invalid
  input). Evidence (certificate,): the prime certificate of p.
- jacobi_symbol(a, n): (a/n), one of -1, 0 and 1, for an odd n >= 1; (a/1) = 1. Evidence ().
- sqrt_mod(a, p): the sorted tuple of every x in [0, p) with x*x = a modulo a prime p, in one of four shapes: (a % 2,)
  for p = 2, (0,) when p divides a, (r, p - r) with 0 < r < p - r for a quadratic residue and () for a non-residue.
  A p that is not prime is invalid input. Evidence (certificate,): the prime certificate of p.

A prime certificate is what mathforge.kernels.certificates.prime_evidence builds, () below psi_13 and a Pocklington
certificate from there on; a certified factorization is ((q, e, certificate), ...) with strictly increasing primes, as
factorization_evidence builds it. mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family,
documentation/families.md is the checklist, and tests/v09/ntheory/test_modular.py is the definition of done.

How produce() computes (docs/v09/cards/m1-ntheory.md). Nothing here is a proof: the checker verifies every value with
algorithms of its own (the builtin pow, Euler's criterion, least absolute remainders, Lucas's test).

- powmod reduces base modulo the modulus. For a negative exponent it first inverts that residue by the extended
  Euclidean algorithm, and a residue without an inverse is invalid input. It then raises the residue to |exponent| by
  a left-to-right binary square-and-multiply loop of its own; nothing in this module calls the builtin pow.
- multiplicative_order answers 1 for a = 1 modulo n, which covers n = 1, where every power is 1 = 0. Otherwise it
  factors n with the integer kernel and builds the factorization of the Carmichael function lambda(n), the least
  common multiple of lambda(p**k) = p**(k - 1) * (p - 1) over the prime powers p**k of n (the kernel factors each
  p - 1), with lambda(2) = 1, lambda(4) = 2 and lambda(2**k) = 2**(k - 2) from k = 3 on. The order of a unit divides
  lambda(n) = prod q**e. For each prime q of lambda(n), y = a**(lambda(n)/q**e) has the order q**f, where q**f is the
  exact power of q in the order of a: the cofactor lambda(n)/q**e has no factor q and holds every other prime to at
  least its power in the order. Raising y to the q-th power until it is 1, at most e times, finds f, and the order is
  the product of the q**f. This is the card's loop that divides each q out of d = lambda(n) while a**(d/q) = 1 (it
  divides q out e - f times), at the cost of one exponentiation by the cofactor per prime and short ones by q, where
  that loop paid an exponentiation by the whole of d for each division: a unit of order 2 modulo 2**1000 took it about
  a thousand exponentiations of a thousand bits, past the default budget, and takes this one about a thousand units.
- primitive_root refuses a p that the kernel's strong probable-prime tests call composite: they never call a prime
  composite, and they are exact below psi_13. It factors p - 1 with the kernel and tries g = 2, 3, ...: the first
  prime q of p - 1, in increasing order, with g**((p - 1)/q) = 1 refutes g and joins the refutations, and the first g
  that no prime refutes is the answer, once g**(p - 1) = 1 completes Lucas's test (otherwise p is a composite that
  fooled the strong tests: invalid input). Only a composite has no generator below p, so g = p is invalid input too.
- legendre_symbol and sqrt_mod first ask kernels.certificates.prime_evidence for the certificate of p, which refuses a
  p that is not prime: invalid input. legendre_symbol then runs the binary Jacobi algorithm below, which gives
  Legendre's symbol for a prime p.
- jacobi_symbol runs the binary algorithm of the card on a reduced modulo n: it removes the factors 2 of a, all at
  once, since each changes the sign exactly when n = 3 or 5 modulo 8 ((2/n) = -1 there), so the parity of their number
  decides; it swaps a and n by reciprocity, which changes the sign when both are 3 modulo 4, and reduces a modulo the
  new n, until a = 0. The symbol is the sign when n ends at 1, and 0 otherwise (a and n shared a prime).
- sqrt_mod answers (a % 2,) for p = 2 and (0,) when p divides a. Otherwise the Jacobi symbol, Legendre's for a prime p,
  separates a non-residue, (), from a residue r, whose roots are x and p - x for one root x: x = r**((p + 1)/4) for
  p = 3 modulo 4; else, for p - 1 = odd * 2**m, Tonelli and Shanks' algorithm with the least non-residue z = 2, 3, ...,
  which Jacobi symbols find, while its loop of up to m(m + 3)/2 multiplications stays within two multiplications per
  bit of p, and Cipolla's algorithm beyond, whose cost does not grow with m (both are in _square_root).

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call, refuses an argument longer than max_integer_bits
with ResourceLimitError, and charges every further step before it runs, on the scale of the cost model of
trusted.integers, so that every valid input costs at least two units. A long division costs division_work; the
square-and-multiply loop costs multiplication_work per squaring and per multiplication, all charged before it starts;
a step of the extended Euclidean algorithm costs a unit per 4,096 bits of the modulus, plus one (its division and its
update of the coefficient are both linear in that length, as the remainder and the coefficient never exceed it
together); a pass of the Jacobi algorithm costs a unit and one more per 4,096 bits for its linear operations, and
division_work of its quotient's length, mostly a word or two, for its long division (_jacobi states the measurement
behind it). The integer kernel and kernels.certificates charge their own work, and a certified factorization is refused
with ResourceLimitError when check_factorization would need more than max_work (trusted.integers.factorization_work),
whatever the production spent, the rule of kernels.certificates: the verifier checks in a fresh budget. Measured on
one machine with CPython, a unit of any of these takes at most about 0.3 microseconds (less on small numbers), the
checker's scale: a modular inverse modulo 65,536 bits takes about 650,000 units (0.16 s), 1.6 million on Fibonacci
numbers, the worst case (0.4 s), and a Jacobi symbol of 65,530 bits about 640,000 (0.1 s), 1.53 million in the binary
algorithm's worst case (0.27 s), so all of them fit the default max_work of 2,000,000.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

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
_SCOPE = "exact integers; certified primes, orders and residues; canonical integer solution families"

SPECS = (
    OperationSpec("powmod", "ntheory",
                  (Argument("base", "int", None, "int"), Argument("exponent", "int", None, "int"),
                   Argument("modulus", "int", None, "int")), "int",
                  "Raise an integer to an integer power modulo a positive modulus; negative powers use the inverse.",
                  _SCOPE, version=6),
    OperationSpec("multiplicative_order", "ntheory",
                  (Argument("a", "int", None, "int"), Argument("n", "int", None, "int")), "int",
                  "Compute the multiplicative order of an integer modulo a positive modulus.", _SCOPE, version=6),
    OperationSpec("primitive_root", "ntheory",
                  (Argument("p", "int", None, "int"),),
                  "int", "Find the smallest primitive root modulo a prime.", _SCOPE, version=6),
    OperationSpec("legendre_symbol", "ntheory",
                  (Argument("a", "int", None, "int"), Argument("p", "int", None, "int")),
                  "int", "Compute the Legendre symbol (a/p) for an odd prime p.", _SCOPE, version=6),
    OperationSpec("jacobi_symbol", "ntheory",
                  (Argument("a", "int", None, "int"), Argument("n", "int", None, "int")),
                  "int", "Compute the Jacobi symbol (a/n) for an odd positive n.", _SCOPE, version=6),
    OperationSpec("sqrt_mod", "ntheory",
                  (Argument("a", "int", None, "int"), Argument("p", "int", None, "int")),
                  "tuple[int, ...]", "List every square root of an integer modulo a prime.", _SCOPE, version=6),
)

# No record classes: every value is an int or a tuple of ints, which serialize and render without one.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.ntheory.modular"

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
_NOT_INVERTIBLE = "base is not invertible modulo modulus: a negative exponent needs gcd(base, modulus) = 1"
_NOT_A_UNIT = "multiplicative_order requires gcd(a, n) = 1: a is not invertible modulo n"
_NOT_PRIME = "{} requires a prime p"
_NOT_ODD_PRIME = "legendre_symbol requires an odd prime p"
_UNCHECKABLE = "Computation limit exceeded: work ({}) for the independent check of the evidence"


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(value, evidence) of an operation of the family, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithms and the evidence. InvalidInput for an argument that is no exact int or
    lies outside the operation's domain, ResourceLimitError past the budget."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the modular family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    values = [args[name] for name in names]
    if any(type(value) is not int for value in values):  # Neither a bool nor a float nor a str is an exact integer.
        raise InvalidInput(f"{operation} requires exact integers: {', '.join(names)}")
    with computation() as budget:
        budget.tick()  # The call itself; on a valid input every operation charges at least one unit more.
        for value in values:
            budget.check_int(value)
        return _PRODUCERS[operation](budget, *values)


def _powmod(budget: Budget, base: int, exponent: int, modulus: int) -> tuple[int, tuple]:
    """base**exponent modulo modulus and the evidence (); a negative exponent raises the inverse of base."""
    if modulus < 1:
        raise InvalidInput("powmod requires a modulus >= 1")
    from ...trusted.integers import multiplication_work

    residue = _reduced(budget, base, modulus)
    if exponent < 0:
        inverse = _inverse(budget, residue, modulus)
        if inverse is None:
            raise InvalidInput(_NOT_INVERTIBLE)
        residue, exponent = inverse, -exponent
    return _power(budget, residue, exponent, modulus, multiplication_work(modulus.bit_length())), ()


def _multiplicative_order(budget: Budget, a: int, n: int) -> tuple[int, tuple]:
    """The order d of the unit a modulo n and the evidence (factors,), the certified factorization of d."""
    if n < 1:
        raise InvalidInput("multiplicative_order requires a modulus n >= 1")
    residue = _reduced(budget, a, n)
    words = (n.bit_length() >> 6) + 1
    budget.tick(1 + (words * words >> 6))  # math.gcd, Lehmer's algorithm: quadratic in the length of n.
    if gcd(residue, n) != 1:
        raise InvalidInput(_NOT_A_UNIT)
    if residue == 1 % n:  # 1 has the order 1, and so has every a modulo n = 1, where 1 = 0.
        return 1, ((),)
    from ...trusted.integers import division_work, factorization_entry_work, multiplication_work

    exponents = _carmichael(budget, n)  # lambda(n) as {q: e}; the order divides it.
    carmichael = 1
    for q, e in exponents.items():
        budget.tick(factorization_entry_work(carmichael.bit_length(), q.bit_length() * e))  # q**e and its product.
        carmichael *= q ** e
    work = multiplication_work(n.bit_length())
    order, pairs = 1, []
    for q in sorted(exponents):
        e = exponents[q]
        # q**e, then the cofactor lambda(n) / q**e, which holds every other prime of lambda(n) to its full power.
        budget.tick(factorization_entry_work(0, q.bit_length() * e)
                    + 1 + division_work(carmichael.bit_length(), q.bit_length() * e))
        power = _power(budget, residue, carmichael // q ** e, n, work)  # Of the order q**f, q's power in ord(a).
        f = 0
        while power != 1:  # n >= 2 here, so 1 is the residue 1 % n.
            if f == e:  # a**lambda(n) = 1 for every unit a: only a wrong factorization of lambda(n) gets here.
                raise ArithmeticError("The Carmichael function is no multiple of the order")
            power = _power(budget, power, q, n, work)
            f += 1
        if f:
            budget.tick(factorization_entry_work(order.bit_length(), q.bit_length() * f))  # q**f and its product.
            order *= q ** f
            pairs.append((q, f))
    return order, (_certified(budget, tuple(pairs)),)


def _primitive_root(budget: Budget, p: int) -> tuple[int, tuple]:
    """The smallest generator g of (Z/p)^* and the evidence (factors of p - 1, refutations of 2, ..., g - 1)."""
    if p < 2:
        raise InvalidInput(_NOT_PRIME.format("primitive_root"))
    budget.tick()  # The primality decision, whose strong tests the kernel charges itself above 2**16.
    if p == 2:  # (Z/2)^* = {1}, which 1 generates: nothing to factor, nothing to refute.
        return 1, ((), ())
    from ...kernels.integers import factor, is_probable_prime
    from ...trusted.integers import division_work, multiplication_work

    if not is_probable_prime(p):  # The strong tests never call a prime composite.
        raise InvalidInput(_NOT_PRIME.format("primitive_root"))
    order = p - 1
    pairs = factor(order)  # Increasing primes; the kernel charges its work to the budget.
    cofactors = []
    for q, _ in pairs:
        budget.tick(1 + division_work(order.bit_length(), q.bit_length()))
        cofactors.append((q, order // q))
    work = multiplication_work(p.bit_length())
    refutations: list[int] = []
    root = 2
    while True:
        if root == p:  # A prime has a generator below it, so p is composite.
            raise InvalidInput(_NOT_PRIME.format("primitive_root"))
        budget.tick()
        refuting = 0
        for q, cofactor in cofactors:
            if _power(budget, root, cofactor, p, work) == 1:  # The order of root divides (p - 1)/q < p - 1.
                refuting = q
                break
        if not refuting:
            break
        refutations.append(refuting)
        root += 1
    if _power(budget, root, order, p, work) != 1:  # Fermat's little theorem fails: p is composite after all.
        raise InvalidInput(_NOT_PRIME.format("primitive_root"))
    return root, (_certified(budget, pairs), tuple(refutations))


def _legendre_symbol(budget: Budget, a: int, p: int) -> tuple[int, tuple]:
    """(a/p) for an odd prime p, by the binary Jacobi algorithm, and the evidence (certificate of p,)."""
    if p < 3 or not p & 1:
        raise InvalidInput(_NOT_ODD_PRIME)
    certificate = _certificate(p, _NOT_ODD_PRIME)
    return _jacobi(budget, a, p), (certificate,)


def _jacobi_symbol(budget: Budget, a: int, n: int) -> tuple[int, tuple]:
    """(a/n) for an odd n >= 1, by the binary algorithm, and the evidence ()."""
    if n < 1 or not n & 1:
        raise InvalidInput("jacobi_symbol requires an odd n >= 1")
    return _jacobi(budget, a, n), ()


def _sqrt_mod(budget: Budget, a: int, p: int) -> tuple[tuple[int, ...], tuple]:
    """Every square root of a modulo the prime p, sorted, and the evidence (certificate of p,)."""
    if p < 2:
        raise InvalidInput(_NOT_PRIME.format("sqrt_mod"))
    certificate = _certificate(p, _NOT_PRIME.format("sqrt_mod"))
    residue = _reduced(budget, a, p)
    if p == 2 or not residue:  # Modulo 2, x*x = x; modulo a prime, x*x = 0 forces x = 0.
        return (residue,), (certificate,)
    if _jacobi(budget, residue, p) != 1:  # Legendre's symbol for a prime p: -1, a non-residue without roots.
        return (), (certificate,)
    root = _square_root(budget, residue, p)
    low = min(root, p - root)  # root and p - root differ, as p is odd and root is not 0.
    return (low, p - low), (certificate,)


_PRODUCERS: dict[str, Callable[..., tuple[object, tuple]]] = {
    "powmod": _powmod,
    "multiplicative_order": _multiplicative_order,
    "primitive_root": _primitive_root,
    "legendre_symbol": _legendre_symbol,
    "jacobi_symbol": _jacobi_symbol,
    "sqrt_mod": _sqrt_mod,
}


def _reduced(budget: Budget, number: int, modulus: int) -> int:
    """number % modulus, in [0, modulus) for a modulus >= 1, its long division charged before it runs."""
    from ...trusted.integers import division_work

    budget.tick(1 + division_work(number.bit_length(), modulus.bit_length()))
    return number % modulus


def _power(budget: Budget, base: int, exponent: int, modulus: int, work: int) -> int:
    """base**exponent modulo modulus, for 0 <= base < modulus, exponent >= 0 and modulus >= 1, by left-to-right binary
    square-and-multiply: never the builtin pow. work is multiplication_work of the modulus's bit length, the charge of
    each squaring (one per bit after the leading one) and each multiplication (one per set bit after it), all charged
    before the loop starts."""
    if not exponent:
        budget.tick()
        return 1 % modulus
    budget.tick(1 + (exponent.bit_length() + exponent.bit_count() - 2) * work)
    result = base
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        result = result * result % modulus
        if bit == "1":
            result = result * base % modulus
    return result


def _inverse(budget: Budget, a: int, m: int) -> int | None:
    """The inverse of a modulo m in [0, m), for 0 <= a < m and m >= 1, by the extended Euclidean algorithm; None when
    gcd(a, m) > 1. Each step is charged before it runs, one unit and one per 4,096 bits of m: its long division and its
    update of the coefficient are linear in the length of m (the first quotient of a short a is long, but then the
    remainder is short; afterwards a remainder and the coefficient beside it never exceed m together)."""
    step = 1 + (m.bit_length() >> 12)
    r0, r1, s0, s1 = m, a, 0, 1  # r_i = s_i * a modulo m throughout.
    while r1:
        budget.tick(step)
        q, r = divmod(r0, r1)
        r0, r1, s0, s1 = r1, r, s1, s0 - q * s1
    return s0 % m if r0 == 1 else None  # r0 = gcd(a, m) = s0 * a modulo m.


def _jacobi(budget: Budget, a: int, n: int) -> int:
    """(a/n) for an odd n >= 1 by the binary algorithm (the module states it). Each pass is charged before it runs:
    its few linear operations on n, a unit per 4,096 bits, and the long division by division_work of the quotient's
    length (mostly a word or two; division_work by the dividend's whole length would charge a quotient as long as n).

    The linear charge is a quarter of the checker's per pass, and this is measured, not a discount: a pass here
    (the lowest set bit, one shift, the swap) takes about a quarter of the time of a pass of the checker's least
    absolute remainders (doubling, comparing, subtracting, the lowest set bit, the shift), about 0.4 microseconds plus
    0.14 per 1,000 bits, so that a unit stays near 0.2 microseconds at every length from 1,000 to 65,536 bits. The
    binary algorithm needs up to about 0.91 passes per bit (quotients 1 and 2 in turn, even and odd remainders
    alternating), the checker's at most about 0.79: at the checker's charge per pass, the worst case of 65,535 bits
    ran out of the default budget after 0.17 seconds, where the checker verifies the symbol with 1.5 million units."""
    from ...trusted.integers import division_work

    budget.tick(1 + division_work(a.bit_length(), n.bit_length()))
    a %= n
    sign = 1
    while a:
        budget.tick(1 + (n.bit_length() >> 12))  # Removing the factors 2, the residues modulo 4 and 8, the swap.
        shift = (a & -a).bit_length() - 1
        a >>= shift
        if shift & 1 and (n & 7) in (3, 5):  # (2/n)**shift, and (2/n) = -1 exactly when n = 3 or 5 modulo 8.
            sign = -sign
        a, n = n, a  # (a/n) = (n/a), up to the sign of reciprocity, for odd coprime a and n.
        if (a & 3) == 3 and (n & 3) == 3:
            sign = -sign
        budget.tick(1 + division_work(a.bit_length() - n.bit_length() + 64, n.bit_length()))
        a %= n
    return sign if n == 1 else 0  # n ends at gcd(a, n): above 1, the symbol is 0.


def _square_root(budget: Budget, residue: int, p: int) -> int:
    """A square root of the quadratic residue 0 < residue < p modulo the odd prime p: residue**((p + 1)/4) for p = 3
    modulo 4. Otherwise p - 1 = odd * 2**m with m >= 2, and the loop of Tonelli and Shanks' algorithm takes up to
    m(m + 3)/2 multiplications beyond its three powers, Cipolla's algorithm about five and a half per bit of p whatever
    m is: Tonelli and Shanks while that loop stays within two multiplications per bit of p, which covers most primes,
    Cipolla beyond (at the prime 13 * 2**1000 + 1 that loop ran past the default budget; Cipolla takes 33,000 units).
    Every multiplication modulo p is charged before it runs."""
    from ...trusted.integers import multiplication_work

    work = multiplication_work(p.bit_length())
    if (p & 3) == 3:  # (residue**((p + 1)/4))**2 = residue * residue**((p - 1)/2) = residue, by Euler's criterion.
        return _power(budget, residue, (p + 1) >> 2, p, work)
    shift = ((p - 1) & (1 - p)).bit_length() - 1  # p - 1 = odd * 2**shift, with shift >= 2.
    if shift * (shift + 3) > 4 * p.bit_length():
        return _cipolla(budget, residue, p, work)
    return _tonelli_shanks(budget, residue, p, shift, work)


def _cipolla(budget: Budget, residue: int, p: int, work: int) -> int:
    """A square root of the quadratic residue 0 < residue < p modulo the odd prime p, by Cipolla's algorithm. For the
    least t >= 1 with w = t*t - residue a non-residue (for half of all t it is), F_p[s]/(s*s - w) is the field of p*p
    elements, where s**p = s * w**((p - 1)/2) = -s, so (t + s)**p = t - s and (t + s)**(p + 1) = t*t - w = residue: its
    square root (t + s)**((p + 1)/2) lies in F_p, where residue has its two square roots. work is multiplication_work
    of p's bit length; every step is charged before it runs."""
    t = 0
    while True:
        t += 1  # t stays below p: half of the t in [1, p) give a non-residue.
        budget.tick(1 + work)  # t*t - residue, reduced modulo p.
        w = (t * t - residue) % p
        if _jacobi(budget, w, p) == -1:  # Legendre's symbol for the prime p.
            break
    exponent = (p + 1) >> 1
    # Left-to-right square-and-multiply on x0 + x1*s, all charged before the loop starts: a squaring takes four
    # multiplications modulo p (at most), a multiplication by t + s one long and two short ones (three, charged).
    budget.tick(1 + (4 * (exponent.bit_length() - 1) + 3 * (exponent.bit_count() - 1)) * work)
    x0, x1 = t, 1
    for bit in bin(exponent)[3:]:  # The bits after the leading 1, from the top.
        x0, x1 = (x0 * x0 + w * (x1 * x1 % p)) % p, 2 * x0 * x1 % p  # (x0 + x1 s)**2, as s*s = w.
        if bit == "1":
            x0, x1 = (t * x0 + w * x1) % p, (x0 + t * x1) % p  # (x0 + x1 s)(t + s).
    if x1:  # The power lies in F_p for a prime p and a residue; only a producer bug gets here.
        raise ArithmeticError("Cipolla's power left F_p")
    return x0


def _tonelli_shanks(budget: Budget, residue: int, p: int, shift: int, work: int) -> int:
    """A square root of the quadratic residue 0 < residue < p modulo the odd prime p = odd * 2**shift + 1, shift >= 2,
    by Tonelli and Shanks' algorithm with the least non-residue z = 2, 3, ... work is multiplication_work of p's bit
    length; every multiplication modulo p is charged before it runs."""
    odd = (p - 1) >> shift
    z = 2
    while _jacobi(budget, z, p) != -1:  # The least quadratic non-residue, which exists for an odd prime.
        z += 1
    # Invariants: x*x = residue * t, the order of t divides 2**(m - 1), and c has the order 2**m exactly.
    m = shift
    c = _power(budget, z, odd, p, work)
    t = _power(budget, residue, odd, p, work)
    x = _power(budget, residue, (odd + 1) >> 1, p, work)
    while t != 1:
        i, square = 0, t  # The least i with t**(2**i) = 1; it is at least 1 and below m.
        while square != 1:
            if i == m - 1:  # t would have the order 2**m: residue was no residue, which the caller rules out.
                raise ArithmeticError("Tonelli-Shanks met a quadratic non-residue")
            budget.tick(work)
            square = square * square % p
            i += 1
        budget.tick((m - i + 2) * work)  # m - i - 1 squarings, then three multiplications.
        b = c
        for _ in range(m - i - 1):
            b = b * b % p  # b = c**(2**(m - i - 1)), of the order 2**(i + 1).
        m = i
        c = b * b % p
        t = t * c % p
        x = x * b % p
    return x


def _carmichael(budget: Budget, n: int) -> dict[int, int]:
    """The prime factorization {q: e} of the Carmichael function lambda(n) of n >= 2: the least common multiple of
    lambda(p**k) = p**(k - 1) * (p - 1) over the prime powers p**k of n, and lambda(2) = 1, lambda(4) = 2,
    lambda(2**k) = 2**(k - 2) from k = 3 on. The integer kernel factors n and each p - 1, charging the budget."""
    from ...kernels.integers import factor

    exponents: dict[int, int] = {}
    for p, k in factor(n):
        budget.tick()
        if p == 2:
            part: tuple[tuple[int, int], ...] = () if k == 1 else ((2, 1),) if k == 2 else ((2, k - 2),)
        else:
            part = factor(p - 1) + (((p, k - 1),) if k > 1 else ())  # p does not divide p - 1.
        for q, e in part:
            if e > exponents.get(q, 0):
                exponents[q] = e
    return exponents


def _certificate(p: int, message: str) -> tuple:
    """prime_evidence(p), the certificate that the checker's prime_proof accepts; InvalidInput(message) when p is not
    prime. ResourceLimitError when the budget cannot certify p, or the check of the certificate would exceed it."""
    from ...kernels.certificates import prime_evidence

    try:
        return prime_evidence(p)
    except InvalidInput as error:  # p is an exact int, so prime_evidence refuses only a p that is not prime.
        raise InvalidInput(message) from error


def _certified(budget: Budget, pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, tuple], ...]:
    """((q, e, certificate), ...) for a prime factorization ((q, e), ...) with increasing primes that the integer
    kernel found, each certificate from prime_evidence: the factors are not searched for twice, as a call of
    factorization_evidence would. ResourceLimitError when check_factorization would need more than max_work, as
    factorization_evidence refuses it: the verifier checks in a fresh budget of the same limits, so whatever the
    production spent, and whether or not a budget was active, the check may use all of max_work."""
    from ...kernels.certificates import prime_evidence
    from ...trusted.integers import factorization_work

    entries = []
    for q, e in pairs:
        try:
            certificate = prime_evidence(q)
        except InvalidInput as error:  # factor() took a composite for a prime: a producer bug, never an input error.
            raise ArithmeticError("Internal factorization produced a composite factor") from error
        entries.append((q, e, certificate))
    factors = tuple(entries)
    allowance = budget.limits.max_work  # The check runs in a budget of its own (kernels.certificates).
    if factorization_work(factors, allowance) > allowance:
        raise ResourceLimitError(_UNCHECKABLE.format(budget.limits.max_work))
    return factors
