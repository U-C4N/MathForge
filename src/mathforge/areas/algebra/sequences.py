"""Polynomial sequences over the exact rationals (mathforge.algebra): the Sturm sequence of a rational polynomial and
the n-th cyclotomic polynomial.

sturm_sequence(polynomial) -> tuple[Polynomial, ...]
    Domain: a nonzero Polynomial (kind "polynomial"): a Symbol variable and a nonempty tuple of at most 1,025 Rationals
    in lowest terms with positive denominators, without a trailing zero. The zero polynomial is invalid input.
    Value: the canonical Sturm sequence, with no scaling: p0 = p, p1 = p', and p(i+1) = -rem(p(i-1), p(i)), the exact
    remainder over Q, while it is nonzero; the sequence ends with its last nonzero member. A constant p gives (p,), a
    linear p gives (p, p'). Every member is a canonical v0.7 mathforge.polynomial.Polynomial in p's variable itself.
    Known values: x^2 - 2 -> (x^2 - 2, 2x, 2); x^3 - x -> (x^3 - x, 3x^2 - 1, 2/3 x, 1); 5 -> (5,); x - 3 -> (x - 3, 1).
    Evidence: the quotients (q1, ..., q(L-1)) for a sequence of L members p0, ..., p(L-1), one per division
    p(i-1) = q(i) p(i) - p(i+1) with p(L) = 0 for the last one; each q(i) is a tuple of canonical Rationals (no int),
    its coefficients lowest degree first, without a trailing zero, of length deg p(i-1) - deg p(i) + 1. A constant p has
    the evidence (). For x^2 - 2: ((Rational(0), Rational(1, 2)), (Rational(0), Rational(1))).

cyclotomic_polynomial(n, variable) -> Polynomial
    Domain: n an exact int >= 1 (kind "int"), variable a Symbol (kind "symbol"). n < 1 is invalid input; n > 100,000 or
    phi(n) > 1,024 exceeds the resource limit, phi(n) decided from the factorization of n before anything is built.
    Value: Phi_n, the n-th cyclotomic polynomial, as a Polynomial in variable: integer coefficients (Rationals with
    denominator 1), monic, of degree phi(n). Known values: 1 -> x - 1, 2 -> x + 1, 6 -> x^2 - x + 1,
    12 -> x^4 - x^2 + 1; Phi_105 has the coefficient -2 at x^7 and at x^41.
    Evidence: ((d, coefficients of Phi_d), ...) for every proper divisor d of n, d strictly increasing, each coefficient
    tuple of exact ints (no Rational), lowest degree first, without a trailing zero. n = 1 has the evidence ().

The family defines no record class: its values are v0.7 Polynomial records and tuples of them, which the shared renderer
presents (the Sturm sequence of x^2 - 2 as the text "(x^2 - 2, 2*x, 2)" and the LaTeX
"\\left(x^{2} - 2, 2\\, x, 2\\right)": the members separated by ", "), so TYPES is empty and there is no RENDER.

The checker, mathforge.certify.algebra.sequences, states the theorems and the exact layouts that it accepts.
docs/v09/cards/m7-extended.md is the specification; mathforge.areas.ntheory.continued_fraction spells out the contract
of a v6 family, documentation/families.md is the checklist, and tests/v09/algebra/test_sequences.py is the
definition of done.

How produce() computes. Nothing here is a proof: the checker verifies the division identities (sturm_sequence) and
Moebius products at a Kronecker point (cyclotomic_polynomial), never these algorithms.

- The inputs. produce() validates every argument itself with exact type tests, also those whose kinds
  operations_v6.run checks (run() admits every object of the Polynomial and Symbol classes, also one made with
  object.__new__). The polynomial must be a Polynomial whose variable passes the Symbol test below and whose
  coefficients are a nonempty tuple of at most 1,025 Rationals of exact ints in lowest terms with positive
  denominators, without a trailing zero, and not the zero polynomial; n must be an exact int; the variable must be a
  Symbol whose fields, read from the instance itself as the checker reads them, have their exact types (a str name, a
  Domain, a tuple of at most three Assumptions with str names and Truth values, a str uid). Anything else is invalid
  input; an integer longer than max_integer_bits exceeds the resource limit.
- sturm_sequence. v0.7's root_isolation.sturm_sequence (kernels.upoly.sturm_chain) makes the square-free part primitive
  and every member primitive with its sign changed as pseudo-remainders need, so it is not the card's canonical form;
  kernels.upoly.divmod_rational pseudo-divides, which multiplies the whole remainder by the divisor's leading integer
  once per quotient term (a degree drop of 999 by a divisor 999 x makes that a thousand passes over a thousand
  coefficients). produce() runs the card's Euclidean algorithm itself, over exact rationals held as reduced
  (numerator, denominator) pairs: p0 = p, p1 = p' (term by term, k * a/b reduced by gcd(k, b)), then each division
  of p(i-1) by p(i) by schoolbook long division against the monic divisor p(i) / lc(p(i)): the quotient term is the
  remainder's top coefficient divided by lc(p(i)), and only the divisor's nonzero coefficients update the remainder,
  so sparse members cost what their nonzero terms cost. Each sum a - c * g is formed over the product of the
  denominators and reduced by one gcd. The quotients are the evidence, the negated remainders the next members, until
  a remainder vanishes; p0 is the request's own record and every later member a Polynomial in its variable.
- cyclotomic_polynomial. n is factored by trial division (n <= 100,000), phi(n) follows from its primes, and n > 100,000
  or phi(n) > 1,024 is refused with ResourceLimitError before any polynomial is built. Phi_d is built for every divisor
  d of n in increasing order from those already built: Phi_1 = x - 1; Phi_d(x) = Phi_r(x^(d/r)) for the radical r < d
  of d; and for a square-free d = m p with p its largest prime, Phi_d(x) = Phi_m(x^p) / Phi_m(x), an exact division by a
  monic integer polynomial (trusted.upoly.exact_quotient). The proper divisors' polynomials are the evidence.

Work. produce() runs in the active computation budget (operations_v6.run opens the request's) or in one of the default
limits opened for a direct call. It charges one unit for the call and one for each coefficient it validates, and every
further step before it runs, on the scale of trusted.integers and trusted.upoly: a rational operation a - c * g costs a
unit, product_work for each of its five products and a gcd's quadratic cost (_gcd_work) for its reduction; a member
or quotient coefficient longer than max_integer_bits, which no Rational of the request's limits holds and the checker
would refuse, ends the request with ResourceLimitError. The unscaled Sturm sequence of a dense polynomial grows its
coefficients quickly: a random integer polynomial of degree 40 with one-digit coefficients takes some 200,000 units
(module test), and the checker exhausts its default 2 million around degree 55 (its own report), where run() answers
resource_limit_exceeded. cyclotomic_polynomial costs a unit per coefficient copied and trusted.upoly's charge for each
exact division: no n with phi(n) <= 1,024 takes more than about 220,000 units (n = 4,290, measured 218,207).
"""

from __future__ import annotations

from math import gcd

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
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

# The scope text of the algebra families of v0.9 (areas/algebra/roots.py), extended by this family's objects.
SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets, "
         "decimal digits, Sturm sequences and cyclotomic polynomials")

SPECS = (
    OperationSpec("sturm_sequence", "algebra",
                  (Argument("polynomial", "Polynomial", None, "polynomial"),),
                  "tuple[Polynomial, ...]", "Compute the Sturm sequence of a nonzero rational polynomial.", SCOPE,
                  version=6),
    OperationSpec("cyclotomic_polynomial", "algebra",
                  (Argument("n", "int", None, "int"), Argument("variable", "Symbol", None, "symbol")),
                  "Polynomial", "Compute the n-th cyclotomic polynomial.", SCOPE, version=6),
)

# No record classes: the values are v0.7 Polynomial records and tuples of them, which the registry does not let a v6
# family claim as its own (every TYPES class must be defined in the family module).
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.algebra.sequences"


# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {spec.name: tuple(argument.name for argument in spec.arguments)
                                          for spec in SPECS}
MAX_ORDER = 100_000  # The card's resource limit on n for cyclotomic_polynomial.
_MAX_ASSUMPTIONS = 3  # A Symbol holds at most one Assumption of each name: nonzero, positive and nonnegative.
_MISSING = object()  # A field that a record forged with object.__new__ lacks.
_NOT_A_POLYNOMIAL = "sturm_sequence requires a Polynomial"
_NOT_CANONICAL = ("sturm_sequence requires a canonical Polynomial: a Symbol variable and a nonempty tuple of at most "
                  "1,025 Rationals in lowest terms with positive denominators, without a trailing zero")
_ZERO_POLYNOMIAL = "sturm_sequence requires a nonzero polynomial: the zero polynomial has no Sturm sequence"
_NOT_A_SYMBOL = "{} requires a Symbol variable"
_TOO_LONG = "Computation limit exceeded: integer bits"

Pair = tuple[int, int]  # (numerator, denominator) of an exact rational in lowest terms, denominator >= 1.
_ZERO: Pair = (0, 1)


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple[object, ...]]:
    """(value, evidence) of sturm_sequence or cyclotomic_polynomial, in the layouts of the module docstring, for
    arguments that operations_v6.run has checked by name. InvalidInput for an argument of the wrong type or outside the
    operation's domain, UnsupportedOperation for another operation, ResourceLimitError past the budget and for the
    card's limits on n."""
    names = _ARGUMENTS.get(operation) if type(operation) is str else None
    if names is None:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the sequences family")
    if type(args) is not dict or set(args) != set(names):
        raise InvalidInput(f"{operation} takes exactly the arguments {', '.join(names)}")
    with computation() as budget:
        budget.tick()  # The call itself; the validation charges at least one unit more.
        if operation == "sturm_sequence":
            return _sturm_sequence(budget, args["polynomial"])
        return _cyclotomic_polynomial(budget, args["n"], args["variable"])


# ---------------------------------------------------------------------------------------------------------------
# Validation

def _field(record: object, name: str) -> object:
    """A field of a record read from the instance itself, _MISSING when it does not hold it (a record forged with
    object.__new__ falls back on class defaults through getattr, which the checker refuses to read)."""
    try:
        fields = vars(record)
    except TypeError:  # No __dict__.
        return _MISSING
    return fields.get(name, _MISSING) if type(fields) is dict else _MISSING


def _plain_symbol(variable: object) -> bool:
    """Whether variable is a Symbol whose own fields have their exact types, the test of the checker."""
    if type(variable) is not Symbol:
        return False
    name, domain = _field(variable, "name"), _field(variable, "domain")
    assumptions, uid = _field(variable, "assumptions"), _field(variable, "uid")
    if (type(name) is not str or type(domain) is not Domain or type(uid) is not str or type(assumptions) is not tuple
            or len(assumptions) > _MAX_ASSUMPTIONS):
        return False
    return all(type(item) is Assumption and type(_field(item, "name")) is str and type(_field(item, "truth")) is Truth
               for item in assumptions)


def _gcd_work(bits: int) -> int:
    """The units of a gcd of integers of at most this length: quadratic (Lehmer), as trusted.upoly.charge_gcd."""
    words = bits >> 6
    return words * words >> 6


def _polynomial(budget: Budget, polynomial: object) -> tuple[Polynomial, list[Pair]]:
    """The polynomial argument and its coefficients as pairs, once it is a canonical nonzero Polynomial (module
    docstring); InvalidInput otherwise, ResourceLimitError for an integer longer than max_integer_bits."""
    if type(polynomial) is not Polynomial:
        raise InvalidInput(_NOT_A_POLYNOMIAL)
    coefficients, variable = _field(polynomial, "coefficients"), _field(polynomial, "variable")
    if not _plain_symbol(variable):
        raise InvalidInput(_NOT_A_SYMBOL.format("sturm_sequence"))
    if type(coefficients) is not tuple or not coefficients or len(coefficients) > MAX_POLYNOMIAL_DEGREE + 1:
        raise InvalidInput(_NOT_CANONICAL)
    pairs = []
    for coefficient in coefficients:
        budget.tick()
        if type(coefficient) is not Rational:  # Neither an int nor a float nor a Rational subclass.
            raise InvalidInput(_NOT_CANONICAL)
        numerator, denominator = getattr(coefficient, "numerator", None), getattr(coefficient, "denominator", None)
        if type(numerator) is not int or type(denominator) is not int or denominator < 1:
            raise InvalidInput(_NOT_CANONICAL)
        budget.check_int(numerator)
        budget.check_int(denominator)
        if denominator != 1:
            budget.tick(_gcd_work(max(numerator.bit_length(), denominator.bit_length())))
            if gcd(numerator, denominator) != 1:
                raise InvalidInput(_NOT_CANONICAL)
        pairs.append((numerator, denominator))
    if not pairs[-1][0]:
        raise InvalidInput(_ZERO_POLYNOMIAL if len(pairs) == 1 else _NOT_CANONICAL)
    return polynomial, pairs


# ---------------------------------------------------------------------------------------------------------------
# sturm_sequence

def _sturm_sequence(budget: Budget,
                    argument: object) -> tuple[tuple[Polynomial, ...], tuple[tuple[Rational, ...], ...]]:
    """The canonical Sturm sequence and its quotients (module docstring)."""
    polynomial, p = _polynomial(budget, argument)
    if len(p) == 1:
        return (polynomial,), ()
    variable = polynomial.variable
    members = [p, _derivative(budget, p)]
    quotients = []
    while True:
        quotient, remainder = _divide(budget, members[-2], members[-1])
        quotients.append(_rationals(budget, quotient))
        if not remainder:
            break
        members.append([(-numerator, denominator) for numerator, denominator in remainder])
    value = (polynomial,) + tuple(Polynomial(_rationals(budget, member), variable) for member in members[1:])
    return value, tuple(quotients)


def _derivative(budget: Budget, pairs: list[Pair]) -> list[Pair]:
    """The derivative of a polynomial of degree >= 1: k * a/b reduced is (k/g) a / (b/g) with g = gcd(k, b), in lowest
    terms as gcd(a, b) = 1 and gcd(k/g, b/g) = 1."""
    from ...trusted.integers import division_work, product_work

    result = []
    for index in range(1, len(pairs)):
        numerator, denominator = pairs[index]
        budget.tick(2 + division_work(denominator.bit_length(), index.bit_length())
                    + product_work(numerator.bit_length(), index.bit_length()))
        divisor = gcd(index, denominator)
        result.append(((index // divisor) * numerator, denominator // divisor))
    return result


def _bits(pair: Pair) -> int:
    return max(pair[0].bit_length(), pair[1].bit_length())


def _quotient(budget: Budget, left: Pair, right: Pair) -> Pair:
    """left / right in lowest terms, right nonzero."""
    from ...trusted.integers import product_work

    left_bits, right_bits = _bits(left), _bits(right)
    budget.tick(1 + 2 * product_work(left_bits, right_bits) + _gcd_work(left_bits + right_bits))
    numerator, denominator = left[0] * right[1], left[1] * right[0]
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _minus_product(budget: Budget, total: Pair, factor: Pair, term: Pair) -> Pair:
    """total - factor * term in lowest terms: five products over the common denominator, then one gcd."""
    from ...trusted.integers import product_work

    total_bits, product_bits = _bits(total), _bits(factor) + _bits(term)
    budget.tick(1 + 2 * product_work(_bits(factor), _bits(term)) + 3 * product_work(total_bits, product_bits)
                + _gcd_work(total_bits + product_bits))
    product_numerator, product_denominator = factor[0] * term[0], factor[1] * term[1]
    numerator = total[0] * product_denominator - product_numerator * total[1]
    if not numerator:
        return _ZERO
    denominator = total[1] * product_denominator
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _divide(budget: Budget, dividend: list[Pair], divisor: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    """(quotient, remainder) of polynomials over Q without trailing zeros, len(dividend) >= len(divisor) >= 1 (member
    lengths strictly decrease); the remainder without trailing zeros, [] for zero (always for a constant divisor).
    Long division against the monic divisor / lc(divisor): only its nonzero coefficients update the remainder."""
    lead = divisor[-1]
    budget.tick(len(divisor))
    monic = [(index, _quotient(budget, pair, lead)) for index, pair in enumerate(divisor[:-1]) if pair[0]]
    top = len(divisor) - 1
    rest = list(dividend)
    quotient = [_ZERO] * (len(dividend) - top)
    for shift in range(len(quotient) - 1, -1, -1):
        budget.tick()
        head = rest[shift + top]
        if not head[0]:
            continue
        quotient[shift] = _quotient(budget, head, lead)
        for index, pair in monic:
            rest[shift + index] = _minus_product(budget, rest[shift + index], head, pair)
    remainder = rest[:top]
    while remainder and not remainder[-1][0]:
        remainder.pop()
    return quotient, remainder


def _rationals(budget: Budget, pairs: list[Pair]) -> tuple[Rational, ...]:
    """Canonical Rationals of reduced pairs; ResourceLimitError for an integer longer than max_integer_bits, which no
    Rational of the request's limits holds (the checker would refuse the claim)."""
    limit = budget.limits.max_integer_bits
    result = []
    for numerator, denominator in pairs:
        budget.tick()
        if numerator.bit_length() > limit or denominator.bit_length() > limit:
            raise ResourceLimitError(_TOO_LONG)
        result.append(trusted_rational(numerator, denominator))
    return tuple(result)


# ---------------------------------------------------------------------------------------------------------------
# cyclotomic_polynomial

def _cyclotomic_polynomial(budget: Budget, n: object, variable: object) -> tuple[Polynomial, tuple[object, ...]]:
    """Phi_n in the variable and the evidence ((d, Phi_d), ...) over the proper divisors d of n (module docstring)."""
    if type(n) is not int:  # Neither a bool nor a float nor a str is an exact integer.
        raise InvalidInput("cyclotomic_polynomial requires an exact integer n")
    if not _plain_symbol(variable):
        raise InvalidInput(_NOT_A_SYMBOL.format("cyclotomic_polynomial"))
    budget.check_int(n)
    if n < 1:
        raise InvalidInput("cyclotomic_polynomial requires n >= 1")
    if n > MAX_ORDER:
        raise ResourceLimitError("Computation limit exceeded: cyclotomic_polynomial takes n <= 100000")
    primes = _prime_factors(budget, n)
    totient = n
    for p in primes:
        totient = totient // p * (p - 1)
    if totient > MAX_POLYNOMIAL_DEGREE:
        raise ResourceLimitError(f"Computation limit exceeded: cyclotomic_polynomial takes phi(n) <= "
                                 f"{MAX_POLYNOMIAL_DEGREE}, the degree bound of a Polynomial (phi({n}) = {totient})")
    from ...trusted.upoly import exact_quotient

    divisors = _divisors(budget, n, primes)
    table: dict[int, tuple[int, ...]] = {}
    for d in divisors:
        own = [p for p in primes if not d % p]
        radical = 1
        for p in own:
            radical *= p
        if d == 1:
            table[d] = (-1, 1)
        elif radical < d:
            table[d] = _spread(budget, table[radical], d // radical)
        else:
            m = d // own[-1]
            table[d] = exact_quotient(_spread(budget, table[m], own[-1]), table[m])  # Charged by trusted.upoly.
    evidence = tuple((d, table[d]) for d in divisors[:-1])
    budget.tick(len(table[n]))
    return Polynomial(tuple(trusted_rational(c, 1) for c in table[n]), variable), evidence


def _prime_factors(budget: Budget, n: int) -> list[int]:
    """The distinct primes of 1 <= n <= MAX_ORDER, increasing, by trial division."""
    primes, rest, candidate = [], n, 2
    while candidate * candidate <= rest:
        budget.tick()
        if not rest % candidate:
            primes.append(candidate)
            while not rest % candidate:
                budget.tick()
                rest //= candidate
        candidate += 1
    if rest > 1:
        primes.append(rest)
    return primes


def _divisors(budget: Budget, n: int, primes: list[int]) -> list[int]:
    """Every divisor of n in increasing order, n last."""
    divisors = [1]
    for p in primes:
        power, powers = p, []
        while not n % power:
            budget.tick()
            powers.append(power)
            power *= p
        budget.tick(len(divisors) * len(powers))
        divisors += [d * q for d in divisors for q in powers]
    budget.tick(len(divisors))
    return sorted(divisors)


def _spread(budget: Budget, coefficients: tuple[int, ...], step: int) -> tuple[int, ...]:
    """The coefficients of P(x^step)."""
    length = (len(coefficients) - 1) * step + 1
    budget.tick(length)
    result = [0] * length
    result[::step] = coefficients
    return tuple(result)
