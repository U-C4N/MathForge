"""Independent checker of the family mathforge.areas.algebra.roots: count_real_roots and complex_roots.

The checker never reruns the producer's algorithms (root_isolation's Sturm chain, polynomial_exact's factorization,
v0.7's quadratic formula through the algebraic normalizer, the producer's sort). It reads the records as plain data,
recognized by the module and name of their class: a Polynomial (coefficients, canonical Rationals lowest degree first
without a trailing zero, at most 1,025 of them; variable, a Symbol), v0.7's ComplexRoot (root, multiplicity) and
ComplexAlgebraic (real, imag), and RealAlgebraicRoot (integer_coefficients, real_index), each field from the record
itself, never from a class-level default that a record forged with object.__new__ would show for a field it left unset
(ComplexAlgebraic declares imag = 0). Every number must have its exact type (type(x) is int: a bool, a float or a str
is no integer; a Rational subclass is no Rational; a list is no tuple) and the canonical form stated below, and the
evidence exactly the layout below; anything else answers False, and so does a claim on an input outside the
operation's domain, which has no value at all (the zero polynomial, an end of the wrong type or side, lower > upper),
and a claim whose arguments are not a plain dict. An Unevaluated value is no claim: run() never hands one to the
checker, and the checker refuses it as it refuses every value that is not an int or a tuple. Work is charged to the
active budget before it runs, in every loop; work past the budget raises ResourceLimitError, which leaves a claim
unverified, never verified. The checker uses no builtin pow and no ** at all.

The theorem that makes True imply the claim, for each operation:

count_real_roots(polynomial, lower, upper) = k, evidence ().
    The polynomial p is nonzero; lower is an int, a canonical Rational or '-inf', upper an int, a canonical Rational
    or '+inf', with lower <= upper when both are finite; k is an int with 0 <= k <= deg p. P is the primitive integer
    form of p, a nonzero rational multiple of p with the same roots. The checker's own Sturm chain of a polynomial
    (certify.upoly.sturm_chain: the subresultant remainders of it and its derivative, signed) ends in a nonzero
    constant multiple of their gcd, so a constant last member proves the polynomial square-free. S, a square-free
    integer polynomial with exactly the distinct roots of p, is found from the chain of P: S = P when its last member
    is a constant; otherwise, with G that last member made primitive, S = P / G, accepted only when G divides both P
    and P' exactly over Z and the chain of S ends in a constant. That suffices whatever G is: S divides P, so it has
    no other root, and a root of P of multiplicity m is a root of G of multiplicity at most m - 1 (G divides P'), so
    it is a root of S; S's chain proves it square-free. Sturm's theorem on S's chain (sign variations with zeros
    skipped, as certify.upoly.count_roots counts them; an infinite end read through the signs of the leading
    coefficients and the degrees) gives the number of distinct real roots of S in (lower, upper]. The closed interval
    [lower, upper] is (lower, upper] with lower added when lower is finite (an infinite end is open by nature), and
    lower is a root exactly when S(lower) = 0, decided by exact evaluation. So k is the number of distinct real roots
    of p in [lower, upper], and no other k passes; a nonzero constant has none.

complex_roots(polynomial) = roots, evidence (lc, factors).
    Let p be the polynomial (nonzero, of degree n) and P its primitive integer form with a positive leading coefficient.
    1. The factorization. factors is ((f, e), ...): each f a tuple of canonical Rationals, lowest degree first, of
       degree 1 or 2 and monic (its last item is Rational(1)); each e an int >= 1; the entries strictly increasing in
       the key (degree, ((numerator, denominator) of each coefficient, lowest degree first)), so the f are distinct;
       the sum of e * deg f is n. Every quadratic x^2 + b x + c has a discriminant D = b^2 - 4c that is no square of a
       rational (D = 0 included), so it has no rational root and is irreducible over Q, as every linear f is. lc is the
       leading coefficient of p, a canonical Rational, and P = prod F^e exactly, F the primitive integer form of f with
       a positive leading coefficient, multiplied out with trusted.upoly. Both sides of that identity are primitive
       (Gauss's lemma) with positive leading coefficients, and positive rational multiples of p / lc and of prod f^e,
       which are monic: so it holds exactly when p = lc * prod f^e. Distinct monic irreducible polynomials over Q are
       coprime and each one is separable, so the complex roots of p are the roots of the f, the roots of different f
       differ, and a root of f is a root of p of multiplicity exactly e.
    2. The roots of each factor. The checker forms them in closed form, each part as u + v sqrt(d) with rationals u, v,
       d: -a for x + a; for x^2 + b x + c, (-b - sqrt(D))/2 and (-b + sqrt(D))/2 when D > 0, and -b/2 - i sqrt(E) and
       -b/2 + i sqrt(E) with E = -D/4 when D < 0 (sqrt(E) a Rational when E is a rational square). It substitutes each
       into its factor by exact arithmetic in Q(sqrt(d)): they are deg f roots of f, distinct as D != 0, so all of them.
    3. Their canonical names. A part with v = 0 is the Rational u. Otherwise d is no rational square (d is D, or E when
       E is no square), u + v sqrt(d) is irrational with the minimal polynomial (X - u)^2 - v^2 d, irreducible over Q,
       whose two real roots are u - |v| sqrt(d) < u + |v| sqrt(d); its name is the RealAlgebraicRoot of that polynomial
       made primitive with a positive leading coefficient (the canonical form that mathforge.algebraic stores) with the
       real_index 1 for v > 0 and 0 for v < 0, the number of its real roots below the part. Every field of each record
       of roots must equal the name of one root of step 2 (exact ints compared exactly: a correct but non-canonical
       name of the same number, a RealAlgebraicRoot of a rational or of a multiple of the minimal polynomial, differs),
       no root is named twice, the multiplicity is the e of the root's factor, and roots has exactly sum of deg f
       records: so roots names every complex root of p once.
    4. The order. roots is strictly increasing by real part, then by imaginary part, decided exactly: the sign of
       a + b sqrt(n1) + c sqrt(n2) for integers a, b, c, n1, n2 >= 0 comes from the signs of the terms and, where they
       disagree, from the sign of a difference of their squares, one square root at a time (no approximation).
    So roots is every complex root of p with its multiplicity, canonically named, in the order of the card, and no
    other tuple passes. A nonzero constant has no root: roots = () with evidence (lc, ()).
    Mignotte's bound keeps forged evidence cheap: a divisor G of P in Z[x] has every coefficient below
    2**deg(G) * ||P||_2 <= 2**(deg(G) + B) in absolute value, B = max_bits(P) + ceil(bits(len(P)) / 2). A factor whose
    numerators reach that bound or whose denominators do not divide lc(P) (lc(F) divides lc(P), and every denominator
    of f divides lc(F)), and a partial product of the multiplication (a divisor of P when the claim is true) that
    exceeds it, are refused before the work grows.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import.
"""

from __future__ import annotations

from math import gcd, isqrt

from mathforge.certify.upoly import sturm_chain
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational, Symbol
from mathforge.trusted import upoly as tp

# The record classes read as data: the module and the name of each.
POLYNOMIAL = ("mathforge.polynomial", "Polynomial")
COMPLEX_ROOT = ("mathforge.complex_numbers", "ComplexRoot")
COMPLEX_ALGEBRAIC = ("mathforge.complex_numbers", "ComplexAlgebraic")
REAL_ALGEBRAIC_ROOT = ("mathforge.algebraic", "RealAlgebraicRoot")

# polynomial.MAX_POLYNOMIAL_DEGREE, which a checker cannot import; a test compares them.
MAX_DEGREE = 1024

# The argument names of each operation, in the order of its spec.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "count_real_roots": ("polynomial", "lower", "upper"),
    "complex_roots": ("polynomial",),
}
_MISSING = object()  # A field that a record forged with object.__new__ lacks.

# A surd (u, v, d) is the real number u + v * sqrt(d), with Rationals u, v and d >= 0; v = 0 and d = 0 for a rational.
Surd = tuple[Rational, Rational, Rational]
# A name: ("rational", numerator, denominator) or ("root", integer coefficients, real index).
Name = tuple[object, ...]


def check_operation(operation, args, value, evidence) -> bool:
    """Whether value, with evidence, is the verified answer of the operation on args, by the theorems of the module.
    False for an unknown operation, for arguments outside the operation's domain and for any malformed value or
    evidence; ResourceLimitError past the budget."""
    if type(operation) is not str or type(args) is not dict:
        return False
    names = _ARGUMENTS.get(operation)
    if names is None or len(args) != len(names) or any(name not in args for name in names):
        return False
    with computation() as budget:  # One budget for every helper: without an active budget, each would open its own.
        coefficients = _coefficients(budget, args["polynomial"])
        if coefficients is None:
            return False
        if operation == "count_real_roots":
            return _count_real_roots(budget, coefficients, args["lower"], args["upper"], value, evidence)
        return _complex_roots(budget, coefficients, value, evidence)


# ---------------------------------------------------------------------------------------------------------------
# Reading the data

def _class(value: object) -> tuple[str, str]:
    kind = type(value)
    return kind.__module__, kind.__qualname__


def _field(record: object, name: str) -> object:
    """A field of a record read as data, _MISSING when the record itself does not hold it. A record with a __dict__
    is read there and only there: getattr would answer a class-level default for a field that a record forged with
    object.__new__ left unset (ComplexAlgebraic declares imag = 0, so a forged root without its imaginary part would
    read as a real one). A record without a __dict__ keeps its fields in slots (Rational), which hold no default and
    raise AttributeError while unset."""
    try:
        fields = vars(record)
    except TypeError:  # No __dict__: slots.
        return getattr(record, name, _MISSING)
    if type(fields) is not dict:
        return _MISSING
    return fields.get(name, _MISSING)


def _canonical(budget: Budget, value: object, numerator_bits: int | None = None,
               denominator_bits: int | None = None) -> bool:
    """A Rational in lowest terms with a positive denominator, its integers within max_integer_bits or the tighter
    bounds given: the lengths are tested before the gcd, which they bound."""
    budget.tick()
    if type(value) is not Rational:
        return False
    numerator, denominator = _field(value, "numerator"), _field(value, "denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        return False
    limit = budget.limits.max_integer_bits
    if numerator_bits is not None:
        limit = min(limit, numerator_bits)
    if numerator.bit_length() > limit:
        return False
    limit = budget.limits.max_integer_bits
    if denominator_bits is not None:
        limit = min(limit, denominator_bits)
    if denominator.bit_length() > limit:
        return False
    if denominator == 1:
        return True
    tp.charge_gcd(max(numerator.bit_length(), denominator.bit_length()))
    return gcd(numerator, denominator) == 1


def _coefficients(budget: Budget, polynomial: object) -> tuple[Rational, ...] | None:
    """The coefficients of a canonical nonzero Polynomial, lowest degree first; None for anything else, the zero
    polynomial included, which neither operation takes."""
    if _class(polynomial) != POLYNOMIAL:
        return None
    coefficients = _field(polynomial, "coefficients")
    if (type(_field(polynomial, "variable")) is not Symbol or type(coefficients) is not tuple
            or not 1 <= len(coefficients) <= MAX_DEGREE + 1):
        return None
    for value in coefficients:
        if not _canonical(budget, value):
            return None
    if not coefficients[-1].numerator:  # A trailing zero, or the zero polynomial.
        return None
    return coefficients


def _integer_form(budget: Budget, coefficients: tuple[Rational, ...]) -> tuple[int, ...]:
    """The primitive integer polynomial with a positive leading coefficient that is a rational multiple of the nonzero
    polynomial with these canonical Rational coefficients (no trailing zero). ResourceLimitError when their common
    denominator passes max_integer_bits: the input is then too long to check."""
    limit = budget.limits.max_integer_bits
    common = 1
    for value in coefficients:
        budget.tick()
        denominator = value.denominator
        if denominator != 1 and common % denominator:
            tp.charge_gcd(max(common.bit_length(), denominator.bit_length()))
            common = common // gcd(common, denominator) * denominator
            if common.bit_length() > limit:
                raise ResourceLimitError("Computation limit exceeded: integer bits")
    integers = []
    for value in coefficients:
        budget.tick()
        tp.charge_product(common.bit_length(), max(value.numerator.bit_length(), value.denominator.bit_length()), 2)
        integers.append(value.numerator * (common // value.denominator))
    tp.charge_gcd(tp.max_bits(integers), len(integers))
    return tp.primitive(tuple(integers))


# ---------------------------------------------------------------------------------------------------------------
# count_real_roots

def _end(budget: Budget, end: object, infinity: str) -> tuple[bool, Rational | None]:
    """(valid, point) for an end of the interval: the Rational of an int or a canonical Rational, None for the infinity
    of its side; (False, None) for anything else, the other infinity included."""
    if type(end) is str:
        return end == infinity, None
    if type(end) is int:
        budget.tick()
        if end.bit_length() > budget.limits.max_integer_bits:
            return False, None
        return True, Rational(end)
    if type(end) is Rational and _canonical(budget, end):
        return True, end
    return False, None


def _count_real_roots(budget: Budget, coefficients: tuple[Rational, ...], lower: object, upper: object,
                      value: object, evidence: object) -> bool:
    (low_valid, low), (high_valid, high) = _end(budget, lower, "-inf"), _end(budget, upper, "+inf")
    degree = len(coefficients) - 1
    if (not low_valid or not high_valid or type(value) is not int or not 0 <= value <= degree
            or type(evidence) is not tuple or evidence):
        return False
    if low is not None and high is not None and low > high:
        return False
    if not degree:  # A nonzero constant has no root.
        return value == 0
    form = _integer_form(budget, coefficients)
    # The chain of P gives S (module docstring) without a separate gcd pass, which on a dense polynomial costs about
    # four times the chain: with certify.upoly.squarefree_part first, a dense polynomial of degree 210 that the
    # producer counts within the default budget (about 1.8 million units) took the checker 2.3 million, and run()
    # lost the answer; the chain alone takes 0.5 million.
    chain = sturm_chain(form)
    part = form
    if len(chain[-1]) > 1:  # A repeated root: S = P / G, G the last member made primitive.
        divisor = tp.primitive(chain[-1])
        division = tp.divmod_exact(form, divisor)
        if division is None or division[1]:
            return False
        common = tp.divmod_exact(tp.derivative(form), divisor)
        if common is None or common[1]:
            return False
        part = division[0]
        chain = sturm_chain(part)
        if len(chain[-1]) > 1:  # S must be square-free.
            return False
    tp.charge(sum(len(member) for member in chain))
    # The distinct real roots of S in (low, high], None for an infinite end (Sturm, as certify.upoly.count_roots).
    count = _variations(chain, low, False) - _variations(chain, high, True)
    if low is not None and tp.sign_at(part, low.numerator, low.denominator) == 0:
        count += 1  # The closed lower end is a root.
    return count == value


def _variations(chain: tuple[tuple[int, ...], ...], point: Rational | None, positive: bool) -> int:
    """The sign variations of a Sturm chain at a Rational point, zeros skipped, or at plus infinity (point None and
    positive) or minus infinity (point None, not positive), read from the leading coefficients and the degrees: the
    rule of certify.upoly.count_roots, whose chain this is."""
    signs = []
    for member in chain:
        if point is None:
            sign = 1 if member[-1] > 0 else -1
            if not positive and (len(member) - 1) % 2:
                sign = -sign
        else:
            sign = tp.sign_at(member, point.numerator, point.denominator)
        if sign:
            signs.append(sign)
    return sum(a != b for a, b in zip(signs, signs[1:]))


# ---------------------------------------------------------------------------------------------------------------
# complex_roots: the factorization

def _complex_roots(budget: Budget, coefficients: tuple[Rational, ...], value: object, evidence: object) -> bool:
    degree = len(coefficients) - 1
    if type(value) is not tuple or len(value) > degree or type(evidence) is not tuple or len(evidence) != 2:
        return False
    lead, factors = evidence
    top = coefficients[-1]
    if (not _canonical(budget, lead) or lead.numerator != top.numerator or lead.denominator != top.denominator
            or type(factors) is not tuple or len(factors) > degree):
        return False
    if not degree:  # A nonzero constant: no factor and no root.
        return not factors and not value
    target = _integer_form(budget, coefficients)
    bound = tp.max_bits(target) + (len(target).bit_length() + 1) // 2  # B of Mignotte's bound (module docstring).
    entries = _factors(budget, factors, degree, target[-1], bound)
    if entries is None or not _multiplies_out(budget, target, entries, bound):
        return False
    expected: dict[tuple[Name, Name], tuple[Surd, Surd, int]] = {}
    count = 0
    for values, _, exponent in entries:
        roots = _roots(budget, values)
        if roots is None:
            return False
        for real, imag in roots:
            budget.tick()
            count += 1
            expected[(_name(budget, real), _name(budget, imag))] = (real, imag, exponent)
    if len(expected) != count or len(value) != count:
        return False
    previous = None
    for record in value:
        budget.tick()
        claimed = _claimed(record)
        if claimed is None:
            return False
        name, multiplicity = claimed
        found = expected.pop(name, None)  # Each root is named once.
        if found is None or found[2] != multiplicity:
            return False
        if previous is not None and not _after(budget, found, previous):
            return False
        previous = found
    return True


def _factors(budget: Budget, factors: tuple, degree: int, lead: int,
             bound: int) -> list[tuple[tuple[Rational, ...], tuple[int, ...], int]] | None:
    """[(coefficients, integer form, exponent), ...] of the evidence's factors when each is a canonical monic
    polynomial of degree 1 or 2 within Mignotte's bound, every quadratic with a discriminant that is no rational square,
    in strictly increasing order of the key, with exponents >= 1 whose degrees add up to the polynomial's; None
    otherwise. lead is the leading coefficient of the polynomial's primitive integer form."""
    entries = []
    previous = None
    total = 0
    for entry in factors:
        budget.tick()
        if type(entry) is not tuple or len(entry) != 2:
            return None
        values, exponent = entry
        if (type(values) is not tuple or len(values) not in (2, 3) or type(exponent) is not int
                or not 1 <= exponent <= degree):
            return None
        size = len(values) - 1
        for item in values:
            # Numerators below 2**(size + bound); denominators that divide lc(F), hence lead: lengths first.
            if not _canonical(budget, item, size + bound, lead.bit_length()):
                return None
            tp.charge_product(lead.bit_length(), item.denominator.bit_length())
            if lead % item.denominator:
                return None
        if values[-1].numerator != 1 or values[-1].denominator != 1:  # Monic.
            return None
        key = (size, tuple((item.numerator, item.denominator) for item in values))
        if previous is not None and key <= previous:  # Strictly increasing: distinct factors.
            return None
        previous = key
        if size == 2 and _square_root(budget, values[1] * values[1] - 4 * values[0]) is not None:
            return None  # A rational root: the quadratic is reducible.
        total += size * exponent
        if total > degree:
            return None
        form = _integer_form(budget, values)
        if tp.max_bits(form) > size + bound:
            return None
        entries.append((values, form, exponent))
    return entries if total == degree else None


def _divisor_sized(polynomial: tuple[int, ...], bound: int) -> bool:
    """Whether every coefficient stays within Mignotte's bound for a divisor of that degree (module docstring)."""
    return tp.max_bits(polynomial) <= len(polynomial) - 1 + bound


def _multiplies_out(budget: Budget, target: tuple[int, ...],
                    entries: list[tuple[tuple[Rational, ...], tuple[int, ...], int]], bound: int) -> bool:
    """Whether target == prod F**e over the integer forms F and exponents e of the entries, multiplied out exactly
    (each power by right-to-left binary powering). Every partial product, every power of F formed on the way and every
    square of one is a divisor of target when the claim is true; one that leaves Mignotte's bound refuses the claim
    before the coefficients grow."""
    product: tuple[int, ...] = (1,)
    for _, form, exponent in entries:
        budget.tick()
        power: tuple[int, ...] = (1,)
        base = form
        while True:
            budget.tick()
            if exponent & 1:
                power = tp.mul(power, base)
                if not _divisor_sized(power, bound):
                    return False
            exponent >>= 1
            if not exponent:
                break
            base = tp.mul(base, base)
            if not _divisor_sized(base, bound):
                return False
        product = tp.mul(product, power)
        if not _divisor_sized(product, bound):
            return False
    return product == target


def _square_root(budget: Budget, value: Rational) -> Rational | None:
    """The rational s >= 0 with s * s = value, for a canonical Rational value; None when there is none."""
    numerator, denominator = value.numerator, value.denominator
    if numerator < 0:
        return None
    tp.charge_product(numerator.bit_length(), numerator.bit_length())  # Two integer square roots and two squares.
    tp.charge_product(denominator.bit_length(), denominator.bit_length())
    budget.tick()
    top, bottom = isqrt(numerator), isqrt(denominator)
    if top * top != numerator or bottom * bottom != denominator:
        return None
    return Rational(top, bottom)


# ---------------------------------------------------------------------------------------------------------------
# complex_roots: the roots, their names and their order

def _roots(budget: Budget, values: tuple[Rational, ...]) -> list[tuple[Surd, Surd]] | None:
    """The roots of the monic linear or irreducible quadratic factor with these coefficients, as (real part, imaginary
    part) surds in closed form (module docstring, step 2), each verified by exact substitution; None if one fails."""
    zero, half = Rational(0), Rational(1, 2)
    if len(values) == 2:
        roots = [((-values[0], zero, zero), (zero, zero, zero))]
    else:
        c, b = values[0], values[1]
        discriminant = b * b - 4 * c  # Never 0 nor a rational square: _factors refused those.
        u = -b * half
        if discriminant > 0:
            roots = [((u, sign * half, discriminant), (zero, zero, zero)) for sign in (-1, 1)]
        else:
            e = -discriminant / 4
            s = _square_root(budget, e)
            if s is None:  # sqrt(E) is irrational.
                roots = [((u, zero, zero), (zero, Rational(sign), e)) for sign in (-1, 1)]
            else:
                roots = [((u, zero, zero), (sign * s, zero, zero)) for sign in (-1, 1)]
    for root in roots:
        budget.tick()
        if not _vanishes(values, root):
            return None
    return roots


def _times(x: tuple[Rational, Rational], y: tuple[Rational, Rational], d: Rational) -> tuple[Rational, Rational]:
    """(r1 + s1 sqrt(d)) (r2 + s2 sqrt(d)) in Q(sqrt(d))."""
    return x[0] * y[0] + x[1] * y[1] * d, x[0] * y[1] + x[1] * y[0]


def _plus(*terms: tuple[Rational, Rational]) -> tuple[Rational, Rational]:
    first, second = Rational(0), Rational(0)
    for term in terms:
        first, second = first + term[0], second + term[1]
    return first, second


def _vanishes(values: tuple[Rational, ...], root: tuple[Surd, Surd]) -> bool:
    """Whether the factor with these coefficients vanishes at x + y i, x = u1 + v1 sqrt(d1) and y = u2 + v2 sqrt(d2), by
    exact arithmetic in Q(sqrt(d)) for the radicand d of its irrational part (a root that _roots forms has at most one):
    x + a and y for x + a; x^2 - y^2 + b x + c and y (2x + b) for x^2 + b x + c."""
    (u1, v1, d1), (u2, v2, d2) = root
    if v1 and v2:
        return False
    d = d1 if v1 else d2
    zero = Rational(0)
    x, y = (u1, v1), (u2, v2)
    if len(values) == 2:
        real, imag = _plus(x, (values[0], zero)), y
    else:
        c, b = values[0], values[1]
        square = _times(x, x, d)
        y_square = _times(y, y, d)
        real = _plus(square, (-y_square[0], -y_square[1]), (b * x[0], b * x[1]), (c, zero))
        imag = _times(y, (2 * x[0] + b, 2 * x[1]), d)
    return not (real[0] or real[1] or imag[0] or imag[1])


def _name(budget: Budget, surd: Surd) -> Name:
    """The canonical name of u + v sqrt(d) (module docstring, step 3); d is no rational square when v != 0."""
    u, v, d = surd
    if not v:
        return "rational", u.numerator, u.denominator
    budget.tick()
    minimal = _integer_form(budget, (u * u - v * v * d, -2 * u, Rational(1)))  # (X - u)^2 - v^2 d.
    return "root", minimal, 1 if v > 0 else 0


def _part(value: object) -> Name | None:
    """The name that a part of a claimed root spells, read as data; None for anything but a Rational with exact ints
    and a RealAlgebraicRoot with three exact int coefficients and an exact int index (only quadratic minimal
    polynomials name an irrational part)."""
    if type(value) is Rational:
        numerator, denominator = _field(value, "numerator"), _field(value, "denominator")
        if type(numerator) is int and type(denominator) is int:
            return "rational", numerator, denominator
        return None
    if _class(value) == REAL_ALGEBRAIC_ROOT:
        coefficients, index = _field(value, "integer_coefficients"), _field(value, "real_index")
        if (type(coefficients) is tuple and len(coefficients) == 3 and type(index) is int
                and all(type(item) is int for item in coefficients)):
            return "root", coefficients, index
    return None


def _claimed(record: object) -> tuple[tuple[Name, Name], int] | None:
    """((name of the real part, name of the imaginary part), multiplicity) of a ComplexRoot record read as data; None
    for anything else."""
    if _class(record) != COMPLEX_ROOT:
        return None
    root, multiplicity = _field(record, "root"), _field(record, "multiplicity")
    if type(multiplicity) is not int or multiplicity < 1 or _class(root) != COMPLEX_ALGEBRAIC:
        return None
    real, imag = _part(_field(root, "real")), _part(_field(root, "imag"))
    if real is None or imag is None:
        return None
    return (real, imag), multiplicity


def _integral(surd: Surd) -> tuple[int, int, int, int]:
    """(P, Q, R, n) with u + v sqrt(d) = (P + Q sqrt(n)) / R, R > 0 and n >= 0 integers: with u = p/q, v = r/s and
    d = m/t, sqrt(d) = sqrt(m t) / t, so P = p s t, Q = r q, R = q s t and n = m t."""
    u, v, d = surd
    p, q, r, s, m, t = u.numerator, u.denominator, v.numerator, v.denominator, d.numerator, d.denominator
    tp.charge_product(max(p.bit_length(), q.bit_length(), r.bit_length(), m.bit_length()),
                      s.bit_length() + t.bit_length() + q.bit_length(), 4)
    return p * s * t, r * q, q * s * t, m * t


def _sign(a: int, b: int, n: int) -> int:
    """The sign of a + b sqrt(n) for integers, n >= 0: the common sign of the terms, else the sign of the larger one,
    by comparing a^2 with b^2 n."""
    sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
    if not sb or not n:
        return sa
    if not sa or sa == sb:
        return sb
    difference = a * a - b * b * n
    return sa if difference > 0 else sb if difference < 0 else 0


def _sign3(budget: Budget, a: int, b: int, n1: int, c: int, n2: int) -> int:
    """The sign of a + b sqrt(n1) + c sqrt(n2) for integers, n1, n2 >= 0: that of a + b sqrt(n1) and that of
    c sqrt(n2) when they agree, else the sign of the larger, by the sign of (a + b sqrt(n1))^2 - c^2 n2
    = (a^2 + b^2 n1 - c^2 n2) + 2ab sqrt(n1)."""
    bits = max(a.bit_length(), b.bit_length(), n1.bit_length(), c.bit_length(), n2.bit_length())
    budget.tick()
    tp.charge_product(3 * bits, 3 * bits, 8)  # The products below: squares of numbers of up to three times the bits.
    first = _sign(a, b, n1)
    second = (c > 0) - (c < 0) if n2 else 0
    if not second:
        return first
    if not first or first == second:
        return second
    difference = _sign(a * a + b * b * n1 - c * c * n2, 2 * a * b, n1)
    return first if difference > 0 else second if difference < 0 else 0


def _compare(budget: Budget, first: Surd, second: Surd) -> int:
    """The sign of first - second, exactly: with first = (P1 + Q1 sqrt(n1)) / R1 and second likewise, R1 R2 > 0 times
    the difference is (P1 R2 - P2 R1) + Q1 R2 sqrt(n1) - Q2 R1 sqrt(n2)."""
    p1, q1, r1, n1 = _integral(first)
    p2, q2, r2, n2 = _integral(second)
    tp.charge_product(max(p1.bit_length(), q1.bit_length(), p2.bit_length(), q2.bit_length()),
                      max(r1.bit_length(), r2.bit_length()), 4)
    return _sign3(budget, p1 * r2 - p2 * r1, q1 * r2, n1, -q2 * r1, n2)


def _after(budget: Budget, current: tuple[Surd, Surd, int], previous: tuple[Surd, Surd, int]) -> bool:
    """Whether the root current comes strictly after previous: a larger real part, or the same and a larger
    imaginary part."""
    real = _compare(budget, current[0], previous[0])
    return real > 0 or (real == 0 and _compare(budget, current[1], previous[1]) > 0)
