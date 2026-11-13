"""Independent checker of the family mathforge.areas.linalg.eigen: eigenvectors.

The checker never reruns the producer's algorithms (Berkowitz's characteristic polynomial, Berlekamp, Hensel and
Zassenhaus, fraction-free Gauss-Jordan elimination). It verifies the characteristic polynomial modulo primes through a
Hessenberg form (certify.linalg.charpoly), the factorization over Q through one product identity and v0.7's
irreducibility criterion (certify.upoly.check_factorization), and each eigenspace through the echelon certificate
(certify.linalg.echelon.check_echelon). It reads the Matrix argument (module mathforge.matrices, fields rows and ncols)
and the records RationalEigenvectors and Eigenspace as data, recognized by the module and name of their class. Every
number must have its exact type (type(x) is int: a bool, a float or a str is no int; an int is no Rational in a
Rational field; a list is no tuple), every Rational of the argument and of the value must be in lowest terms with a
positive denominator, and the value and the evidence must have exactly the shapes below; anything else answers False.
Work is charged to the active budget before it runs, in every loop, and so is the one builtin power of the checker,
the modulus of an irreducibility certificate.

The theorem that makes True imply the claim:

eigenvectors(M) = RationalEigenvectors(eigenspaces, residual), evidence (charpoly, factorization, echelons).
    M is an n x n Matrix of canonical Rationals, n >= 0. s is the least common multiple of the denominators of its
    entries and A = s M the integer matrix of the entries s * M[i][j]; p(t) = det(t I - M), monic of degree n.
    1. The characteristic polynomial. charpoly is (1, C_1, ..., C_n), a tuple of n + 1 ints. check_charpoly(A,
       charpoly) holds exactly when the C_k are the coefficients of det(t I - A), highest power first (a Hadamard
       bound on them and agreement modulo primes whose product exceeds twice it). Then p(t) = s^-n det(s t I - A) =
       sum_k C_k s^-k t^(n-k), and the checker forms these coefficients itself, in lowest terms.
    2. The factorization. factorization is ((q, e, irreducibility), ...), each q a monic tuple of canonical Rationals
       lowest degree first, of degree >= 1, and e an int >= 1. The primitive integer forms (positive leading
       coefficients) satisfy integer_form(p) = prod integer_form(q)^e; the product is primitive too (Gauss's lemma),
       and both sides are positive rational multiples of the monic p and prod q^e, so p = prod q^e. Then
       check_factorization(p, factorization) holds: the q are distinct, in strictly increasing order of the key
       (degree, ((numerator, denominator) of each coefficient)), and each one is irreducible over Q (check_irreducible:
       ("linear",) for degree 1, else v0.7's modular certificate: the factors modulo a prime power, irreducible and
       pairwise coprime modulo the prime, multiplying to the factor, and no product of a subset of at most half the
       degree dividing it over Q). Q[t] has unique factorization, so this is the factorization of p. An irreducible
       polynomial of degree >= 2 has no rational root and distinct monic linear factors have distinct roots, so a
       rational lambda is a root of p exactly when t - lambda is one of the q, and its multiplicity as a root is that
       e. The checker requires the degree-1 entries (-lambda, 1) with their e to be exactly the (value, multiplicity)
       of the eigenspaces, whose values strictly increase: so the values are all the rational eigenvalues of M in
       increasing order, each once, and the multiplicities their algebraic multiplicities. It requires
       integer_form(residual) = prod over deg q >= 2 of integer_form(q)^e, so that, as above, residual = prod over
       deg q >= 2 of q^e: the monic quotient of p by prod (t - lambda)^m, with no rational root.
    3. The eigenspaces. echelons holds one pair (basis_rows, modulus) for each eigenspace, in their order. The basis B
       of an eigenspace has the shape of a canonical nullspace basis (vector j is n canonical Rationals whose last
       nonzero entry is 1, at its free column f_j; f_1 < f_2 < ...; vector j is 0 at f_1, ..., f_(j-1)). Let P be the
       other columns p_1 < ... < p_r and R the r x n matrix with R[i][p_k] = 1 if i = k else 0, and R[i][f_j] =
       -B[j][p_i]. With that shape R is in reduced row echelon form with the pivot columns P (row i is 0 before p_i,
       since every B[j] is 0 after f_j). Let A_l be M - lambda I with each row scaled by the lcm of its denominators
       (the integer rows of trusted.intmatrix.clear_row_denominators, formed here with charged lcms), which has the
       row space of M - lambda I. check_echelon(A_l, n, D R, D, P, basis_rows, modulus), D the lcm of the denominators
       of R, holds exactly when A_l = A_l[:, P] R (its C2, so rank A_l <= r and the rows of A_l lie in the row space of
       R) and the r x r minor of A_l at the rows basis_rows and the columns P has a determinant that is a unit modulo
       modulus (its C3, so rank A_l >= r): then R is the reduced row echelon form of M - lambda I, which is unique.
       The nullspace of M - lambda I is that of R, and its canonical basis, one vector per free column f with 1 at f,
       0 at the other free columns and -R[i][f] at p_i, is B entry by entry, by the construction of R: B is the
       canonical basis of the eigenspace of lambda.
    So the value is the claim, and no other value passes: the eigenvalues, the multiplicities and the residual are
    fixed by the factorization of p, and each basis by the reduced row echelon form of M - lambda I.

The order of the work. Cheap tests come first, so that a wrong claim fails before the work that it could make large:
the shapes of the value and of the evidence (exact types, lengths bounded by n, the pattern of zeros and ones of each
basis, every int within max_integer_bits, no gcd yet); the characteristic polynomial; the screens below, which bound
every Rational of the claim by the argument before its lowest terms are tested; the product identity, the eigenvalues
and the residual; the eigenspaces; and last check_factorization, for the irreducibility certificates. Their check is
the one whose work can grow exponentially, with the number of lifted factors (v0.7's criterion tries every product of
a subset of at most half the degree), so it runs only for a factorization of p that matches the value in every other
respect. A wrong claim that only it refutes names a factor of degree >= 2 with a rational root, and that root's linear
factor is one lifted factor: the subsets of one factor, tried first, find it.

Fast refusal of oversized forgeries. The value and the evidence are screened before the work that they could make
large, with necessary conditions that a true claim meets. Every int of the value and the evidence has at most
max_integer_bits bits, the shapes are bounded by n, and modulus < 2**61 (kernels.linalg takes its moduli from
trusted.primes, all below 2**61). With D(t) = s^n p(t) = sum_j C_(n-j) s^j t^j, an integer polynomial with leading
coefficient s^n, and norm >= log2 of its 2-norm (the bit length of its largest coefficient plus that of n + 1): a monic
factor q of p of degree d has a primitive integer form Q that divides the primitive form of D in Z[t] (Gauss), so every
denominator of q divides lc(Q), which divides s^n, and every numerator has at most d + norm + 1 bits (Mignotte: a
coefficient of Q is at most binomial(d, j) times the 2-norm of the primitive form of D, itself at most that of D). The
factors of the factorization and the residual must meet both, and the eigenvalues must equal the roots of the linear
factors, before any of their lowest terms or order is tested. A modular irreducibility certificate (prime, exponent,
lifted) is accepted only with prime <= 3 L + 3, exponent * (bitlen(prime) - 1) <= min(max_integer_bits, 2 (B +
bitlen(prime)) + 8) and a modulus prime^exponent of at most max_integer_bits bits (check_irreducible raises
ResourceLimitError past either of those two integer bounds, and the producer's lifting refuses to pass them), with at
most degree lifted factors of total degree degree and coefficients in [0, modulus); here L = bitlen(s^n) + n bitlen(n)
+ (2n - 1)(n + norm) + 1 and B = bitlen(s^n) + 2n + norm + 2. The producer's prime is the least prime that divides no
leading coefficient and keeps the polynomial S that it lifts square-free, S the primitive integer form of the factor
(polynomial_exact.factor_with_evidence), a divisor of the primitive form of D: every smaller prime divides lc(S)
disc(S), which is nonzero with log2 |lc(S) disc(S)| <= L (lc(S) | s^n, |disc(S)| <= deg(S)^deg(S) ||S||_2^(2 deg(S) -
1) by Hadamard's inequality on the Sylvester matrix of S and S', ||S||_2 <= 2^n ||D||_2 by Mignotte), so their product
is at most 2^L, and with theta(x) >= (x - 1)/3 for x >= 2 (Chebyshev's function, the sum of log q over the primes
q <= x) the prime is at most 3 L ln 2 + 2. Its lifting stops at the first modulus above twice the coefficient bound of
S, at most 2^B, so prime^exponent <= 2^B prime; the bound allows twice that.
In each eigenspace, R = G^-1 A_l[basis_rows] for the nonsingular minor G of C3, so by Cramer's rule every entry of R is
a quotient of two r x r minors of A_l: the lcm of the denominators of R divides det G, and the numerators and
denominators of R, of D R and of the basis have at most sum_i (bitlen of the largest entry of row i of A_l +
bitlen(n)) + n bits (Hadamard). The basis entries must meet that before their lowest terms are tested.

ResourceLimitError, the claim then being unverified: a matrix with more entries than max_matrix_entries (charged first,
as the producer charges it), s or a coefficient of p in lowest terms longer than max_integer_bits, a diagonal entry or
a row scale of M - lambda I past it, and any work past max_work. These depend on the argument M alone, or on the parts
of the claim already verified, never on a forgery: oversized forged values and evidence answer False. One exception is
inherent in v0.7's irreducibility criterion: its work grows with the degree and with the number of lifted factors that
a certificate of the right shape and size names, so such a certificate can take more than the budget and leave a claim
unverified rather than refused (a true claim can be made unverifiable that way, by merging two of its irreducible
factors into one that a valid-looking certificate calls irreducible). It never verifies a false claim.

A checker imports only mathforge.certify, mathforge.trusted, mathforge.errors, mathforge.limits and mathforge.model,
and never the producer, not even to name a record class: it compares (type(value).__module__,
type(value).__qualname__) instead. tools/check_layering.py follows every import, and tests/v08/test_layering.py
rejects importlib, __import__, sys.modules and every import of sys, builtins or importlib.
"""

from __future__ import annotations

from math import gcd

from mathforge.certify.linalg.charpoly import check_charpoly
from mathforge.certify.linalg.echelon import check_echelon
from mathforge.certify.upoly import check_factorization
from mathforge.limits import Budget, ResourceLimitError, computation
from mathforge.model import Rational, trusted_rational
from mathforge.trusted import upoly as tp

OPERATION = "eigenvectors"
ARGUMENT = "M"
MATRIX = ("mathforge.matrices", "Matrix")  # Module and name of the argument's class, read as data.
FAMILY = "mathforge.areas.linalg.eigen"
RECORD = (FAMILY, "RationalEigenvectors")  # Module and name of the value's class.
SPACE = (FAMILY, "Eigenspace")  # Module and name of the class of each eigenspace.
MODULUS_LIMIT = 1 << 61  # Echelon moduli are words: trusted.primes.moduli() yields only primes below 2**61.
_ZERO = trusted_rational(0, 1)
_BITS = "Computation limit exceeded: integer bits"


def _identity(value: object) -> tuple[str, str]:
    cls = type(value)
    return cls.__module__, cls.__qualname__


def _gcd(a: int, b: int) -> int:
    """gcd(a, b), charged before it runs: the long division of the longer integer by the shorter one, then Lehmer's
    algorithm on the shorter length (the cost model of trusted.upoly)."""
    long, short = sorted((a.bit_length(), b.bit_length()), reverse=True)
    tp.charge_product(long, short)
    tp.charge_gcd(short)
    return gcd(a, b)


def _exact(value: object, max_bits: int) -> bool:
    """A Rational of two exact ints with a positive denominator, both within max_bits. Lowest terms are tested apart
    (_lowest), once the integers are known to be small enough for it; a Rational made with object.__new__ may lack a
    field or break that form."""
    if type(value) is not Rational:
        return False
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    return (type(numerator) is int and type(denominator) is int and denominator >= 1
            and numerator.bit_length() <= max_bits and denominator.bit_length() <= max_bits)


def _lowest(value: Rational) -> bool:
    """An _exact Rational in lowest terms: zero is 0/1."""
    return value.denominator == 1 or _gcd(value.numerator, value.denominator) == 1


def _one(value: Rational) -> bool:
    return value.numerator == 1 and value.denominator == 1


def _less(left: Rational, right: Rational) -> bool:
    """left < right for Rationals with positive denominators, by cross multiplication, charged before it runs."""
    tp.charge_product(max(left.numerator.bit_length(), right.numerator.bit_length()),
                      max(left.denominator.bit_length(), right.denominator.bit_length()), 2)
    return left.numerator * right.denominator < right.numerator * left.denominator


# ---------------------------------------------------------------------------------------------------------------
# Shapes: the argument, the value and the evidence

def _matrix(matrix: object, budget: Budget) -> tuple[tuple[Rational, ...], ...] | None:
    """The rows of a square Matrix of canonical Rationals within max_integer_bits, else None (the Matrix constructor
    and the decoder build no other; s, and so the evidence, would depend on how a forged entry is written). Its entries
    are charged first, as the producer charges them: past max_matrix_entries, ResourceLimitError."""
    if _identity(matrix) != MATRIX:
        return None
    rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)
    if type(rows) is not tuple or type(ncols) is not int:
        return None
    budget.matrix(len(rows), ncols)
    if ncols != len(rows):
        return None
    max_bits = budget.limits.max_integer_bits
    for row in rows:
        budget.tick(1 + ncols)
        if type(row) is not tuple or len(row) != ncols:
            return None
        for entry in row:
            if not (_exact(entry, max_bits) and _lowest(entry)):
                return None
    return rows


def _free_columns(basis: tuple, n: int, budget: Budget, max_bits: int) -> tuple[int, ...] | None:
    """The free columns of a basis in the shape of a canonical nullspace basis of an n x n matrix: each vector a tuple
    of n exact Rationals whose last nonzero entry is 1, at its free column; the free columns strictly increasing; each
    vector 0 at the free columns of the vectors before it; every zero 0/1. None for any other basis. The lowest terms
    of the other entries are tested with the eigenspace, after the screen that bounds them."""
    free: list[int] = []
    for vector in basis:
        budget.tick(1 + n + len(free))
        if type(vector) is not tuple or len(vector) != n:
            return None
        last = -1
        for index, entry in enumerate(vector):
            if not _exact(entry, max_bits):
                return None
            if entry.numerator:
                last = index
            elif entry.denominator != 1:
                return None
        if last < 0 or not _one(vector[last]) or (free and last <= free[-1]):
            return None
        if any(vector[column].numerator for column in free):
            return None
        free.append(last)
    return tuple(free)


def _claim(value: object, n: int, budget: Budget):
    """(spaces, residual) of a RationalEigenvectors for an n x n matrix in the canonical shape, spaces being
    ((eigenvalue, multiplicity, basis, free columns), ...); None for any other value. This is the canonical form that
    the family's records validate, re-read here field by field (a record made with object.__new__ skipped that
    validation), with the multiplicities and the residual degree adding up to n. The order of the eigenvalues and the
    lowest terms of the Rationals are tested later, once the screens have bounded them (check_operation)."""
    if _identity(value) != RECORD:
        return None
    spaces, residual = getattr(value, "eigenspaces", None), getattr(value, "residual", None)
    if (type(spaces) is not tuple or len(spaces) > n or type(residual) is not tuple
            or not 1 <= len(residual) <= n + 1):
        return None
    max_bits = budget.limits.max_integer_bits
    budget.tick(len(residual))
    if not all(_exact(coefficient, max_bits) for coefficient in residual):
        return None
    if not _one(residual[-1]) or len(residual) == 2 or not residual[0].numerator:
        return None  # Monic, and neither of degree 1 nor with the root 0: both would be rational roots.
    parsed = []
    total = 0
    for space in spaces:
        budget.tick()
        if _identity(space) != SPACE:
            return None
        eigenvalue, multiplicity = getattr(space, "value", None), getattr(space, "multiplicity", None)
        basis = getattr(space, "basis", None)
        if (not _exact(eigenvalue, max_bits) or type(multiplicity) is not int or not 1 <= multiplicity <= n
                or type(basis) is not tuple or not 1 <= len(basis) <= multiplicity):
            return None
        free = _free_columns(basis, n, budget, max_bits)
        if free is None:
            return None
        parsed.append((eigenvalue, multiplicity, basis, free))
        total += multiplicity
    if total + len(residual) - 1 != n:
        return None
    return tuple(parsed), residual


def _evidence(evidence: object, n: int, spaces: int, budget: Budget):
    """(charpoly, factorization, echelons) of evidence in the layout of the module docstring, with every number of its
    exact type and within max_integer_bits, else None. The screens that need p come later (_screened)."""
    if type(evidence) is not tuple or len(evidence) != 3:
        return None
    charpoly, factorization, echelons = evidence
    max_bits = budget.limits.max_integer_bits
    if type(charpoly) is not tuple or len(charpoly) != n + 1:
        return None
    budget.tick(1 + n)
    if any(type(c) is not int or c.bit_length() > max_bits for c in charpoly) or charpoly[0] != 1:
        return None
    if type(factorization) is not tuple or len(factorization) > n:
        return None
    degree = 0
    for entry in factorization:
        budget.tick()
        if type(entry) is not tuple or len(entry) != 3:
            return None
        coefficients, multiplicity = entry[0], entry[1]
        if (type(coefficients) is not tuple or not 2 <= len(coefficients) <= n + 1 or type(multiplicity) is not int
                or not 1 <= multiplicity <= n):
            return None
        budget.tick(len(coefficients))
        if not all(_exact(c, max_bits) for c in coefficients):
            return None
        degree += (len(coefficients) - 1) * multiplicity
        if degree > n:
            return None
    if degree != n or type(echelons) is not tuple or len(echelons) != spaces:
        return None
    for pair in echelons:
        budget.tick()
        if type(pair) is not tuple or len(pair) != 2:
            return None
        rows, modulus = pair
        if type(rows) is not tuple or len(rows) > n or type(modulus) is not int or not 2 <= modulus < MODULUS_LIMIT:
            return None
        budget.tick(len(rows))
        if any(type(row) is not int for row in rows):
            return None
    return charpoly, factorization, echelons


# ---------------------------------------------------------------------------------------------------------------
# The characteristic polynomial

def _common_denominator(rows: tuple, budget: Budget, max_bits: int) -> int:
    """The least common multiple of the denominators of all entries; ResourceLimitError past max_bits."""
    s = 1
    for row in rows:
        budget.tick(1 + len(row))
        for entry in row:
            denominator = entry.denominator
            if denominator != 1:
                tp.charge_product(s.bit_length(), denominator.bit_length(), 3)  # s % d, s // g and the product.
                if s % denominator:
                    s = s // _gcd(s, denominator) * denominator
                    if s.bit_length() > max_bits:
                        raise ResourceLimitError(_BITS)
    return s


def _scaled(rows: tuple, s: int, budget: Budget) -> tuple[tuple[int, ...], ...]:
    """The integer rows of s times the rows (every denominator divides s)."""
    bits = s.bit_length()
    scaled = []
    for row in rows:
        budget.tick(1 + len(row))
        values = []
        for entry in row:
            numerator, denominator = entry.numerator, entry.denominator
            tp.charge_product(max(numerator.bit_length(), denominator.bit_length()), bits, 2)
            values.append(numerator * s if denominator == 1 else numerator * (s // denominator))
        scaled.append(tuple(values))
    return tuple(scaled)


def _powers(s: int, n: int, budget: Budget) -> list[int]:
    """s**0, ..., s**n by repeated multiplication, each product charged before it runs."""
    powers = [1]
    for _ in range(n):
        budget.tick()
        tp.charge_product(powers[-1].bit_length(), s.bit_length())
        powers.append(powers[-1] * s)
    return powers


def _characteristic(charpoly: tuple, powers: list[int], budget: Budget, max_bits: int) -> tuple[Rational, ...]:
    """p(t) = sum_k C_k s^-k t^(n-k) as canonical Rationals, lowest degree first; ResourceLimitError for a coefficient
    longer than max_bits in lowest terms."""
    n = len(charpoly) - 1
    coefficients = []
    for j in range(n + 1):
        budget.tick()
        value, power = charpoly[n - j], powers[n - j]
        if not value:
            coefficients.append(_ZERO)
            continue
        divisor = _gcd(value, power)
        tp.charge_product(max(value.bit_length(), power.bit_length()), divisor.bit_length(), 2)
        numerator, denominator = value // divisor, power // divisor
        if numerator.bit_length() > max_bits or denominator.bit_length() > max_bits:
            raise ResourceLimitError(_BITS)
        coefficients.append(trusted_rational(numerator, denominator))
    return tuple(coefficients)


# ---------------------------------------------------------------------------------------------------------------
# The factorization: screens (module docstring), lowest terms and the product identity

def _fits(coefficients: tuple, lead: int, norm: int, budget: Budget) -> bool:
    """Gauss and Mignotte for a monic factor of p: every denominator divides lead = s^n and every numerator has at
    most degree + norm + 1 bits."""
    limit = len(coefficients) + norm  # degree + norm + 1
    lead_bits = lead.bit_length()
    for c in coefficients:
        budget.tick()
        numerator, denominator = c.numerator, c.denominator
        if numerator.bit_length() > limit or denominator.bit_length() > lead_bits:
            return False
        if denominator != 1:
            tp.charge_product(lead_bits, denominator.bit_length())
            if lead % denominator:
                return False
    return True


def _certificate_fits(certificate: object, degree: int, prime_cap: int, lift_bits: int, budget: Budget) -> bool:
    """The shape of check_irreducible's evidence for a factor of that degree, within the bounds of the module
    docstring: ("linear",), or ("modular", prime, exponent, lifted) with at most degree lifted factors of total degree
    degree, each a tuple of ints in [0, prime**exponent). check_irreducible then raises no ResourceLimitError for the
    length of its modulus."""
    if type(certificate) is not tuple or not certificate or type(certificate[0]) is not str:
        return False
    if degree == 1:
        return len(certificate) == 1 and certificate[0] == "linear"
    if len(certificate) != 4 or certificate[0] != "modular":
        return False
    _, prime, exponent, lifted = certificate
    if type(prime) is not int or not 2 <= prime <= prime_cap or type(exponent) is not int or exponent < 1:
        return False
    bits = prime.bit_length()
    max_bits = budget.limits.max_integer_bits
    if exponent * (bits - 1) > min(max_bits, 2 * (lift_bits + bits) + 8):
        return False
    tp.charge_power(bits, exponent)  # The power has at most exponent * bits <= 2 max_bits bits.
    modulus = prime ** exponent
    if modulus.bit_length() > max_bits:
        return False
    if type(lifted) is not tuple or not 1 <= len(lifted) <= degree:
        return False
    total = 0
    for factor in lifted:
        budget.tick()
        if type(factor) is not tuple or not 2 <= len(factor) <= degree + 1:
            return False
        budget.tick(len(factor))
        if any(type(c) is not int or not 0 <= c < modulus for c in factor):
            return False
        total += len(factor) - 1
    return total == degree


def _screened(charpoly: tuple, powers: list[int], factorization: tuple, residual: tuple, budget: Budget) -> bool:
    """Whether the factors, their certificates and the residual meet the necessary conditions of the module docstring
    for D(t) = s^n p(t), whose coefficient of t^j is C_(n-j) s^j."""
    n = len(charpoly) - 1
    lead = powers[n]
    budget.tick(1 + n)
    widest = max((charpoly[n - j].bit_length() + powers[j].bit_length() for j in range(n + 1) if charpoly[n - j]),
                 default=0)
    norm = widest + (n + 1).bit_length()
    lead_bits = lead.bit_length()
    prime_cap = 3 * (lead_bits + n * n.bit_length() + (2 * n - 1) * (n + norm) + 1) + 3
    lift_bits = lead_bits + 2 * n + norm + 2
    for coefficients, _, certificate in factorization:
        if not _fits(coefficients, lead, norm, budget):
            return False
        if not _certificate_fits(certificate, len(coefficients) - 1, prime_cap, lift_bits, budget):
            return False
    return _fits(residual, lead, norm, budget)


def _lowest_monic(polynomials: list, budget: Budget) -> bool:
    """Every coefficient in lowest terms and every polynomial monic; run after the screens, which bound these
    integers by the argument."""
    for coefficients in polynomials:
        if not _one(coefficients[-1]):
            return False
        for c in coefficients:
            budget.tick()
            if not _lowest(c):
                return False
    return True


def _form(coefficients: tuple, budget: Budget) -> tuple[int, ...]:
    """The primitive integer polynomial with a positive leading coefficient of a monic polynomial with canonical
    coefficients (what certify.upoly.integer_form returns), each lcm, product, gcd and quotient charged before it
    runs: the coefficients times the lcm of their denominators, divided by their content."""
    scale = 1
    for c in coefficients:
        budget.tick()
        denominator = c.denominator
        if denominator != 1:
            tp.charge_product(scale.bit_length(), denominator.bit_length(), 3)  # scale % d, scale // g, the product.
            if scale % denominator:
                scale = scale // _gcd(scale, denominator) * denominator
    values = []
    content = 0
    for c in coefficients:
        budget.tick()
        tp.charge_product(c.numerator.bit_length(), scale.bit_length(), 2)
        value = c.numerator * (scale // c.denominator)
        values.append(value)
        if content != 1:
            content = _gcd(content, value)
    if content == 1:  # The leading coefficient is scale > 0: monic polynomials need no sign.
        return tuple(values)
    quotients = []
    for value in values:
        budget.tick()
        tp.charge_product(value.bit_length(), content.bit_length())
        quotients.append(value // content)
    return tuple(quotients)


# ---------------------------------------------------------------------------------------------------------------
# The eigenspaces

def _difference(entry: Rational, eigenvalue: Rational) -> Rational:
    """entry - eigenvalue, the Rational subtraction with its products and gcds charged before it runs (the Rational
    constructor path refuses a result past max_integer_bits)."""
    numerator_bits = max(entry.numerator.bit_length(), eigenvalue.numerator.bit_length())
    denominator_bits = max(entry.denominator.bit_length(), eigenvalue.denominator.bit_length())
    tp.charge_product(numerator_bits, denominator_bits, 3)
    tp.charge_gcd(numerator_bits + denominator_bits, 2)
    return entry - eigenvalue


def _eigenspace(rows: tuple, eigenvalue: Rational, basis: tuple, free: tuple[int, ...], basis_rows: tuple,
                modulus: int, budget: Budget) -> bool:
    """Whether basis is the canonical nullspace basis of M - eigenvalue I read from its reduced row echelon form:
    check_echelon of the row-scaled integer rows of M - eigenvalue I against the reduced rows R that the basis
    determines, certified by the minor on basis_rows modulo modulus (module docstring, step 3)."""
    n = len(rows)
    max_bits = budget.limits.max_integer_bits
    int_rows = []
    for i, row in enumerate(rows):
        budget.tick(1 + n)
        shifted = (row[:i] + (_difference(row[i], eigenvalue),) + row[i + 1:],)
        int_rows.append(_scaled(shifted, _common_denominator(shifted, budget, max_bits), budget)[0])
    limit, widest = n, 0  # Hadamard: the minors of int_rows have at most `limit` bits.
    for row in int_rows:
        budget.tick(1 + n)
        bits = tp.max_bits(row)
        limit += bits + n.bit_length()
        widest = max(widest, bits)
    free_set = set(free)
    pivots = tuple(column for column in range(n) if column not in free_set)
    if len(basis_rows) != len(pivots):
        return False
    denominator = 1
    for vector in basis:
        budget.tick(1 + len(pivots))
        for column in pivots:
            entry = vector[column]
            if not entry.numerator:
                continue  # Zero is 0/1 (_free_columns).
            d = entry.denominator
            if entry.numerator.bit_length() > limit or d.bit_length() > limit or not _lowest(entry):
                return False
            if d != 1:
                tp.charge_product(denominator.bit_length(), d.bit_length(), 3)
                if denominator % d:
                    denominator = denominator // _gcd(denominator, d) * d
                    if denominator.bit_length() > limit:
                        return False
    numerators = []
    numerator_bits = denominator.bit_length()
    for pivot in pivots:
        budget.tick(1 + n)
        row = [0] * n
        row[pivot] = denominator
        for vector, column in zip(basis, free):
            entry = vector[pivot]
            if entry.numerator:
                tp.charge_product(entry.numerator.bit_length(), denominator.bit_length(), 2)
                value = -entry.numerator * (denominator // entry.denominator)
                row[column] = value
                numerator_bits = max(numerator_bits, value.bit_length())
        numerators.append(tuple(row))
    # check_echelon's C2 forms about n * len(free) * len(pivots) products of an entry of int_rows by a numerator, and
    # counts them only after the loop: their size is charged here, before it runs.
    tp.charge_product(widest, numerator_bits, n * len(free) * len(pivots))
    return check_echelon(tuple(int_rows), n, tuple(numerators), denominator, pivots, basis_rows, modulus)


# ---------------------------------------------------------------------------------------------------------------

def check_operation(operation, args, value, evidence):
    """True when value is the RationalEigenvectors of the argument M, certified by evidence (module docstring)."""
    if type(operation) is not str or operation != OPERATION or type(args) is not dict or len(args) != 1:
        return False
    if ARGUMENT not in args:
        return False
    with computation() as budget:
        rows = _matrix(args[ARGUMENT], budget)
        if rows is None:
            return False
        n = len(rows)
        claim = _claim(value, n, budget)
        if claim is None:
            return False
        spaces, residual = claim
        parts = _evidence(evidence, n, len(spaces), budget)
        if parts is None:
            return False
        charpoly, factorization, echelons = parts
        max_bits = budget.limits.max_integer_bits
        # 1. The characteristic polynomial of M, from that of the integer matrix s M.
        s = _common_denominator(rows, budget, max_bits)
        if not check_charpoly(_scaled(rows, s, budget), charpoly):
            return False
        powers = _powers(s, n, budget)
        p = _characteristic(charpoly, powers, budget, max_bits)
        # 2. p = prod q^e, the eigenvalues and their multiplicities, and the residual; irreducibility comes last.
        if not _screened(charpoly, powers, factorization, residual, budget):
            return False
        if not _lowest_monic([q for q, _, _ in factorization] + [residual], budget):
            return False
        linear_product: tuple[int, ...] = (1,)
        residual_product: tuple[int, ...] = (1,)
        linear = []
        for q, e, _ in factorization:
            form = _form(q, budget)
            if len(q) == 2:
                linear.append((-q[0].numerator, q[0].denominator, e))
            for _ in range(e):
                budget.tick()
                if len(q) == 2:
                    linear_product = tp.mul(linear_product, form)
                else:
                    residual_product = tp.mul(residual_product, form)
        if tp.mul(linear_product, residual_product) != _form(p, budget):
            return False
        budget.tick(1 + len(spaces) + len(linear))
        claimed = [(eigenvalue.numerator, eigenvalue.denominator, m) for eigenvalue, m, _, _ in spaces]
        if len(set(linear)) != len(linear) or len(claimed) != len(linear) or set(claimed) != set(linear):
            return False
        for (left, _, _, _), (right, _, _, _) in zip(spaces, spaces[1:]):
            budget.tick()
            if not _less(left, right):
                return False
        if _form(residual, budget) != residual_product:
            return False
        # 3. Each eigenspace: the canonical basis of the nullspace of M - lambda I.
        for (eigenvalue, _, basis, free), (basis_rows, modulus) in zip(spaces, echelons):
            if not _eigenspace(rows, eigenvalue, basis, free, basis_rows, modulus, budget):
                return False
        # 4. The order, the canonical form, the product again and the irreducibility of every factor.
        return check_factorization(p, factorization)
