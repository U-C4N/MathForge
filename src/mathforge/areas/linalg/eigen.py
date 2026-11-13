"""Rational eigenvalues of exact rational matrices, with their algebraic multiplicities and eigenspace bases
(mathforge.linalg).

eigenvectors(M) takes a square Matrix M of exact Rationals, n x n with n >= 0 (MCP kind and annotation "Matrix"), and
returns a RationalEigenvectors record. Let p(t) = det(t I - M), the characteristic polynomial, monic of degree n.

- eigenspaces: one Eigenspace for each distinct rational root lambda of p, strictly increasing in lambda:
  Eigenspace(value=lambda, multiplicity=m, basis=B), m >= 1 the multiplicity of lambda as a root of p (the algebraic
  multiplicity) and B the canonical basis of the nullspace of M - lambda I read from its reduced row echelon form R:
  one vector per free column f (a column of R without a pivot), in increasing f, holding 1 at f, 0 at every other free
  column and -R[i][f] at the pivot column of each row i of R. Each vector is a tuple of n Rationals; its last nonzero
  entry is the 1 at its free column. 1 <= len(B) <= m (geometric multiplicity at most algebraic).
- residual: the monic polynomial p / prod (t - lambda)^m over the eigenspaces, as its tuple of Rationals lowest degree
  first; it has no rational root, so it is never of degree 1 and its constant term is never 0. (1,) when every
  eigenvalue is rational, also for the 0 x 0 matrix, whose value is RationalEigenvectors((), (1,)).

Known values (docs/v09/cards/m4-linalg-geometry-stats-discrete.md): [[2, 1], [1, 2]] gives 1 with basis ((-1, 1),)
and 3 with ((1, 1),); [[0, 2], [1, 0]] gives no eigenspace and the residual (-2, 0, 1), t^2 - 2; [[1, 1], [0, 1]] gives
1 with multiplicity 2 and basis ((1, 0),); the 3 x 3 identity gives 1 with multiplicity 3 and the standard basis.

Evidence: (charpoly, factorization, echelons), the layout that mathforge.certify.linalg.eigen checks.

- charpoly: (1, C_1, ..., C_n), a tuple of ints: the coefficients of det(t I - s M), highest power first, where s is
  the least common multiple of the denominators of all entries of M (s = 1 for an integer matrix and for n = 0) and s M
  is the integer matrix of the entries s * M[i][j]. kernels.linalg.charpoly of those integer rows returns this list.
  Then p(t) = sum_k C_k s^-k t^(n-k).
- factorization: the factorization of p over Q into distinct monic irreducible factors, a tuple of entries
  (coefficients, multiplicity, irreducibility), () for n = 0. coefficients is a tuple of canonical Rationals, lowest
  degree first, monic, of degree >= 1; multiplicity an int >= 1; irreducibility is ("linear",) for degree 1, else v0.7's
  ("modular", prime, exponent, lifted) for the primitive integer form of the factor, which certify.upoly
  .check_irreducible accepts (polynomial_exact._irreducibility_evidence and kernels.upoly.factor_square_free build it:
  the least prime that divides no leading coefficient and keeps the factor's square-free polynomial square-free, and
  Hensel lifting one p-adic digit at a time until the modulus exceeds twice the coefficient bound). The entries are
  strictly increasing in the key (degree, tuple((c.numerator, c.denominator) for c in coefficients)), the order of
  polynomial_exact.factor_with_evidence. Its degree-1 entries (-lambda, 1) are exactly the eigenvalues with their
  multiplicities, and the product of the others to their multiplicities is the residual.
- echelons: one pair (basis_rows, modulus) for each eigenspace, in the order of eigenspaces: the part of the echelon
  certificate of certify.linalg.echelon.check_echelon that the basis does not determine. Let A be the integer matrix of
  M - lambda I with each row scaled by the least common multiple of its denominators
  (trusted.intmatrix.clear_row_denominators), P the pivot columns (the columns that are no free column of the basis,
  increasing) and r = len(P) = n - len(basis). basis_rows is a tuple of r distinct row indices and modulus an int with
  2 <= modulus < 2**61 such that the r x r minor of A at the rows basis_rows (in that order) and the columns P has a
  determinant that is a unit modulo modulus. kernels.linalg.echelon(A, n) returns them as its basis_rows and modulus
  ((), 2 for rank 0); the rows of R are its numerators over its denominator, from which the basis is read.

The checker re-reads every number with its exact type (a bool, a float or a str is no int, an int is no Rational in a
Rational field, a list is no tuple), and it raises ResourceLimitError, the claim then being unverified, for a matrix
with more than max_matrix_entries entries, for s, a coefficient of p, or a diagonal entry or a row scale of
M - lambda I longer than max_integer_bits, and for work past max_work (which the subsets of an irreducibility
certificate with many lifted factors can take). It refuses an irreducibility certificate whose prime or exponent lies
beyond generous bounds derived from p (the producer's least good prime and digit lifting stay far below them), or whose
modulus prime**exponent is longer than max_integer_bits, so the producer uses that evidence. It tests the
irreducibility certificates last, after the product identity, the eigenvalues, the residual and the eigenspaces.

How produce() computes (docs/v09/cards/m4-linalg-geometry-stats-discrete.md). Nothing here is a proof: the checker
verifies the characteristic polynomial, the factorization and every eigenspace with certificates of its own.

- It charges one unit for the call and max_matrix_entries for M (current_budget().matrix) before it validates
  anything but the class of M, so that a 65 x 65 M exceeds the budget whatever its shape; then the shape (square,
  n x n with n >= 0) and the entries: exact Rationals with int parts, a positive denominator and in lowest terms, each
  integer within max_integer_bits. A Matrix built by its constructor holds no other entry. One forged without it may,
  and the checker refuses an entry such as 4/2 (s, and so the evidence, would depend on how it is written), so
  produce() rejects it as invalid input. The 0 x 0 matrix gives RationalEigenvectors((), (1,)) with the evidence
  ((1,), (), ()).
- s, the least common multiple of the denominators, grown one denominator at a time; the characteristic polynomial
  (1, C_1, ..., C_n) of s M by kernels.linalg.charpoly (Berkowitz's division-free algorithm); the coefficients
  C_k / s^k of p in lowest terms, and from them the primitive integer form of p.
- The factorization of p over Q that polynomial_exact.factor_with_evidence computes, run on integer tuples through
  the kernels.upoly port that it runs: the square-free decomposition of the primitive integer form (decided at once
  when the form and its derivative are coprime modulo a word prime, the usual case, whose decomposition is the form
  itself; square_free_integer's pseudo-remainder sequence otherwise), then Berlekamp,
  Hensel lifting and Zassenhaus recombination of each part (factor_square_free). Each factor of degree >= 2 carries
  its own lift data, kernels.upoly.lift_data of its primitive integer form, which is factor_with_evidence's evidence
  (factor_square_free returns exactly that for a part that is irreducible, and it is not computed twice). The entries
  come in factor_with_evidence's order.
- The eigenvalues are the roots -c of the linear factors t + c, in increasing order, each with the multiplicity of its
  factor; the residual is the product of the other factors to their multiplicities, formed from their primitive
  integer forms (trusted.upoly.mul) and made monic.
- For each eigenvalue lambda: the rows of M - lambda I, each scaled by the least common multiple of its denominators
  (the integer rows A_l that trusted.intmatrix.clear_row_denominators gives and the checker forms), the reduced row
  echelon form of A_l by kernels.linalg.echelon (fraction-free Gauss-Jordan elimination), the canonical nullspace basis
  read from its numerators over its denominator, and its (basis_rows, modulus). The kernel's modulus is a word prime
  below 2**61; under max_integer_bits < 61 the least prime that does not divide the minor's determinant replaces it.
  When the kernel refuses a pivot past max_integer_bits (its last pivot is the whole minor, however small the reduced
  rows), Gauss-Jordan elimination over Q with entries in lowest terms takes over, and the modulus is then a prime of at
  most max_integer_bits bits at which the minor is nonsingular.

Limits. ResourceLimitError refuses the request wherever the checker raises it: for s, a coefficient of p in lowest
terms, a diagonal entry or a row scale of M - lambda I longer than max_integer_bits. It also refuses what the checker
would answer False although it is true: an int of the evidence or a Rational of the value longer than
max_integer_bits (operations_v6.run refuses those as well, before the checker). Three refusals of the algorithms
remain for claims that the checker would accept under the same limits. v0.7's factorization lifts every square-free
part of degree >= 2 to a modulus above twice its coefficient bound even when all its factors are linear, so under
max_integer_bits = 64 the matrix diag(2^-62, 1), with p = (t - 2^-62)(t - 1), is refused. A characteristic polynomial
with a repeated factor goes through the square-free decomposition's pseudo-remainder sequence, whose leading
coefficients may pass max_integer_bits. And Gauss-Jordan elimination over Q, which runs only after the kernel refuses a
pivot, refuses an entry of an intermediate state past max_integer_bits; that binds only under limits a few bits above
the size of the entries. Long entries meet max_work first: a 20 x 20 matrix of random 200-bit entries, whose
characteristic polynomial has coefficients of about 4,000 bits, is refused before the digit-by-digit Hensel lifting of
its irreducible characteristic polynomial, whose cost is charged in advance (Work). produce() refuses a Result that the
verifier would not admit under
max_evidence_items: the verifier counts every item of the stored Result among its evidence items, the value twice (in
the Result and in its step), the evidence once, the request twice and the Result's own fields with its four stored
checks (_RESULT_ITEMS), so every Result that run() returns verifies again under the limits it was made with, as does
the Result decoded from its JSON text. Under the default limits that never binds: a 64 x 64 M with 64 basis vectors
takes about 50,000 of the 100,000 items.

Work. Every step charges the request's budget before it runs. The kernels charge their own batched ticks: Berkowitz
1 + size^3 for each trailing block, an elimination one row length per row rewrite, trusted.upoly its coefficient
counts. produce() charges its own steps on the scale of trusted.upoly (the checker's measure): a unit per entry or
coefficient, charge_product and charge_gcd for the long products, quotients and gcds of the scales, the powers of s and
the lowest terms. Before Berkowitz and before each elimination it charges an estimate of the long-integer products
that their ticks leave out, from the growth of their integers (each product with an entry and each sum of n terms
adds at most the bits of the widest entry and of n): nothing for entries of a few dozen bits, and enough for long
entries that max_work bounds the time (a 30 x 30 matrix of 2,000-bit entries is refused at once; Berkowitz alone would
take seconds on it). Before each Hensel lifting it charges the same way the long products of every lifting step, known
from the prime and the coefficient bound (_charge_lifting): trusted.upoly charges one unit per coefficient operation
whatever its size, and a 10 x 10 matrix of random 1,000-bit entries took 4 seconds in the lifting before this charge.
Berkowitz's ticks add up to the sum of 1 + size^3 for size = 1, ..., n, about n^4 / 4, known
before it starts: a request whose budget they would exceed is refused at once. Under the default max_work of
2,000,000 no matrix larger than 52 x 52 fits, whatever its entries, and a random 52 x 52 one does not fit either (its
irreducible characteristic polynomial of degree 52 takes the rest).

mathforge.areas.ntheory.continued_fraction spells out the contract of a v6 family, documentation/families.md is the
checklist, and tests/v09/linalg/test_eigen.py is the definition of done.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from ...errors import InvalidInput, UnsupportedOperation
from ...limits import Budget, ResourceLimitError, computation
from ...matrices import Matrix
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

# The scope text of every operation of wave M4 (docs/v09/cards/m4-linalg-geometry-stats-discrete.md).
_SCOPE = "exact rational vectors, matrices and data; certified linear-algebra, geometry, statistics and counts"

SPECS = (
    OperationSpec("eigenvectors", "linalg",
                  (Argument("M", "Matrix", None, "matrix"),),
                  "RationalEigenvectors",
                  "Find every rational eigenvalue of a square rational matrix with its multiplicity and eigenspace "
                  "basis.", _SCOPE, version=6),
)


def _exact_rational(budget: Budget, value: object, what: str) -> None:
    """Refuse, with InvalidInput, a field that is not a Rational of two exact ints with a positive denominator (a
    Rational made with object.__new__ may lack a field or break that form); ResourceLimitError for an integer longer
    than max_integer_bits. Lowest terms are the Rational constructor's guarantee; the checker recomputes them."""
    if type(value) is not Rational:  # An int is no Rational here, nor is a bool or a float.
        raise InvalidInput(f"The {what} is an exact Rational")
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if type(numerator) is not int or type(denominator) is not int or denominator < 1:
        raise InvalidInput(f"The {what} is a Rational of two exact integers")
    budget.check_int(numerator)
    budget.check_int(denominator)


def _is_one(value: Rational) -> bool:
    return value.numerator == 1 and value.denominator == 1


@dataclass(frozen=True)
class Eigenspace:
    """A rational eigenvalue, its algebraic multiplicity and the canonical basis of its eigenspace.

    The canonical form: value is a Rational; multiplicity an int >= 1; basis a tuple of between 1 and multiplicity
    vectors, each a tuple of the same positive number of Rationals. The basis has the shape of the nullspace basis read
    from a reduced row echelon form: the last nonzero entry of each vector is 1, at its free column; the free columns
    strictly increase from one vector to the next; and each vector is 0 at the free columns of the vectors before it
    (it is 0 at the later ones, which lie after its last nonzero entry)."""
    value: Rational
    multiplicity: int
    basis: tuple[tuple[Rational, ...], ...]

    def __post_init__(self):
        with computation() as budget:
            _exact_rational(budget, self.value, "eigenvalue")
            multiplicity, basis = self.multiplicity, self.basis
            if type(multiplicity) is not int or multiplicity < 1:  # Neither bool nor float is an exact integer.
                raise InvalidInput("The multiplicity of an eigenvalue is an exact integer >= 1")
            budget.check_int(multiplicity)
            if type(basis) is not tuple or not basis or len(basis) > multiplicity:
                raise InvalidInput("An eigenspace basis is a nonempty tuple of at most multiplicity vectors")
            size = len(basis[0]) if type(basis[0]) is tuple else 0
            free: list[int] = []
            for vector in basis:
                budget.tick()
                if type(vector) is not tuple or not vector or len(vector) != size:
                    raise InvalidInput("The vectors of an eigenspace basis are nonempty tuples of one length")
                last = -1
                for index, entry in enumerate(vector):
                    budget.tick()
                    _exact_rational(budget, entry, "eigenvector entry")
                    if entry.numerator:
                        last = index
                if last < 0 or not _is_one(vector[last]) or (free and last <= free[-1]):
                    raise InvalidInput("Each basis vector ends with 1 at its free column, after the previous one's")
                budget.tick(len(free))
                if any(vector[column].numerator for column in free):
                    raise InvalidInput("Each basis vector is 0 at the free columns of the vectors before it")
                free.append(last)


@dataclass(frozen=True)
class RationalEigenvectors:
    """Every rational eigenvalue of a square rational matrix with its eigenspace, and the rest of the characteristic
    polynomial.

    The canonical form: eigenspaces is a tuple of Eigenspace records, strictly increasing in value, whose vectors all
    have one length; residual a nonempty tuple of Rationals, lowest degree first, monic (its last entry is 1), not of
    degree 1 and with a nonzero constant term (a monic polynomial with either has a rational root). With eigenspaces,
    the multiplicities and the degree of residual add up to the length of the vectors, the size of the matrix."""
    eigenspaces: tuple[Eigenspace, ...]
    residual: tuple[Rational, ...]

    def __post_init__(self):
        with computation() as budget:
            eigenspaces, residual = self.eigenspaces, self.residual
            if type(eigenspaces) is not tuple:
                raise InvalidInput("The eigenspaces are a tuple of Eigenspace records")
            if type(residual) is not tuple or not residual:
                raise InvalidInput("The residual is a nonempty tuple of Rationals, lowest degree first")
            for coefficient in residual:
                budget.tick()
                _exact_rational(budget, coefficient, "residual coefficient")
            if not _is_one(residual[-1]):
                raise InvalidInput("The residual is monic")
            if len(residual) == 2 or not residual[0].numerator:
                raise InvalidInput("The residual has no rational root: it is not of degree 1 and its constant term "
                                   "is not 0")
            previous, size, total = None, None, 0
            for space in eigenspaces:
                budget.tick()
                if type(space) is not Eigenspace:
                    raise InvalidInput("The eigenspaces are Eigenspace records")
                # Read defensively: a record made with object.__new__ skipped its own validation.
                value, multiplicity = getattr(space, "value", None), getattr(space, "multiplicity", None)
                basis = getattr(space, "basis", None)
                if (type(value) is not Rational or type(multiplicity) is not int or type(basis) is not tuple
                        or not basis or type(basis[0]) is not tuple):
                    raise InvalidInput("The eigenspaces are Eigenspace records")
                length = len(basis[0])
                if size is not None and length != size:
                    raise InvalidInput("Every eigenspace basis has vectors of one length, the size of the matrix")
                if previous is not None and not previous < value:
                    raise InvalidInput("The eigenspaces are strictly increasing in value")
                previous, size, total = value, length, total + multiplicity
            if size is not None and total + len(residual) - 1 != size:
                raise InvalidInput("The multiplicities and the degree of the residual add up to the size of the "
                                   "matrix")


TYPES = {"rational_eigenvectors": RationalEigenvectors, "eigenspace": Eigenspace}
CHECKER = "mathforge.certify.linalg.eigen"


_ONE = trusted_rational(1, 1)
_ZERO = trusted_rational(0, 1)
_NOT_A_MATRIX = "eigenvectors requires a square Matrix M of exact rationals"
_BITS = "Computation limit exceeded: integer bits"
# What the verifier counts among its evidence items besides the value (twice), the evidence (once) and the Matrix M
# (twice, in the request of the Result and in that of its step): the Result's own fields with its four stored checks
# (_RESULT_ITEMS, the count of mathforge.areas.ntheory.quadratic_irrational), and in each request the request itself,
# its operation name, its tuple of arguments, the pair ("M", M) and the name "M" (_REQUEST_ITEMS). The verifier walks
# the whole Result (Budget.inspect); tests/v09/linalg/test_eigen.py measures that walk and fails when the
# layout of contracts.Result changes.
_RESULT_ITEMS = 54
_REQUEST_ITEMS = 5


def produce(operation: str, args: dict[str, object]) -> tuple[object, tuple]:
    """(RationalEigenvectors, (charpoly, factorization, echelons)) for args = {"M": M}, computed as the module states.
    InvalidInput for an M that is no square Matrix of canonical Rationals; ResourceLimitError past the budget, whose
    max_matrix_entries is charged before anything but the class of M is validated."""
    if type(operation) is not str or operation != "eigenvectors":
        raise UnsupportedOperation(f"{operation!r} is not an operation of the eigen family")
    if type(args) is not dict or set(args) != {"M"}:
        raise InvalidInput("eigenvectors takes exactly the argument M")
    matrix = args["M"]
    with computation() as budget:
        budget.tick()  # The call itself.
        if type(matrix) is not Matrix:  # A subclass, the rows alone or any other object is no Matrix.
            raise InvalidInput(_NOT_A_MATRIX)
        rows, ncols = getattr(matrix, "rows", None), getattr(matrix, "ncols", None)  # A forged Matrix may lack them.
        if type(rows) is not tuple or type(ncols) is not int:
            raise InvalidInput(_NOT_A_MATRIX)
        budget.matrix(len(rows), ncols)  # First: a 65 x 65 M exceeds the budget before its shape is judged.
        n = len(rows)
        if ncols != n:
            raise InvalidInput(f"eigenvectors requires a square matrix M, not {n} x {ncols}")
        _validate_entries(budget, rows, n)
        if n == 0:
            value, evidence = RationalEigenvectors((), (_ONE,)), ((1,), (), ())
        else:
            value, evidence = _eigenvectors(budget, rows, n)
        _check_items(budget, n, value, evidence)
        return value, evidence


def _validate_entries(budget: Budget, rows: tuple, n: int) -> None:
    """InvalidInput for a row that is no tuple of n entries and an entry that is no Rational of two exact ints with a
    positive denominator in lowest terms (only a Matrix forged without its constructor holds one: the checker refuses
    an entry such as 4/2, whose s, and so the evidence, would depend on how it is written); ResourceLimitError for an
    integer longer than max_integer_bits."""
    for row in rows:
        budget.tick(1 + n)
        if type(row) is not tuple or len(row) != n:
            raise InvalidInput(_NOT_A_MATRIX)
        for entry in row:
            _exact_rational(budget, entry, "entry of M")
            if entry.denominator != 1 and _gcd(entry.numerator, entry.denominator) != 1:
                raise InvalidInput("The entries of M are Rationals in lowest terms")


def _gcd(a: int, b: int) -> int:
    """gcd(a, b), charged before it runs on the scale of trusted.upoly: the long division of the longer integer by the
    shorter one, then Lehmer's algorithm on the shorter length."""
    from ...trusted import upoly as tp

    longer, shorter = sorted((a.bit_length(), b.bit_length()), reverse=True)
    tp.charge_product(longer, shorter)
    tp.charge_gcd(shorter)
    return gcd(a, b)


def _lowest(numerator: int, denominator: int, limit: int) -> Rational:
    """numerator / denominator in lowest terms (denominator != 0), its gcd and quotients charged before they run;
    ResourceLimitError when the result has an integer longer than limit bits (the checker answers False to it)."""
    from ...trusted import upoly as tp

    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if denominator != 1 and numerator:
        divisor = _gcd(numerator, denominator)
        if divisor != 1:
            tp.charge_product(max(numerator.bit_length(), denominator.bit_length()), divisor.bit_length(), 2)
            numerator, denominator = numerator // divisor, denominator // divisor
    elif not numerator:
        denominator = 1
    if numerator.bit_length() > limit or denominator.bit_length() > limit:
        raise ResourceLimitError(_BITS)
    return trusted_rational(numerator, denominator)


def _product_extra(left_bits: int, right_bits: int) -> int:
    """What one product of integers of these lengths costs beyond one unit (trusted.upoly.size_factor - 1)."""
    return (left_bits >> 6) * (right_bits >> 6) >> 8


def _eigenvectors(budget: Budget, rows: tuple, n: int) -> tuple[RationalEigenvectors, tuple]:
    """The value and the evidence for a validated n x n M, n >= 1 (module docstring, "How produce() computes")."""
    from ...kernels import linalg as kernel
    from ...kernels import upoly as polynomials
    from ...trusted import upoly as tp

    limit = budget.limits.max_integer_bits
    # 1. The characteristic polynomial of s M (Berkowitz), and from it p(t) = sum_k C_k s^-k t^(n-k).
    s = _common_denominator(budget, rows, limit)
    scaled = _scaled(budget, rows, s)
    _charge_berkowitz(budget, scaled, n)
    charpoly = tuple(kernel.charpoly(scaled))
    budget.tick(1 + n)
    if any(c.bit_length() > limit for c in charpoly):
        raise ResourceLimitError(_BITS)  # An int of the evidence that the checker would answer False.
    powers = [1]
    for _ in range(n):
        budget.tick()
        tp.charge_product(powers[-1].bit_length(), s.bit_length())
        powers.append(powers[-1] * s)
    for j in range(n + 1):  # The coefficients of p in lowest terms: the checker refuses one past the limit.
        budget.tick()
        _lowest(charpoly[n - j], powers[n - j], limit)
    integer = []  # D(t) = s^n p(t), whose coefficient of t^j is C_(n-j) s^j; its primitive form is p's.
    for j in range(n + 1):
        budget.tick()
        tp.charge_product(charpoly[n - j].bit_length(), powers[j].bit_length())
        integer.append(charpoly[n - j] * powers[j])
    tp.charge_gcd(max(value.bit_length() for value in integer), n + 1)
    primitive = tp.primitive(tp.trim(integer))
    # 2. The factorization of p over Q, with v0.7's irreducibility evidence of each factor.
    found = []
    for part, multiplicity in _square_free_parts(primitive):
        _charge_lifting(budget, part)  # factor_square_free lifts the whole part first.
        factors = polynomials.factor_square_free(part)
        for factor, certificate in factors:
            if len(factor) > 2 and len(factors) > 1:
                # A factor split off a reducible part: its own lift data, as factor_with_evidence gives it (the part's
                # lifted factors would certify it too, but the evidence then would depend on how p splits into parts).
                _charge_lifting(budget, factor)
                prime, exponent, _, lifted = polynomials.lift_data(factor)
                certificate = ("modular", prime, exponent, tuple(lifted))
            found.append((factor, multiplicity, certificate))
    entries = []
    for factor, multiplicity, certificate in found:
        budget.tick(len(factor))
        lead = factor[-1]
        tp.charge_gcd(max(tp.max_bits(factor), 1), len(factor))
        monic = tuple(_lowest(value, lead, limit) for value in factor)
        entries.append((monic, multiplicity, certificate, factor))
    entries.sort(key=lambda entry: (len(entry[0]) - 1, tuple((c.numerator, c.denominator) for c in entry[0])))
    factorization = tuple((monic, multiplicity, certificate) for monic, multiplicity, certificate, _ in entries)
    # The rational eigenvalues, in increasing order, and the residual.
    eigenvalues = [(trusted_rational(-monic[0].numerator, monic[0].denominator), multiplicity)
                   for monic, multiplicity, _, _ in entries if len(monic) == 2]
    if len(eigenvalues) > 1:
        widest = max(max(value.numerator.bit_length(), value.denominator.bit_length()) for value, _ in eigenvalues)
        tp.charge_product(widest, widest, 2 * len(eigenvalues) * len(eigenvalues).bit_length())
        eigenvalues.sort(key=lambda item: item[0])  # Rational's own order: cross multiplication.
    product: tuple[int, ...] = (1,)
    for _, multiplicity, _, factor in entries:
        if len(factor) > 2:
            for _ in range(multiplicity):
                budget.tick()
                product = tp.mul(product, factor)  # Primitive by Gauss's lemma, with a positive leading coefficient.
    if len(product) == 1:
        residual: tuple[Rational, ...] = (_ONE,)
    else:
        budget.tick(len(product))
        tp.charge_gcd(tp.max_bits(product), len(product))
        residual = tuple(_lowest(value, product[-1], limit) for value in product)
    # 3. The eigenspaces, each with its echelon certificate.
    spaces, echelons = [], []
    for eigenvalue, multiplicity in eigenvalues:
        basis, pair = _eigenspace(budget, rows, n, eigenvalue, limit)
        spaces.append(Eigenspace(eigenvalue, multiplicity, basis))
        echelons.append(pair)
    value = RationalEigenvectors(tuple(spaces), residual)
    return value, (charpoly, factorization, tuple(echelons))


def _square_free_parts(f: tuple[int, ...]) -> list[tuple[tuple[int, ...], int]]:
    """kernels.upoly.square_free_integer(f) for the primitive integer form f of p (positive leading coefficient), with
    the square-free case decided first modulo word primes: [(f, 1)] when f mod q and f' mod q are coprime for a prime q
    that does not divide lc(f). That is sufficient: a square g^2 | f with g primitive of degree >= 1 (Gauss) gives
    g | f', and q does not divide lc(g), so g mod q keeps its degree and divides both residues. square_free_integer
    answers the same [(f, 1)] for such an f, but through a pseudo-remainder sequence whose leading coefficients pass
    max_integer_bits long before those of f do (a triangular 4 x 4 matrix of entries of at most 9 bits was refused there
    under 48-bit integers, although its claim fits); a characteristic polynomial with a repeated factor still takes
    that sequence."""
    from ...kernels import upoly as polynomials
    from ...trusted import upoly as tp
    from ...trusted.primes import MODULI

    if len(f) > 2:
        derivative = tp.derivative(f)
        bits = tp.max_bits(f) + len(f).bit_length()
        for prime in MODULI[:3]:  # A word prime divides the discriminant of f only by bad luck; three suffice.
            if f[-1] % prime:
                tp.charge_product(bits, prime.bit_length(), 2 * len(f))  # The residues of f and f'.
                reduced, slope = tp.mod(f, prime), tp.mod(derivative, prime)
                if slope and tp.gcd_mod(reduced, slope, prime) == (1,):
                    return [(f, 1)]
    return polynomials.square_free_integer(f)


def _charge_lifting(budget: Budget, f: tuple[int, ...]) -> None:
    """Charge, before kernels.upoly.lift_data(f) runs (itself, or as the first step of factor_square_free), the
    long-integer products of its Hensel lifting that trusted.upoly's charges, one unit per coefficient operation, leave
    out. The lifting goes one p-adic digit at a time, p = good_prime(f), from p to the first p^e above twice the
    coefficient bound of f: e - 1 steps, the k-th working modulo p^(k+1). Each step reduces lc(f)^-1 f (m = len(f)
    products by the coefficients of f, m reductions and one modular inverse), multiplies the lifted factors one after
    the other (at most m^2 / 2 products and m reductions) and adds the corrections (m reductions), all of integers of at
    most bitlen(p^(k+1)) bits. With
    16 log2(p) < L = bitlen(p^16), p^k has at most k L / 16 + 1 bits and passes 2^b once k (L - 1) / 16 >= b. A total
    past what is left of max_work is refused before the lifting starts: a 10 x 10 matrix of random 1,000-bit entries,
    whose characteristic polynomial lifts over 4,000 digits modulo 5, took 4 seconds before this charge and is now
    refused at once."""
    from ...kernels import upoly as polynomials
    from ...trusted import upoly as tp

    m = len(f)
    if m <= 2:
        return  # factor_square_free answers a linear part without lifting.
    prime, _ = polynomials.good_prime(f)
    coefficient_bits = tp.max_bits(f)
    tp.charge_product(coefficient_bits, coefficient_bits, m)  # The 2-norm of the coefficient bound.
    target = (2 * polynomials.coefficient_bound(f)).bit_length()
    log16 = (prime ** 16).bit_length()
    steps = 16 * target // (log16 - 1) + 1
    room = budget.limits.max_work - budget.work
    extra = 0
    for k in range(1, steps):
        size = ((k + 1) * log16 >> 4) + 1
        words = size >> 6
        extra += ((m * m // 2 + 3 * m) * (words * words >> 8) + m * _product_extra(coefficient_bits, size)
                  + (words * words >> 6))
        if extra > room:
            raise ResourceLimitError(f"Computation limit exceeded: work ({budget.limits.max_work})")
    if extra:
        budget.tick(extra)


def _common_denominator(budget: Budget, rows: tuple, limit: int) -> int:
    """s, the least common multiple of the denominators of all entries, grown one denominator at a time;
    ResourceLimitError once it is longer than limit bits (the checker's refusal), before a much longer lcm is formed."""
    from ...trusted import upoly as tp

    s = 1
    for row in rows:
        budget.tick(1 + len(row))
        for entry in row:
            denominator = entry.denominator
            if denominator != 1:
                tp.charge_product(s.bit_length(), denominator.bit_length(), 3)  # s % d, s // g and the product.
                if s % denominator:
                    s = s // _gcd(s, denominator) * denominator
                    if s.bit_length() > limit:
                        raise ResourceLimitError(_BITS)
    return s


def _scaled(budget: Budget, rows: tuple, s: int) -> tuple[tuple[int, ...], ...]:
    """The integer rows of s M (every denominator divides s)."""
    from ...trusted import upoly as tp

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


def _widest(rows) -> int:
    """The greatest bit length of the entries of integer rows."""
    return max((max(max(row), -min(row)).bit_length() for row in rows if row), default=0)


def _charge_berkowitz(budget: Budget, scaled: tuple, n: int) -> None:
    """Charge, before kernels.linalg.charpoly runs, the long-integer products that its ticks (1 + size^3 per trailing
    block) leave out. With entries of at most w bits and g = w + bitlen(n), the k-th vector of a block of the given
    size has entries of at most w + k g bits, a product of it with an entry adds at most g, and after a block the
    coefficients have at most size * g bits: each product of the block's vector loop and of its Toeplitz convolution is
    charged by those lengths (trusted.upoly.size_factor - 1). Entries of a few dozen bits cost nothing more. The
    kernel's own ticks are known in advance, so a request that they would take past max_work is refused before
    Berkowitz starts, rather than after most of its work."""
    w = _widest(scaled)
    grow = w + n.bit_length()
    extra = 0
    for size in range(2, n + 1):
        inner = size - 1
        for k in range(size - 1):
            extra += inner * (inner + 1) * _product_extra(w, w + k * grow)
        extra += (size + 1) * size * _product_extra(w + size * grow, (size - 1) * grow)
    ticks = sum(1 + size * size * size for size in range(1, n + 1))
    if budget.work + extra + ticks > budget.limits.max_work:
        raise ResourceLimitError(f"Computation limit exceeded: work ({budget.limits.max_work})")
    if extra:
        budget.tick(extra)


def _charge_elimination(budget: Budget, int_rows: tuple, n: int, limit: int) -> None:
    """Charge, before kernels.linalg.echelon runs, the long-integer products that its ticks (one row length per row
    rewrite) leave out: after k pivots its entries are k x k minors, of at most k (w + bitlen(n)) bits (Hadamard) and
    never much past limit bits (it refuses a longer pivot), and each of the (n - 1) n entries that pivot rewrites takes
    two products and a division of that length."""
    w = _widest(int_rows)
    grow = w + n.bit_length()
    extra = 0
    for k in range(1, n + 1):
        bits = min(k * grow, limit + grow)
        extra += 3 * (n - 1) * n * _product_extra(bits, bits)
    if extra:
        budget.tick(extra)


def _eigenspace(budget: Budget, rows: tuple, n: int, eigenvalue: Rational,
                limit: int) -> tuple[tuple[tuple[Rational, ...], ...], tuple[tuple[int, ...], int]]:
    """The canonical nullspace basis of M - eigenvalue I and its (basis_rows, modulus): the rows of M - eigenvalue I
    each scaled by the lcm of its denominators (trusted.intmatrix.clear_row_denominators, the rows the checker forms),
    their reduced row echelon form by kernels.linalg.echelon, and the basis read from its numerators over its
    denominator. ResourceLimitError for a diagonal entry or a row scale past limit bits (the checker's refusals) and for
    a basis entry past it (the checker would answer False)."""
    from ...kernels import linalg as kernel
    from ...trusted import upoly as tp
    from ...trusted.intmatrix import clear_row_denominators

    shifted = []
    for i, row in enumerate(rows):
        budget.tick(1 + n)
        entry = row[i]
        numerator_bits = max(entry.numerator.bit_length(), eigenvalue.numerator.bit_length())
        denominator_bits = max(entry.denominator.bit_length(), eigenvalue.denominator.bit_length())
        tp.charge_product(numerator_bits, denominator_bits, 3)
        tp.charge_gcd(numerator_bits + denominator_bits, 2)
        shifted_row = row[:i] + (entry - eigenvalue,) + row[i + 1:]  # Rational subtraction refuses a long result.
        # The row scale: at most the product of the denominators and at most limit bits (else refused); scaling every
        # entry by it, and the lcms that form it.
        scale_bits = min(sum(value.denominator.bit_length() for value in shifted_row), limit)
        widest = max(value.numerator.bit_length() for value in shifted_row)
        tp.charge_product(widest, scale_bits, n)
        tp.charge_gcd(scale_bits, n)
        shifted.append(shifted_row)
    int_rows, _ = clear_row_denominators(shifted)
    _charge_elimination(budget, int_rows, n, limit)
    try:
        reduced = kernel.echelon(int_rows, n)
    except ResourceLimitError:
        # The fraction-free elimination refuses a pivot past limit bits: its last pivot is the whole r x r minor,
        # however small the reduced rows are ([[0, 0, a], [b, 0, c], [0, 0, 0]] with 31-bit a and b has the pivot a b,
        # 62 bits, and the reduced rows of 0s and 1s). Gauss-Jordan over Q keeps every entry in lowest terms instead;
        # it refuses only an entry of its own past the limit, and a request past max_work stays refused.
        return _rational_eigenspace(budget, int_rows, n, limit)
    pivots, numerators, denominator = reduced.pivots, reduced.numerators, reduced.denominator
    pivot_set = set(pivots)
    free = [column for column in range(n) if column not in pivot_set]
    if not free:
        raise ArithmeticError("A root of the characteristic polynomial without an eigenvector")
    basis = []
    for column in free:
        budget.tick(1 + n)
        vector = [_ZERO] * n
        vector[column] = _ONE
        for numerator_row, pivot in zip(numerators, pivots):
            value = numerator_row[column]
            if value:
                vector[pivot] = _lowest(-value, denominator, limit)
        basis.append(tuple(vector))
    return tuple(basis), (reduced.basis_rows, _fitting_modulus(budget, reduced.denominator, reduced.modulus, limit))


def _fitting_modulus(budget: Budget, determinant: int, modulus: int, limit: int) -> int:
    """The kernel's modulus, a word prime below 2**61, when it has at most limit bits; else the least prime that does
    not divide determinant, the kernel's denominator: the determinant, up to sign, of the minor at basis_rows and the
    pivot columns, so its residue is a unit modulo that prime, which is all that the checker asks of the modulus. The
    verifier refuses an int of the evidence past max_integer_bits, so without this every eigenvalue of a matrix whose
    M - lambda I has a nonzero row was refused under max_integer_bits < 61. The primes tried before the answer all
    divide determinant, so there are at most bitlen(determinant) of them; ResourceLimitError when the answer itself
    is longer than limit bits (then so is any modulus >= 2)."""
    if modulus.bit_length() <= limit:
        return modulus
    from ...trusted import upoly as tp

    bits = determinant.bit_length()
    candidate = 2
    while True:
        budget.tick(1 + candidate.bit_length())
        if candidate.bit_length() > limit:
            raise ResourceLimitError(_BITS)
        if all(candidate % divisor for divisor in range(2, isqrt(candidate) + 1)):  # Trial division: tiny.
            tp.charge_product(bits, candidate.bit_length())
            if determinant % candidate:
                return candidate
        candidate += 1


def _bits(values) -> int:
    """The greatest bit length of a numerator or a denominator of the Rationals."""
    return max(max(value.numerator.bit_length(), value.denominator.bit_length()) for value in values)


def _rational_eigenspace(budget: Budget, int_rows: tuple, n: int,
                         limit: int) -> tuple[tuple[tuple[Rational, ...], ...], tuple[tuple[int, ...], int]]:
    """The canonical nullspace basis of the n x n integer rows A_l (those of M - lambda I, row-scaled) and its
    (basis_rows, modulus), by Gauss-Jordan elimination over Q: the kernel's choice of pivot rows (the first row at or
    below the pivot position with a nonzero entry, rows swapped as it swaps them), each pivot row divided by its pivot,
    every other row cleared in that column. Each new entry is formed over one denominator and reduced once (_lowest),
    refused when it is past limit bits in lowest terms; each row operation is charged by the size of its entries before
    it runs. basis_rows are the original
    indices of the pivot rows, whose minor at the pivot columns is nonsingular, and the modulus a prime of at most limit
    bits that does not divide its determinant (_minor_modulus)."""
    from ...trusted import upoly as tp
    from ...trusted.intmatrix import integer

    rows = [[integer(value) for value in row] for row in int_rows]
    order = list(range(n))
    pivots: list[int] = []
    for column in range(n):
        r = len(pivots)
        if r == n:
            break
        budget.tick(1 + n)
        found = next((i for i in range(r, n) if rows[i][column].numerator), None)
        if found is None:
            continue
        rows[r], rows[found] = rows[found], rows[r]
        order[r], order[found] = order[found], order[r]
        pivot_row = rows[r]
        pivot = pivot_row[column]
        budget.tick(n)
        bits = _bits(pivot_row)
        tp.charge_product(bits, bits, 2 * n)  # A quotient: two products and a gcd per entry.
        tp.charge_gcd(2 * bits, n)
        a, b = pivot.numerator, pivot.denominator
        pivot_row = rows[r] = [_lowest(value.numerator * b, value.denominator * a, limit) for value in pivot_row]
        pivot_bits = _bits(pivot_row)
        for i in range(n):
            row = rows[i]
            if i == r or not row[column].numerator:
                continue
            c, d = row[column].numerator, row[column].denominator
            budget.tick(n)
            bits = max(_bits(row), pivot_bits)
            tp.charge_product(2 * bits, 2 * bits, 6 * n)  # value - factor * other, over one denominator.
            # Each entry in one step, value - (c/d) (e/f) = (value.numerator d f - value.denominator c e) /
            # (value.denominator d f), reduced once: only the entries of the state are bounded, not their products.
            rows[i] = [_lowest(value.numerator * d * other.denominator - value.denominator * c * other.numerator,
                               value.denominator * d * other.denominator, limit)
                       for value, other in zip(row, pivot_row)]
        pivots.append(column)
    pivot_set = set(pivots)
    free = [column for column in range(n) if column not in pivot_set]
    if not free:
        raise ArithmeticError("A root of the characteristic polynomial without an eigenvector")
    basis = []
    for column in free:
        budget.tick(1 + n)
        vector = [_ZERO] * n
        vector[column] = _ONE
        for reduced_row, pivot in zip(rows, pivots):
            value = reduced_row[column]
            if value.numerator:
                vector[pivot] = trusted_rational(-value.numerator, value.denominator)
        basis.append(tuple(vector))
    basis_rows = tuple(order[:len(pivots)])
    return tuple(basis), (basis_rows, _minor_modulus(budget, int_rows, basis_rows, tuple(pivots), limit))


def _minor_modulus(budget: Budget, int_rows: tuple, basis_rows: tuple[int, ...], pivots: tuple[int, ...],
                   limit: int) -> int:
    """A prime of at most limit bits that does not divide the determinant of the nonsingular minor of int_rows at
    basis_rows and pivots, tested by elimination modulo each candidate: the word primes of trusted.primes, below 2**61,
    when limit allows them, else the primes below 2**limit in decreasing order. A candidate that fails divides the
    determinant, which has at most H = sum over the rows of (bitlen of the widest entry + bitlen(r)) bits (Hadamard),
    so at most H / (bitlen(candidate) - 1) candidates fail. 2 for rank 0, whose minor is empty; ResourceLimitError when
    every prime of at most limit bits divides the determinant."""
    from ...trusted import upoly as tp
    from ...trusted.primes import moduli, primes_below

    r = len(pivots)
    if not r:
        return 2
    minor = [[int_rows[i][j] for j in pivots] for i in basis_rows]
    bits = _widest(minor)
    for prime in (moduli() if limit >= 61 else primes_below(1 << limit)):
        budget.tick(1 + r * r * r)
        tp.charge_product(bits, prime.bit_length(), r * r)
        if _nonsingular_mod(minor, prime):
            return prime
    raise ResourceLimitError(_BITS)


def _nonsingular_mod(minor: list[list[int]], prime: int) -> bool:
    """Whether a square integer matrix is nonsingular modulo a prime, by Gaussian elimination over GF(prime)."""
    rows = [[value % prime for value in row] for row in minor]
    size = len(rows)
    for column in range(size):
        found = next((i for i in range(column, size) if rows[i][column]), None)
        if found is None:
            return False
        rows[column], rows[found] = rows[found], rows[column]
        inverse = pow(rows[column][column], -1, prime)
        pivot_row = rows[column]
        for i in range(column + 1, size):
            factor = rows[i][column] * inverse % prime
            if factor:
                rows[i] = [(a - factor * b) % prime for a, b in zip(rows[i], pivot_row)]
    return True


def _check_items(budget: Budget, n: int, value: RationalEigenvectors, evidence: tuple) -> None:
    """ResourceLimitError when the verifier would not admit the stored Result under max_evidence_items: it counts the
    Result's own items (_RESULT_ITEMS), the request twice, the value twice and the evidence once, as Budget.inspect
    walks them; a separate budget of the same limits counts them here."""
    budget.tick(1 + n)
    counter = Budget(budget.limits)
    request = _REQUEST_ITEMS + 3 + n + 3 * n * n  # The Matrix: itself, its rows tuple, n rows, n^2 Rationals, ncols.
    counter.evidence(_RESULT_ITEMS + 2 * request)
    counter.inspect(value, evidence=True)
    counter.inspect(value, evidence=True)
    counter.inspect(evidence, evidence=True)
    budget.tick(counter.evidence_count >> 4)  # The walk itself.


def _polynomial(coefficients: tuple[Rational, ...], renderer, child) -> str:
    r"""x^2 - 6*x + 6 from (6, -6, 1), in the card's variable x: the terms from the highest power down, zero terms left
    out; in LaTeX x^{2} - 6\, x + 6. The polynomial is the residual, monic of degree >= 2, so its first term is a power
    of x."""
    latex = renderer.latex
    parts: list[str] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if not coefficient.numerator:
            continue
        negative = coefficient.numerator < 0
        magnitude = abs(coefficient)
        if power == 0:
            term = child(magnitude)
        else:
            digits = renderer.integer(power)
            monomial = "x" if power == 1 else "x" + ("^{" + digits + "}" if latex else "^" + digits)
            term = monomial if _is_one(magnitude) else renderer.join((child(magnitude), r"\, " if latex else "*",
                                                                      monomial))
        if parts:
            parts.append(" - " if negative else " + ")
        elif negative:
            parts.append("-")
        parts.append(term)
    return renderer.join(parts)


def _render_eigenspace(space: Eigenspace, renderer, child) -> str:
    r"""3 (multiplicity 1): [(1, 1)]; in LaTeX 3\ \left(\text{multiplicity }1\right)\colon\ \left[\left(1,
    1\right)\right]."""
    vectors = renderer.join((child(vector) for vector in space.basis), ", ")
    multiplicity = renderer.parens(renderer.join((renderer.label("multiplicity "),
                                                  renderer.integer(space.multiplicity))))
    if renderer.latex:
        return renderer.join((child(space.value), r"\ ", multiplicity, r"\colon\ \left[", vectors, r"\right]"))
    return renderer.join((child(space.value), " ", multiplicity, ": [", vectors, "]"))


def _render_eigenvectors(value: RationalEigenvectors, renderer, child) -> str:
    r"""1 (multiplicity 1): [(-1, 1)]; 3 (multiplicity 1): [(1, 1)], each eigenspace in turn, then the residual
    unless it is 1: no rational eigenvalue; residual x^2 - 2. In LaTeX the parts are separated by ;\quad and the
    words are \text{...}."""
    parts = [child(space) for space in value.eigenspaces] or [renderer.label("no rational eigenvalue")]
    if len(value.residual) > 1:
        parts.append(renderer.join((renderer.label("residual "), _polynomial(value.residual, renderer, child))))
    return renderer.join(parts, r";\quad " if renderer.latex else "; ")


RENDER = {RationalEigenvectors: _render_eigenvectors, Eigenspace: _render_eigenspace}
