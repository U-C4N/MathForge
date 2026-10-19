"""Independent checkers of the linear algebra kernel (mathforge.kernels.linalg).

They verify certificates in exact integer and modular arithmetic and never rerun an elimination of the producer:
a determinant through a Hadamard bound and residues modulo word-sized primes (determinant), an inverse through the
integer product A_int B_int = diag(s t) (inverse), a reduced row echelon form through the echelon certificate
(echelon), a PLU factorization through its integer product (lu), a characteristic polynomial through its values
modulo primes (charpoly) and a trace through its sum (trace). The integer forms come from mathforge.trusted.intmatrix.
"""


def square_rows(A):
    """The rows of a square matrix given as a Matrix (its rows and ncols) or as a tuple of row tuples, every entry an
    exact Rational; None for anything else. A record class is never imported: a checker reads its fields."""
    from mathforge.model import Rational
    rows = getattr(A, "rows", A)
    if type(rows) is not tuple:
        return None
    n = len(rows)
    ncols = getattr(A, "ncols", n)
    if ncols != n or type(ncols) is not int:
        return None
    for row in rows:
        if type(row) is not tuple or len(row) != n:
            return None
        for value in row:
            if type(value) is not Rational:
                return None
    return rows


def matrix_rows(A):
    """The rows and the column count of a matrix given as a Matrix or as (rows, ncols), every entry an exact Rational;
    None for anything else."""
    from mathforge.model import Rational
    if type(A) is tuple and len(A) == 2 and type(A[1]) is int:
        rows, ncols = A
    else:
        rows, ncols = getattr(A, "rows", None), getattr(A, "ncols", None)
    if type(rows) is not tuple or type(ncols) is not int or ncols < 0:
        return None
    for row in rows:
        if type(row) is not tuple or len(row) != ncols:
            return None
        for value in row:
            if type(value) is not Rational:
                return None
    return rows, ncols
