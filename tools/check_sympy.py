"""REL-13: Optional development oracle, never imported by the runtime package.

Install the pinned oracle with ``pip install sympy==1.14.0`` and run this script.
Core tests and package installation have no dependency on this check.
"""

from random import Random

import mathforge as mf


def main() -> None:
    try:
        import sympy as sp
    except ImportError as exc:
        raise SystemExit("Optional oracle is absent; install sympy==1.14.0 to run this check") from exc
    random = Random(20409)
    for case in range(100):
        size = random.randint(1, 4)
        rows = tuple(tuple(random.randint(-5, 5) for _ in range(size)) for _ in range(size))
        actual, oracle = mf.Matrix(rows), sp.Matrix(rows)
        assert mf.rank(actual) == oracle.rank(), (case, rows)
        expected_det = oracle.det()
        assert mf.det(actual) == mf.Rational(int(expected_det.p), int(expected_det.q)), (case, rows)
        oracle_reduced, pivots = oracle.rref()
        reduced = mf.rref(actual)
        expected_rows = tuple(tuple(mf.Rational(int(oracle_reduced[i, j].p), int(oracle_reduced[i, j].q))
                                    for j in range(size)) for i in range(size))
        assert reduced.matrix == mf.Matrix(expected_rows) and reduced.pivot_columns == pivots
    print("100 bounded exact matrix cases agree with the optional SymPy oracle.")


if __name__ == "__main__":
    main()
