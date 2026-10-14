"""Reproducible exact SDK timings against the optional development SymPy oracle.

Run each backend sequentially on an otherwise idle machine. MathForge timings
include automatic proof production and verification; SymPy returns a value only.
These are API latency measurements, not equivalent algorithm-only costs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
import tempfile
from time import perf_counter_ns
import zipfile


INPUTS = {
    "rref_16": "16x16 tridiagonal integer matrix: diagonal 17, adjacent diagonals 1",
    "real_roots_quintic": "All real roots and multiplicities of x**5 - x - 1",
    "evaluate_algebraic_point": "(x+1)/(x**2+1) evaluated at x=sqrt(2)",
    "algebraic_double_pole": "Two-sided limit of 1/(x**2-2)**2 at x=sqrt(2)",
}


def mathforge_cases(mf):
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000707")
    rows = tuple(tuple(17 if i == j else int(abs(i-j) == 1)
                       for j in range(16)) for i in range(16))
    matrix = mf.Matrix(rows)
    polynomial = x**5-x-1
    point = mf.sqrt(2)
    function = mf.rational_function(x+1, x*x+1, variable=x)
    pole = mf.rational_function(1, (x*x-2)**2, variable=x)
    expected_point = mf.algebraic((point+1)/3)

    def rref_ok(value):
        return (value.pivot_columns == tuple(range(16))
                and all(value.matrix[i, j] == mf.Rational(int(i == j))
                        for i in range(16) for j in range(16)))

    def roots_ok(value):
        return (len(value) == 1 and value[0].multiplicity == 1
                and value[0].root == mf.RealAlgebraicRoot((-1, -1, 0, 0, 0, 1), 0))

    return {
        "rref_16": (lambda: mf.operations.rref(matrix), rref_ok),
        "real_roots_quintic": (lambda: mf.operations.real_roots(polynomial, x), roots_ok),
        "evaluate_algebraic_point": (
            lambda: mf.operations.evaluate_algebraic(function, {x: point}),
            lambda value: mf.compare_real(value, expected_point) == 0),
        "algebraic_double_pole": (
            lambda: mf.operations.limit(pole, x, point, side="both"),
            lambda value: value.kind == "positive_infinity" and value.value is None),
    }


def sympy_cases(sp):
    x = sp.Symbol("x", real=True)
    matrix = sp.Matrix(16, 16, lambda i, j: 17 if i == j else int(abs(i-j) == 1))
    polynomial = x**5-x-1
    function = (x+1)/(x*x+1)
    point = sp.sqrt(2)
    pole = 1/(x*x-2)**2
    expected_roots = [(sp.CRootOf(x**5-x-1, 0), 1)]
    return {
        "rref_16": (lambda: matrix.rref(),
                    lambda value: value == (sp.eye(16), tuple(range(16)))),
        "real_roots_quintic": (lambda: sp.real_roots(polynomial, multiple=False),
                               lambda value: value == expected_roots),
        "evaluate_algebraic_point": (
            lambda: function.subs(x, point),
            lambda value: sp.simplify(value-(1+point)/3) == 0),
        "algebraic_double_pole": (
            lambda: sp.limit(pole, x, point, dir="+-"),
            lambda value: value == sp.oo),
    }


def summarize(samples):
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[(95*len(ordered)+99)//100-1],
    }


def source_hash(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mathforge", "sympy"), required=True)
    parser.add_argument("--snapshot", type=Path, help="Frozen MathForge source ZIP")
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmups < 1:
        parser.error("repeat and warmups must be positive")
    if args.snapshot and args.backend != "mathforge":
        parser.error("snapshot only applies to MathForge")

    with tempfile.TemporaryDirectory(prefix="mathforge-speed-") as temporary:
        source = Path(__file__).resolve().parents[1]/"src"
        if args.snapshot:
            source = Path(temporary)/"src"
            with zipfile.ZipFile(args.snapshot) as archive:
                for member in archive.infolist():
                    if not member.filename.startswith("src/mathforge/") or member.is_dir():
                        continue
                    target = (Path(temporary)/member.filename).resolve()
                    if not target.is_relative_to(Path(temporary).resolve()):
                        raise ValueError("Snapshot contains an unsafe path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
        mf = None
        if args.backend == "mathforge":
            sys.path.insert(0, str(source))
            import mathforge as mf
            cases = mathforge_cases(mf)
            version = mf.__version__
            modes = ("normal_sdk",)
            reset = lambda: None
            library_path = str(Path(mf.__file__).resolve())
            fingerprint = source_hash(Path(mf.__file__).parent)
        else:
            import sympy as sp
            from sympy.core.cache import clear_cache
            cases = sympy_cases(sp)
            version = sp.__version__
            modes = ("reset_global_caches", "warm_cache")

            def reset():
                clear_cache()
                sp.CRootOf.clear_cache()

            library_path = str(Path(sp.__file__).resolve())
            fingerprint = None

        records = []
        for name, (operation, valid) in cases.items():
            for mode in modes:
                for _ in range(args.warmups):
                    if mode == "reset_global_caches":
                        reset()
                    operation()
                # Validation is outside timing; retain every result until after
                # sampling so validators cannot populate a subsequent hot cache.
                gc.collect()
                results, samples = [], []
                for _ in range(args.repeat):
                    if mode == "reset_global_caches":
                        reset()
                    start = perf_counter_ns()
                    value = operation()
                    samples.append((perf_counter_ns()-start)/1_000_000)
                    results.append(value)
                for result in results:
                    if mf is not None:
                        assert result.exactness is mf.Exactness.EXACT
                        assert result.completeness is mf.Completeness.COMPLETE
                        assert result.verification and all(
                            check.status is mf.CheckStatus.VERIFIED
                            for check in result.verification)
                        assert valid(result.unwrap()), (name, result)
                    else:
                        assert valid(result), (name, result)
                if mf is not None:
                    assert mf.verify(results[-1]).verified
                record = dict(case=name, input=INPUTS[name], cache_mode=mode,
                              all_results_validated=True, **summarize(samples))
                records.append(record)
                print(f"{args.backend} {version} {name} {mode}: "
                      f"{record['median_ms']:.4f} ms", flush=True)

        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "version": version,
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "library_path": library_path,
            "source_sha256": fingerprint,
            "snapshot": str(args.snapshot) if args.snapshot else None,
            "repeat": args.repeat, "warmups_per_case_per_mode": args.warmups,
            "clock": "perf_counter_ns", "gc_enabled_during_timing": gc.isenabled(),
            "result_retention": "Each block retains its results until validation after timing",
            "scope": "Prebuilt input; SDK call only; no parse/render/JSON/memory tracing",
            "mathforge_scope": "Automatic evidence production and independent verification included",
            "sympy_scope": "Exact value computation without MathForge-style proof verification",
            "cache_note": "SymPy reset clears global and CRootOf caches outside timing; "
                          "this is not a cold process. MathForge root cache is operation-local.",
            "historical_table_note": "Historical timing script was not found. RREF and quintic "
                                     "use exact_core fixtures; point and pole use explicit "
                                     "representative inputs, not a claimed reconstruction.",
            "unmeasured": [{"case": "algebraic_endpoint_function_range",
                            "reason": "No public general function-range API or historical fixture found"}],
            "measurements": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
