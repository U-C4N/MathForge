"""Fresh, exact public-API comparison: MathForge 0.7 and optional SymPy.

Run with the development environment containing SymPy, for example:
  python benchmarks/fresh_comparison.py --output benchmarks/v07-fresh-comparison.json

Sixteen explicit fixtures, sequential isolated process rounds, interleaved
backend order, exact output validation, and raw samples. No timing CI threshold.
MathForge includes its automatic evidence/verification contract; SymPy does not.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from math import comb, ceil
import os
from pathlib import Path
import platform
from random import Random
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter_ns


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Case:
    name: str
    label: str
    inputs: str
    mathforge: object
    sympy: object
    normalize_mf: object
    validate: object
    mf_api: str
    sp_api: str


def fixtures(mf, sp):
    from sympy.stats import Binomial, FiniteRV, density, variance

    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000717")
    sx = sp.Symbol("x", real=True)
    st = sp.Symbol("t")

    def convert(value):
        if type(value) is mf.Rational:
            return sp.Rational(value.numerator, value.denominator)
        if type(value) is mf.RealAlgebraicRoot:
            p = sum(c*st**i for i, c in enumerate(value.integer_coefficients))
            return sp.Poly(p, st).real_roots()[value.real_index]
        if type(value) is mf.RealFunction:
            return convert(value.expression)
        if type(value) is mf.RealExpr:
            if value.op == "scalar": return convert(value.value)
            if value.op == "symbol":
                assert value.value == x
                return sx
            if value.op == "pi": return sp.pi
            if value.op == "E": return sp.E
            args = tuple(convert(a) for a in value.args)
            if value.op == "pow": return args[0]**value.value
            if value.op == "add": return args[0]+args[1]
            if value.op == "mul": return args[0]*args[1]
            if value.op == "div": return args[0]/args[1]
            return (sp.Abs if value.op == "abs" else getattr(sp, value.op))(*args)
        if type(value) is int: return sp.Integer(value)
        raise TypeError(type(value))

    def equal(expected):
        return lambda value: value == expected

    def symbolic(expected):
        return lambda value: not value.has(sp.Integral, sp.Derivative, sp.Order) and sp.simplify(value-expected) == 0

    def parse(text):
        return mf.parse_function(text, symbols={"x": x})

    tri = tuple(tuple(17 if i == j else int(abs(i-j) == 1) for j in range(16)) for i in range(16))
    ma, sa = mf.Matrix(tri), sp.Matrix(tri)
    dense = tuple(tuple(50 if i == j else (3*i+5*j)%11-5 for j in range(8)) for i in range(8))
    md, sd = mf.Matrix(dense), sp.Matrix(dense)
    det_expected = sd.det(method="domain-ge")
    rows = tuple((1, t, t*t) for t in range(-2, 3))
    rhs = (10, -2, 7, 2, 18)
    ml, sl, sb = mf.Matrix(rows), sp.Matrix(rows), sp.Matrix(rhs)

    def fit():
        solution = sl.solve_least_squares(sb, method="CH")
        return tuple(solution), (), tuple(sb-sl*solution)

    def mf_fit(value):
        return (tuple(convert(c) for c in value.particular),
                tuple(tuple(convert(c) for c in row) for row in value.basis),
                tuple(convert(c) for c in value.residual))

    quintic, sq = x**5-x-1, sx**5-sx-1
    root_expected = [(sp.CRootOf(sq, 0), 1)]
    eval_source = mf.rational_function(x+1, x*x+1, variable=x)
    eval_sp = (sx+1)/(sx*sx+1)
    point2, spoint2 = mf.sqrt(2), sp.sqrt(2)
    pole = mf.rational_function(1, (x*x-3)**2, variable=x)
    spole, point3, spoint3 = 1/(sx*sx-3)**2, mf.sqrt(3), sp.sqrt(3)
    derivative, sderivative = parse("exp(x)*sin(2*x)+cos(x*x)"), sp.exp(sx)*sp.sin(2*sx)+sp.cos(sx*sx)
    derivative_expected = (8*sx**3*sp.sin(sx*sx)-12*sx*sp.cos(sx*sx)
                           -11*sp.exp(sx)*sp.sin(2*sx)-2*sp.exp(sx)*sp.cos(2*sx))
    taylor, staylor = parse("exp(x)*sin(x)+log(1+x)"), sp.exp(sx)*sp.sin(sx)+sp.log(1+sx)
    taylor_expected = 2*sx+sx**2/2+2*sx**3/3-sx**4/4+sx**5/6-sp.Rational(8, 45)*sx**6
    primitive, sprimitive = parse("(x**3-2*x+1)*exp(2*x)"), (sx**3-2*sx+1)*sp.exp(2*sx)
    definite, sdefinite = parse("1/(1+x*x)**2"), 1/(1+sx*sx)**2
    sinc, ssinc = parse("sin(x)/x"), sp.sin(sx)/sx
    complex_value, scomplex = mf.complex_algebraic(3, 4), 3+4*sp.I
    p, sp_p = mf.Rational(2, 7), sp.Rational(2, 7)
    weights = {sp.Integer(k): sp.Rational(comb(40, k)*2**k*5**(40-k), 7**40) for k in range(41)}
    support = ((-7, 3), (-2, 1), (-1, 5), (0, 1), (2, 7), (4, 3), (3, 1), (11, 2))
    outcomes = tuple(mf.Rational(a, b) for a, b in support)
    soutcomes = tuple(sp.Rational(a, b) for a, b in support)
    distribution = mf.FiniteDistribution(outcomes, (mf.Rational(1, 8),)*8)
    rv = FiniteRV("V", {v: sp.Rational(1, 8) for v in soutcomes})
    avg = sum(soutcomes)/8
    variance_expected = sum((v-avg)**2 for v in soutcomes)/8

    def primitive_ok(value):
        return not value.has(sp.Integral) and sp.simplify(sp.diff(value, sx)-sprimitive) == 0

    cases = [
        Case("rref16", "16x16 RREF", "A[i,i]=17; adjacent diagonals=1; other entries=0",
             lambda: mf.operations.rref(ma), lambda: sa.rref(),
             lambda v: (sp.Matrix([[convert(c) for c in row] for row in v.matrix.rows]), v.pivot_columns),
             equal((sp.eye(16), tuple(range(16)))), "operations.rref(A)", "A.rref()"),
        Case("det8", "8x8 dense determinant", "A[i,i]=50; other A[i,j]=(3*i+5*j)%11-5; zero-based",
             lambda: mf.operations.det(md), lambda: sd.det(method="bareiss"), convert,
             equal(det_expected), "operations.det(A)", "A.det(method='bareiss')"),
        Case("least_squares", "5x3 least squares", "A rows=(1,t,t^2), t=-2..2; b=(10,-2,7,2,18)",
             lambda: mf.operations.least_squares(ml, rhs), fit, mf_fit,
             equal(((1, 2, 3), (), (1, -4, 6, -4, 1))), "operations.least_squares(A,b)",
             "A.solve_least_squares(b,method='CH') plus residual b-A*x"),
        Case("quintic_roots", "Quintic real roots", "x^5-x-1; expression inputs; all real roots with multiplicity",
             lambda: mf.operations.real_roots(quintic, x), lambda: sp.real_roots(sq, multiple=False),
             lambda v: [(convert(r.root), r.multiplicity) for r in v], equal(root_expected),
             "operations.real_roots(expression,x)", "real_roots(expression,multiple=False)"),
        Case("algebraic_eval", "Algebraic evaluation", "(x+1)/(x^2+1) at x=sqrt(2)",
             lambda: mf.operations.evaluate_algebraic(eval_source, {x: point2}),
             lambda: eval_sp.subs(sx, spoint2), convert, symbolic((1+spoint2)/3),
             "operations.evaluate_algebraic(f,{x:sqrt(2)})", "f.subs(x,sqrt(2))"),
        Case("double_pole", "Algebraic double-pole limit", "1/(x^2-3)^2 as x approaches sqrt(3), both sides",
             lambda: mf.operations.limit(pole, x, point3, side="both"),
             lambda: sp.limit(spole, sx, spoint3, dir="+-"),
             lambda v: sp.oo if v.kind == "positive_infinity" and v.value is None else None,
             equal(sp.oo), "operations.limit(f,x,sqrt(3),side='both')", "limit(f,x,sqrt(3),dir='+-')"),
        Case("derivative3", "Third derivative", "d^3/dx^3 [exp(x)*sin(2*x)+cos(x^2)]",
             lambda: mf.operations.function_differentiate(derivative, x, n=3),
             lambda: sp.diff(sderivative, sx, 3), convert, symbolic(derivative_expected),
             "operations.function_differentiate(f,x,n=3)", "diff(f,x,3)"),
        Case("taylor6", "Taylor degree 6", "exp(x)*sin(x)+log(1+x); center=0; degree=6",
             lambda: mf.operations.taylor(taylor, x, point=0, order=6),
             lambda: sp.series(staylor, sx, 0, 7).removeO(), lambda v: convert(v.polynomial),
             symbolic(taylor_expected), "operations.taylor(f,x,point=0,order=6)", "series(f,x,0,7).removeO()"),
        Case("elementary_integral", "Polynomial-exponential primitive", "Integral of (x^3-2*x+1)*exp(2*x)",
             lambda: mf.operations.function_antiderivative(primitive, x), lambda: sp.integrate(sprimitive, sx),
             convert, primitive_ok, "operations.function_antiderivative(f,x)", "integrate(f,x)"),
        Case("definite_integral", "Proper rational integral", "Integral of 1/(1+x^2)^2 from 0 to 1",
             lambda: mf.operations.function_integrate(definite, x, 0, 1),
             lambda: sp.integrate(sdefinite, (sx, 0, 1)), convert, symbolic(sp.Rational(1, 4)+sp.pi/8),
             "operations.function_integrate(f,x,0,1)", "integrate(f,(x,0,1))"),
        Case("sinc_limit", "Sinc limit", "sin(x)/x as x approaches 0, both sides",
             lambda: mf.operations.function_limit(sinc, x, 0, side="both"),
             lambda: sp.limit(ssinc, sx, 0, dir="+-"),
             lambda v: convert(v.value) if v.kind == "finite" else None, equal(sp.Integer(1)),
             "operations.function_limit(f,x,0,side='both')", "limit(f,x,0,dir='+-')"),
        Case("factor_integer", "Integer factorization", "600851475143 = 71*839*1471*6857",
             lambda: mf.operations.factor_integer(600851475143), lambda: sp.factorint(600851475143),
             lambda v: dict(v.factors) if v.sign == 1 else None, equal({71: 1, 839: 1, 1471: 1, 6857: 1}),
             "operations.factor_integer(n)", "factorint(n)"),
        Case("binomial", "Binomial coefficient", "C(200,80)",
             lambda: mf.operations.binomial(200, 80), lambda: sp.binomial(200, 80), convert,
             equal(sp.Integer(comb(200, 80))), "operations.binomial(200,80)", "binomial(200,80)"),
        Case("complex_sqrt", "Principal complex square root", "sqrt(3+4*i) = 2+i",
             lambda: mf.operations.complex_sqrt(complex_value), lambda: sp.expand_complex(sp.sqrt(scomplex)),
             lambda v: convert(v.real)+sp.I*convert(v.imag), symbolic(2+sp.I),
             "operations.complex_sqrt(z)", "expand_complex(sqrt(z))"),
        Case("binomial_pmf", "Full binomial PMF", "Binomial(n=40,p=2/7); all 41 probabilities materialized",
             lambda: mf.operations.binomial_distribution(40, p), lambda: density(Binomial("B", 40, sp_p)).dict,
             lambda v: {convert(k): convert(p) for k, p in zip(v.outcomes, v.probabilities)},
             equal(weights), "operations.binomial_distribution(40,2/7)", "density(Binomial('B',40,2/7)).dict"),
        Case("finite_variance", "Finite distribution variance", "Uniform on (-7/3,-2,-1/5,0,2/7,4/3,3,11/2); PMF prebuilt",
             lambda: mf.operations.distribution_variance(distribution), lambda: variance(rv), convert,
             equal(variance_expected), "operations.distribution_variance(D)", "stats.variance(FiniteRV)"),
    ]
    return cases


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT/"src/mathforge").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode()+b"\0"+path.read_bytes())
    return digest.hexdigest()


def statistics(samples):
    values = sorted(samples)
    return dict(samples_ms=samples, median_ms=median(samples), min_ms=values[0],
                max_ms=values[-1], p95_ms=values[ceil(.95*len(values))-1])


def worker(args):
    sys.path.insert(0, str(ROOT/"src"))
    import mathforge as mf
    import sympy as sp
    from sympy.core.cache import clear_cache
    from sympy.ntheory.factor_ import factor_cache

    def reset():
        clear_cache()
        sp.CRootOf.clear_cache()
        factor_cache.cache_clear()

    def valid(case, value, backend, *, recheck=False):
        if backend == "mathforge":
            assert value.execution_status is mf.ExecutionStatus.COMPLETED, value.error
            assert value.outcome is mf.Outcome.VALUE and value.deferred is None
            assert value.exactness is mf.Exactness.EXACT and value.completeness is mf.Completeness.COMPLETE
            assert value.verification and all(c.status is mf.CheckStatus.VERIFIED for c in value.verification)
            if recheck: assert mf.verify(value).verified
            value = case.normalize_mf(value.unwrap())
        assert case.validate(value), (case.name, backend, value)

    cases = fixtures(mf, sp)
    if args.cases:
        requested = set(args.cases.split(","))
        assert requested <= {c.name for c in cases}, requested
        cases = [c for c in cases if c.name in requested]
    rng = Random(20261014+args.round_index)
    rng.shuffle(cases)
    measurements = []
    for case in cases:
        for backend, operation in (("mathforge", case.mathforge), ("sympy", case.sympy)):
            for _ in range(args.warmups):
                reset()
                valid(case, operation(), backend, recheck=True)
        samples = {mode: [] for mode in ("mathforge", "sympy_reset", "sympy_warm")}
        order_log = []
        for _ in range(args.repeat):
            modes = list(samples)
            rng.shuffle(modes)
            order_log.append(modes)
            for mode in modes:
                operation = case.mathforge if mode == "mathforge" else case.sympy
                gc.collect()
                if mode == "sympy_reset": reset()
                if mode == "sympy_warm":
                    # Prime immediately before every warm sample: another mode
                    # may have reset SymPy caches since its previous invocation.
                    primed = operation()
                    del primed
                enabled = gc.isenabled()
                gc.disable()
                try:
                    start = perf_counter_ns()
                    result = operation()
                    elapsed = (perf_counter_ns()-start)/1_000_000
                finally:
                    if enabled: gc.enable()
                samples[mode].append(elapsed)
                valid(case, result, "mathforge" if mode == "mathforge" else "sympy")
                del result
        row = dict(case=case.name, label=case.label, inputs=case.inputs,
                   mathforge_api=case.mf_api, sympy_api=case.sp_api,
                   all_results_exact_and_validated=True, invocation_order=order_log,
                   **{mode: statistics(values) for mode, values in samples.items()})
        measurements.append(row)
        print(f"round {args.round_index+1} {case.name}: MF={row['mathforge']['median_ms']:.3f}, "
              f"SP-reset={row['sympy_reset']['median_ms']:.3f}, SP-warm={row['sympy_warm']['median_ms']:.3f} ms", flush=True)
    return dict(round_index=args.round_index, source_sha256=fingerprint(),
                mathforge_version=mf.__version__, sympy_version=sp.__version__,
                python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cases", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--round-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.repeat, args.warmups) < 1:
        parser.error("rounds, repeat and warmups must be positive")
    if args.worker:
        report = worker(args)
    else:
        runs = []
        with tempfile.TemporaryDirectory(prefix="mathforge-fresh-benchmark-") as temporary:
            for index in range(args.rounds):
                path = Path(temporary)/f"round-{index}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--round-index", str(index), "--repeat", str(args.repeat),
                           "--warmups", str(args.warmups), "--output", str(path)]
                if args.cases: command.extend(("--cases", args.cases))
                subprocess.run(command, check=True, cwd=ROOT)
                runs.append(json.loads(path.read_text(encoding="utf-8")))
        assert len({r["source_sha256"] for r in runs}) == 1
        summary = []
        for first in runs[0]["measurements"]:
            records = [next(row for row in run["measurements"] if row["case"] == first["case"]) for run in runs]
            item = {key: first[key] for key in ("case", "label", "inputs", "mathforge_api", "sympy_api")}
            for mode in ("mathforge", "sympy_reset", "sympy_warm"):
                item[mode] = statistics([value for row in records for value in row[mode]["samples_ms"]])
                item[mode]["round_medians_ms"] = [row[mode]["median_ms"] for row in records]
            item["mathforge_over_sympy_reset"] = item["mathforge"]["median_ms"]/item["sympy_reset"]["median_ms"]
            summary.append(item)
        summary.sort(key=lambda row: row["case"])
        report = dict(created_utc=datetime.now(timezone.utc).isoformat(), rounds=args.rounds,
                      repeat_per_round=args.repeat, warmups_per_case_per_backend=args.warmups,
                      samples_per_case_per_mode=args.rounds*args.repeat,
                      benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      method={
                          "scope": "Prebuilt mathematical inputs; public API call; no parsing, JSON, rendering or validation timing",
                          "comparison": "MathForge automatic evidence production and independent verification included; SymPy value computation only",
                          "cache_reset": "SymPy core cache, CRootOf cache and factor_cache cleared before each reset sample, outside timing; not cold process startup",
                          "warm_cache": "One untimed priming call immediately before each measured warm call",
                          "gc": "Collection before each call; cyclic GC disabled only during timed call; one result retained then validated and discarded",
                          "ordering": "Fresh sequential process per round; deterministic shuffled cases and interleaved backend order for every repetition",
                          "validation": "Every output checked exactly; MathForge complete exact value + verified labels required; independent re-verification during warmup",
                          "aggregation": "Median of all per-call samples; raw samples, p95 and per-round medians retained; wall clock is non-gating",
                      }, summary=summary, runs=runs)
        print("\nAggregate medians (ms):", flush=True)
        for row in summary:
            print(f"{row['case']}: MF={row['mathforge']['median_ms']:.5f}, "
                  f"SP-reset={row['sympy_reset']['median_ms']:.5f}, "
                  f"SP-warm={row['sympy_warm']['median_ms']:.5f}, "
                  f"MF/SP-reset={row['mathforge_over_sympy_reset']:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
