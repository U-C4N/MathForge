"""REL-13: Optional development oracle, never imported by the runtime package.

Install the pinned oracle with ``pip install sympy==1.14.0`` and run this script.
Core tests and package installation have no dependency on this check.
"""

from random import Random

import mathforge as mf


def _rational(sp, value):
    return sp.Rational(value.numerator, value.denominator)


def _poly(sp, value, x):
    return sp.Poly(sum(_rational(sp, c) * x**i for i, c in enumerate(value.coefficients)), x, domain=sp.QQ)


def _scalar(sp, value, x):
    if type(value) is mf.Rational:
        return _rational(sp, value)
    if type(value) is mf.RealAlgebraicRoot:
        polynomial = sum(c*x**i for i, c in enumerate(value.integer_coefficients))
        return sp.Poly(polynomial, x).real_roots()[value.real_index]
    if type(value) is mf.Sqrt:
        return sp.sqrt(_scalar(sp, value.radicand, x))
    if type(value) is mf.Add:
        return sp.Add(*(_scalar(sp, term, x) for term in value.terms))
    if type(value) is mf.Mul:
        return sp.Mul(*(_scalar(sp, term, x) for term in value.factors))
    if type(value) is mf.Pow:
        return _scalar(sp, value.base, x)**value.exponent
    raise AssertionError(("unexpected exact scalar", value))


def _same_algebraic(sp, actual, expected, x):
    # A zero difference has minimal polynomial x: this tests exact identity,
    # including the selected real conjugate, without a float tolerance.
    difference = _scalar(sp, actual, x) - expected
    assert sp.Poly(sp.minpoly(difference, x), x).monic().as_expr() == x, (actual, expected)


def check_polynomials(sp, rng):
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000601")
    sx = sp.Symbol("x")
    for case in range(30):
        def polynomial():
            values = [mf.Rational(rng.randint(-7, 7), rng.randint(1, 5)) for _ in range(rng.randint(1, 5))]
            return mf.Polynomial(tuple(values) + (mf.Rational(rng.randint(1, 5)),), x)
        f, g = polynomial(), polynomial()
        sf, sg = _poly(sp, f, sx), _poly(sp, g, sx)
        quotient, remainder = mf.poly_divmod(f, g)
        sq, sr = sp.div(sf, sg)
        assert _poly(sp, quotient, sx) == sq and _poly(sp, remainder, sx) == sr, case
        assert _poly(sp, mf.poly_gcd(f, g), sx) == sp.gcd(sf, sg).monic(), case
        factors = mf.factor(f)
        coefficient, oracle_factors = sp.factor_list(sf)
        expected = {}
        scalar = coefficient
        for factor, multiplicity in oracle_factors:
            scalar *= factor.LC()**multiplicity
            expected[tuple(factor.monic().all_coeffs())] = multiplicity
        actual = {tuple(_poly(sp, entry.polynomial, sx).all_coeffs()): entry.multiplicity
                  for entry in factors.factors}
        assert _rational(sp, factors.coefficient) == scalar and actual == expected, case
        # SymPy 1.14.0's PRS path swaps lower-degree inputs without the
        # odd-degree sign (e.g. resultant(x-1, x**3+2) == -3). Normalize
        # explicitly so this oracle tests the Sylvester determinant convention.
        expected_resultant = (sp.resultant(sf, sg) if sf.degree() >= sg.degree()
                              else (-1)**(sf.degree()*sg.degree())*sp.resultant(sg, sf))
        assert _rational(sp, mf.resultant(f, g)) == expected_resultant, case
        assert _rational(sp, mf.discriminant(f)) == sp.discriminant(sf), case
        derivative = mf.diff(f.to_expression(), x)
        assert _poly(sp, mf.Polynomial.from_expression(derivative, x), sx) == sf.diff(), case

    for case in range(12):
        a, b = rng.randint(-4, 4), rng.randint(1, 6)
        expression = (x-a)**2 * (x*x-b)
        roots = mf.real_roots(expression, x)
        expected = sp.Poly((sx-a)**2*(sx*sx-b), sx).real_roots(multiple=False)
        assert len(roots) == len(expected), case
        for actual, (expected_root, multiplicity) in zip(roots, expected):
            assert actual.multiplicity == multiplicity, case
            _same_algebraic(sp, actual.root, expected_root, sx)

    for case in range(12):
        pole, a = rng.randint(-3, 3), rng.randint(1, 5)
        function = mf.rational_function(x*x+a, x-pole, variable=x)
        symbolic = (sx*sx+a)/(sx-pole)
        for side, direction in (("left", "-"), ("right", "+")):
            actual = mf.limit(function, x, pole, side=side)
            expected = sp.limit(symbolic, sx, pole, dir=direction)
            assert actual.kind == ("positive_infinity" if expected == sp.oo else "negative_infinity"), case
        point = pole+2
        actual = mf.limit(function, x, point)
        assert actual.kind == "finite" and _scalar(sp, actual.value, sx) == symbolic.subs(sx, point), case
        derivative = mf.diff(function, x)
        quotient = _poly(sp, derivative.numerator, sx).as_expr()/_poly(sp, derivative.denominator, sx).as_expr()
        assert sp.cancel(quotient-sp.diff(symbolic, sx)) == 0, case
    return 54


def check_new_algebra(sp):
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000602")
    sx = sp.Symbol("x")
    sqrt2, sqrt3 = mf.sqrt(2), mf.sqrt(3)
    cases = (
        (lambda: mf.algebraic(sqrt2+sqrt3), sp.sqrt(2)+sp.sqrt(3)),
        (lambda: mf.algebraic_div(1, sqrt2+sqrt3), 1/(sp.sqrt(2)+sp.sqrt(3))),
        (lambda: mf.algebraic_pow(sqrt2+1, -2), (sp.sqrt(2)+1)**-2),
        (lambda: mf.algebraic((sqrt2+sqrt3)*(sqrt2-sqrt3)), -1),
        (lambda: mf.evaluate_algebraic(x*x+x+1, {x: sqrt2}), 3+sp.sqrt(2)),
        (lambda: mf.integrate(x*x, x, 0, sqrt2), 2*sp.sqrt(2)/3),
    )
    for operation, expected in cases:
        _same_algebraic(sp, operation(), expected, sx)
    function = mf.rational_function(x+1, (x-1)**2*(x*x+1), variable=x)
    decomposition = mf.apart(function)
    actual = _poly(sp, decomposition.polynomial, sx).as_expr()
    for term in decomposition.terms:
        actual += _poly(sp, term.numerator, sx).as_expr()/_poly(sp, term.factor, sx).as_expr()**term.power
    expected = sp.apart((sx+1)/((sx-1)**2*(sx*sx+1)), sx)
    assert sp.cancel(actual-expected) == 0
    minimal = mf.minimal_polynomial(sqrt2+sqrt3, x)
    assert _poly(sp, minimal, sx).monic().as_expr() == sp.minpoly(sp.sqrt(2)+sp.sqrt(3), sx)
    for rows in (((0, 2), (1, 0)), ((0, -1), (1, 0)), ((2, 0), (0, 2))):
        actual = mf.real_eigenvalues(mf.Matrix(rows))
        expected = sp.Matrix(rows).charpoly(sx).as_poly().real_roots(multiple=False)
        assert len(actual) == len(expected), rows
        for record, (root, multiplicity) in zip(actual, expected):
            assert record.multiplicity == multiplicity, rows
            _same_algebraic(sp, record.root, root, sx)
    return len(cases)+5


def check_v07(sp, rng):
    """Seeded external checks across every new non-transcendental family."""
    from sympy.functions.combinatorial.numbers import stirling as oracle_stirling
    from sympy.ntheory.modular import solve_congruence
    from sympy.stats import Binomial, E, variance, density
    count = 0
    for _ in range(25):
        a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        assert mf.integer_gcd(a, b) == int(sp.gcd(a, b))
        assert mf.integer_lcm(a, b) == int(sp.ilcm(a, b))
        g, s, t = mf.integer_xgcd(a, b)
        assert g == int(sp.gcd(a, b)) and a*s+b*t == g
        n = rng.randint(1, 100_000)
        actual = mf.factor_integer(-n)
        assert actual.sign == -1 and dict(actual.factors) == sp.factorint(n)
        assert mf.is_prime(n) == sp.isprime(n)
        assert mf.divisors(n) == tuple(map(int, sp.divisors(n)))
        assert mf.totient(n) == int(sp.totient(n))
        congruences = tuple((rng.randint(-10, 10), rng.randint(1, 15)) for _ in range(3))
        actual_crt, expected_crt = mf.crt(congruences), solve_congruence(*congruences)
        assert ((actual_crt.residue, actual_crt.modulus) == (None, None) if expected_crt is None
                else (actual_crt.residue, actual_crt.modulus) == tuple(map(int, expected_crt)))
        count += 1
    for _ in range(15):
        n, k = rng.randint(0, 25), rng.randint(0, 25)
        assert mf.factorial(n) == int(sp.factorial(n))
        assert mf.binomial(n, k) == int(sp.binomial(n, k))
        assert mf.stirling(n, k) == int(oracle_stirling(n, k, kind=2))
        assert mf.fibonacci(n) == int(sp.fibonacci(n))
        assert mf.catalan(n) == int(sp.catalan(n))
        assert mf.permutations_count(n, k) == (int(sp.factorial(n)/sp.factorial(n-k)) if k <= n else 0)
        counts = tuple(rng.randint(0, 6) for _ in range(3))
        assert mf.multinomial(counts) == int(sp.factorial(sum(counts))/sp.prod(sp.factorial(k) for k in counts))
        count += 1
    for case in range(10):
        n, numerator = rng.randint(1, 9), rng.randint(1, 4)
        p = mf.Rational(numerator, 5)
        actual = mf.binomial_distribution(n, p)
        oracle = Binomial("B" + str(case), n, _rational(sp, p))
        expected_mass = density(oracle).dict
        assert {_rational(sp, x): _rational(sp, q) for x, q in zip(actual.outcomes, actual.probabilities)} == expected_mass
        assert _rational(sp, mf.expectation(actual)) == E(oracle)
        assert _rational(sp, mf.distribution_variance(actual)) == variance(oracle)
        values = tuple(mf.Rational(rng.randint(-10, 10), rng.randint(1, 4)) for _ in range(7))
        symbolic = tuple(_rational(sp, v) for v in values)
        center = sum(symbolic)/len(symbolic)
        assert _rational(sp, mf.mean(values)) == center
        assert _rational(sp, mf.median(values)) == sorted(symbolic)[len(symbolic)//2]
        for ddof in (0, 1):
            expected = sum((v-center)**2 for v in symbolic)/(len(symbolic)-ddof)
            assert _rational(sp, mf.variance(values, ddof)) == expected
        paired = tuple(reversed(values))
        expected_covariance = sum((a-center)*(b-center) for a, b in zip(symbolic, reversed(symbolic)))/(len(symbolic)-1)
        assert _rational(sp, mf.covariance(values, paired, 1)) == expected_covariance
        count += 1
    for _ in range(10):
        rows = tuple(tuple(rng.randint(-3, 3) for _ in range(3)) for _ in range(4))
        A, S = mf.Matrix(rows), sp.Matrix(rows)
        rowspace, columnspace = mf.rowspace(A), mf.columnspace(A)
        assert len(rowspace) == len(columnspace) == S.rank()
        if rowspace:
            row_matrix = sp.Matrix([[_rational(sp, v) for v in row] for row in rowspace])
            assert S.col_join(row_matrix).rank() == S.rank()
        assert tuple(tuple(_rational(sp, v) for v in column) for column in columnspace) == tuple(tuple(v) for v in S.columnspace())
        rhs = tuple(rng.randint(-5, 5) for _ in range(4))
        answer = mf.least_squares(A, rhs)
        particular = sp.Matrix([_rational(sp, v) for v in answer.particular])
        residual = sp.Matrix(rhs)-S*particular
        assert S.T*residual == sp.zeros(3, 1)
        assert tuple(residual) == tuple(_rational(sp, v) for v in answer.residual)
        assert len(answer.basis) == 3-S.rank()
        for vector in answer.basis:
            assert S*sp.Matrix([_rational(sp, v) for v in vector]) == sp.zeros(4, 1)
        square = mf.Matrix(tuple(row for row in rows[:3]))
        decomposition = mf.lu(square)
        convert = lambda matrix: sp.Matrix([[_rational(sp, v) for v in row] for row in matrix.rows])
        assert convert(decomposition.P)*sp.Matrix(rows[:3]) == convert(decomposition.L)*convert(decomposition.U)
        count += 1
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000704")
    sx, st = sp.symbols("x t")
    for denominator in (x, (x*x+1)**2, x**5-x+1):
        source = mf.rational_function(1, denominator, variable=x)
        primitive = mf.rational_antiderivative(source, x)
        rational = _poly(sp, primitive.rational_part.numerator, sx).as_expr()/_poly(sp, primitive.rational_part.denominator, sx).as_expr()
        derivative = sp.diff(rational, sx)
        for term in primitive.log_terms:
            q = sum(_rational(sp, c)*st**i for i, c in enumerate(term.residue_polynomial))
            h = sum(_rational(sp, c)*st**j*sx**i for i, row in enumerate(term.logarithm_coefficients) for j, c in enumerate(row))
            # RootSum's generic symmetric expansion grows needlessly for the
            # irreducible quintic. SymPy's independent field arithmetic computes
            # the same exact sum as the multiplication-operator trace instead.
            field = sp.QQ.frac_field(sx)
            modulus = sp.Poly(q, st, domain=field)
            inverse = sp.invert(sp.Poly(h, st, domain=field), modulus)
            summand = (sp.Poly(st*sp.diff(h, sx), st, domain=field)*inverse).rem(modulus)
            trace = sum((sp.Poly(st**i, st, domain=field)*summand).rem(modulus).nth(i)
                        for i in range(modulus.degree()))
            derivative += trace
        expected = _poly(sp, source.numerator, sx).as_expr()/_poly(sp, source.denominator, sx).as_expr()
        assert sp.cancel(derivative-expected) == 0
        count += 1
    for a, b, c in ((1, 0, 2), (1, -2, 1), (-2, 3, 7), (2, 4, 5)):
        actual = mf.quadratic_roots(a, b, c)
        expected = sp.roots(a*sx*sx+b*sx+c, sx)
        converted = {_scalar(sp, item.root.real, sx)+sp.I*_scalar(sp, item.root.imag, sx): item.multiplicity for item in actual}
        assert len(converted) == len(expected)
        for value, multiplicity in converted.items():
            assert any(sp.simplify(value-root) == 0 and multiplicity == k for root, k in expected.items())
        count += 1
    return count


def main() -> None:
    try:
        import sympy as sp
    except ImportError as exc:
        raise SystemExit("Optional oracle is absent; install sympy==1.14.0 to run this check") from exc
    if sp.__version__ != "1.14.0":
        raise SystemExit("The reproducible development oracle requires sympy==1.14.0")
    random = Random(20409)
    x = mf.Symbol("lambda", uid="00000000-0000-4000-8000-000000000603")
    sx = sp.Symbol("lambda")
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
        assert _poly(sp, mf.charpoly(actual, x), sx) == oracle.charpoly(sx).as_poly().set_domain(sp.QQ), case
    polynomial_cases = check_polynomials(sp, random)
    algebraic_cases = check_new_algebra(sp)
    v07_cases = check_v07(sp, random)
    print(f"100 exact matrix/charpoly, {polynomial_cases} polynomial/root/calculus and "
          f"{algebraic_cases} algebraic/partial-fraction plus {v07_cases} v0.7 integer/combinatorial/probability/statistics/linear/LRT/complex groups agree with SymPy 1.14.0.")


if __name__ == "__main__":
    main()
