"""REL-13: Optional development oracle, never imported by the runtime package.

Install the pinned oracle with ``pip install sympy==1.14.0`` and run this script.
Core tests and package installation have no dependency on this check.
"""

from random import Random

import mathforge as mf

# The operations (registry or legacy catalogue names) whose answers the seeded groups below compare with SymPy: the
# v0.8 coverage gate's differential criterion reads this set. Calls that only build inputs (Symbol, Rational,
# Polynomial, Matrix, sqrt, rational_function) compare nothing; mf.diff is the differentiate operation.
COVERED_OPERATIONS = frozenset({
    # main: matrices
    "rank", "det", "rref", "charpoly",
    # check_polynomials
    "poly_divmod", "poly_gcd", "factor", "resultant", "discriminant", "differentiate", "real_roots", "limit",
    # check_new_algebra
    "algebraic", "algebraic_div", "algebraic_pow", "evaluate_algebraic", "integrate", "apart", "minimal_polynomial",
    "real_eigenvalues",
    # check_v07
    "integer_gcd", "integer_lcm", "integer_xgcd", "factor_integer", "is_prime", "divisors", "totient", "crt",
    "factorial", "binomial", "stirling", "fibonacci", "catalan", "permutations_count", "multinomial",
    "binomial_distribution", "expectation", "distribution_variance", "mean", "median", "variance", "covariance",
    "rowspace", "columnspace", "least_squares", "lu", "rational_antiderivative", "quadratic_roots",
    # check_v08
    "continued_fraction",
    # check_v09_modular, check_v09_prime_search, check_v09_diophantine, check_v09_quadratic_irrational,
    # check_v09_arithmetic_functions
    "powmod", "multiplicative_order", "primitive_root", "legendre_symbol", "jacobi_symbol", "sqrt_mod",
    "next_prime", "prev_prime", "linear_diophantine", "quadratic_continued_fraction", "convergents",
    "pell_fundamental", "moebius", "divisor_sigma", "integer_nth_root",
    # check_v09_roots, check_v09_polynomial_tools, check_v09_radicals, check_v09_piecewise,
    # check_v09_interval_algebra, check_v09_digits
    "count_real_roots", "complex_roots", "interpolate", "denest_sqrt", "solve_abs", "solve_sqrt", "set_union",
    "set_intersection", "set_difference", "set_complement", "digits",
    # check_v09_sums, check_v09_power_series, check_v09_trig_powers
    "sum_polynomial", "geometric_sum", "geometric_series", "polynomial_geometric_series", "series_coefficients",
    "trig_power_antiderivative", "trig_power_integral",
    # check_v09_powers, check_v09_eigen, check_v09_products, check_v09_plane, check_v09_bivariate, check_v09_counting
    "matrix_power", "eigenvectors", "dot", "cross", "projection", "triple_product", "distance", "polygon_area",
    "line_intersection", "orientation", "correlation", "linear_regression", "partitions", "derangements", "bell",
    "stirling1",
    # check_v09_sequences, check_v09_approximation, check_v09_symmetric, check_v09_laurent, check_v09_discrete,
    # check_v09_forms, check_v09_lattice, check_v09_hull, check_v09_order, check_v09_markov, check_v09_exact_tests
    "sturm_sequence", "cyclotomic_polynomial", "rational_approximation", "set_symmetric_difference", "residue",
    "discrete_log", "two_squares", "definiteness", "gram_schmidt", "hermite_normal_form", "smith_normal_form",
    "convex_hull", "quantile", "mode", "stationary_distribution", "fisher_exact",
})


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


def check_v08(sp):
    """The v0.8 pilot family on 200 seeded rationals; its own stream leaves the earlier checks' draws unchanged."""
    rng, cases = Random(80801), 200
    for _ in range(cases):
        p, q = rng.randint(-10**12, 10**12), rng.randint(1, 10**12)
        expected = list(sp.continued_fraction(sp.Rational(p, q)))
        assert expected == list(mf.ntheory.continued_fraction(mf.Rational(p, q)).terms), (p, q)
    return cases


def check_v09_modular(sp):
    """The v0.9 modular family on 100 seeded cases; like check_v08, each v0.9 group draws from its own stream."""
    rng, cases = Random(90901), 100
    for _ in range(cases):
        modulus, exponent = rng.randint(2, 10**6), rng.randint(0, 60)
        base = rng.randint(-10**6, 10**6)
        assert mf.ntheory.powmod(base, exponent, modulus) == int(sp.Integer(base)**exponent % modulus)
        unit = rng.randint(-10**6, 10**6)
        while sp.gcd(unit, modulus) != 1:
            unit = rng.randint(-10**6, 10**6)
        inverse = sp.mod_inverse(unit, modulus)
        assert mf.ntheory.powmod(unit, -exponent - 1, modulus) == int(sp.Integer(inverse)**(exponent + 1) % modulus)
        assert mf.ntheory.multiplicative_order(unit, modulus) == int(sp.n_order(unit, modulus)), (unit, modulus)
        p = int(sp.nextprime(rng.randint(2, 10**6)))  # An odd prime.
        a, n = rng.randint(-10**6, 10**6), 2 * rng.randint(0, 10**6) + 1
        assert mf.ntheory.primitive_root(p) == int(sp.primitive_root(p)), p
        assert mf.ntheory.legendre_symbol(a, p) == int(sp.legendre_symbol(a, p)), (a, p)
        assert mf.ntheory.jacobi_symbol(a, n) == int(sp.jacobi_symbol(a, n)), (a, n)
        roots = sp.sqrt_mod(a, p, all_roots=True) or ()  # The sorted list of all roots, empty for none.
        assert mf.ntheory.sqrt_mod(a, p) == tuple(int(root) for root in roots), (a, p)
    return cases


def check_v09_prime_search(sp):
    """next_prime and prev_prime on 100 seeded integers each."""
    rng, cases = Random(90902), 100
    for _ in range(cases):
        n, m = rng.randint(-10, 10**12), rng.randint(3, 10**12)
        assert mf.ntheory.next_prime(n) == int(sp.nextprime(n)), n
        assert mf.ntheory.prev_prime(m) == int(sp.prevprime(m)), m
    return cases


def check_v09_diophantine(sp):
    """linear_diophantine on 100 seeded equations a*x + b*y = c with a, b != 0, often sharing a factor, against
    diop_linear: SymPy's solution (x0 + u*t, y0 + v*t) made canonical (u > 0, then 0 <= x < u), or (None, None)."""
    from sympy.solvers.diophantine.diophantine import diop_linear
    rng, cases = Random(90903), 100
    sx, sy = sp.symbols("x y", integer=True)
    for _ in range(cases):
        common = rng.randint(1, 30)
        a = common * rng.choice((-1, 1)) * rng.randint(1, 10**6)
        b = common * rng.choice((-1, 1)) * rng.randint(1, 10**6)
        c = rng.randint(-10**9, 10**9)
        solution = diop_linear(a*sx + b*sy - c)
        if solution[0] is None:
            expected = (None, None, None, None)
        else:
            t = (solution[0].free_symbols | solution[1].free_symbols).pop()
            (x0, y0), (u, v) = ((int(part.subs(t, 0)) for part in solution),
                                (int(part.coeff(t)) for part in solution))
            if u < 0:
                u, v = -u, -v
            shift = x0 // u
            expected = (x0 - shift * u, y0 - shift * v, u, v)
        actual = mf.ntheory.linear_diophantine(a, b, c)
        assert (actual.x, actual.y, actual.dx, actual.dy) == expected, (a, b, c)
    return cases


def check_v09_quadratic_irrational(sp):
    """60 seeded cases: positive quadratic irrationals against continued_fraction_periodic, the convergents of
    rationals against continued_fraction_convergents and fundamental Pell solutions against diop_DN(d, 1)."""
    from sympy.solvers.diophantine.diophantine import diop_DN
    rng, cases = Random(90904), 60
    for _ in range(cases):
        d = rng.randint(2, 500)
        while sp.integer_nthroot(d, 2)[1]:  # sqrt(d) must be irrational.
            d += 1
        p, q, r = rng.randint(0, 30), rng.randint(1, 3), rng.randint(1, 12)
        # (p + q*sqrt(d))/r = (p + sqrt(d*q*q))/r, which SymPy expands as a list of the preperiod's terms that ends
        # with the list of the period's terms.
        expected = sp.continued_fraction_periodic(p, r, d * q * q)
        actual = mf.ntheory.quadratic_continued_fraction(p, q, d, r)
        assert actual.preperiod == tuple(int(term) for term in expected[:-1]), (p, q, d, r)
        assert actual.period == tuple(int(term) for term in expected[-1]), (p, q, d, r)
        numerator, denominator = rng.randint(-10**12, 10**12), rng.randint(1, 10**12)
        oracle = list(sp.continued_fraction_convergents(sp.continued_fraction(sp.Rational(numerator, denominator))))
        convergents = mf.ntheory.convergents(mf.Rational(numerator, denominator))
        assert [_rational(sp, value) for value in convergents] == oracle, (numerator, denominator)
        assert [mf.ntheory.pell_fundamental(d)] == [(int(x), int(y)) for x, y in diop_DN(d, 1)], d
    return cases


def check_v09_arithmetic_functions(sp):
    """moebius, divisor_sigma and integer_nth_root on 100 seeded cases each."""
    rng, cases = Random(90905), 100
    for _ in range(cases):
        n, k = rng.randint(1, 10**9), rng.randint(0, 5)
        assert mf.ntheory.moebius(n) == int(sp.mobius(n)), n
        assert mf.ntheory.divisor_sigma(n, k) == int(sp.divisor_sigma(n, k)), (n, k)
        radicand, degree = rng.randint(0, 10**40), rng.randint(1, 12)
        assert mf.ntheory.integer_nth_root(radicand, degree) == int(sp.integer_nthroot(radicand, degree)[0])
    return cases



def _real_set(sp, value, x):
    """A v0.9 solution set (EmptySet, UniversalSet or IntervalSet) as a SymPy set; None is an infinite end."""
    if type(value) is mf.EmptySet:
        return sp.S.EmptySet
    if type(value) is mf.UniversalSet:
        return sp.S.Reals
    end = lambda point, infinity: infinity if point is None else _scalar(sp, point, x)
    return sp.Union(*(sp.Interval(end(item.lower, -sp.oo), end(item.upper, sp.oo), not item.left_closed,
                                  not item.right_closed) for item in value.intervals))


def _same_real_set(sp, actual, expected, points=()):
    """Two SymPy sets agree at every end of either, between neighbouring ends and beyond them: finite unions of
    intervals are equal exactly when they agree there."""
    ends = sorted(set(actual.boundary) | set(expected.boundary) | set(points))
    probes = [*ends, *((a + b)/2 for a, b in zip(ends, ends[1:]))]
    probes += [ends[0] - 1, ends[-1] + 1] if ends else [0]
    for point in probes:
        assert bool(actual.contains(point)) == bool(expected.contains(point)), (actual, expected, point)


def check_v09_roots(sp):
    """count_real_roots against Poly.count_roots and complex_roots against roots() on 40 seeded polynomials whose
    irreducible factors have degree at most two."""
    rng, cases = Random(90906), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000906")
    sx = sp.Symbol("x")
    for _ in range(cases):
        factors = [sx - sp.Rational(rng.randint(-6, 6), rng.randint(1, 3)) for _ in range(rng.randint(0, 3))]
        factors += [sx**2 + rng.randint(-4, 4)*sx + rng.randint(-4, 6) for _ in range(rng.randint(0, 2))]
        expression = sp.expand(rng.randint(1, 3) * sp.Mul(*factors)**rng.randint(1, 2))
        oracle = sp.Poly(expression, sx, domain=sp.QQ)
        polynomial = mf.Polynomial(tuple(mf.Rational(int(c.p), int(c.q)) for c in reversed(oracle.all_coeffs())), x)
        assert mf.algebra.count_real_roots(polynomial) == oracle.count_roots(), expression
        lower, upper = sorted((rng.randint(-6, 6), rng.randint(-6, 6)))
        assert mf.algebra.count_real_roots(polynomial, lower, upper) == oracle.count_roots(lower, upper), expression
        expected = sp.roots(expression, sx)
        actual = {_scalar(sp, item.root.real, sx) + sp.I*_scalar(sp, item.root.imag, sx): item.multiplicity
                  for item in mf.algebra.complex_roots(polynomial)}
        assert sorted(actual.values()) == sorted(expected.values()), expression
        for value, multiplicity in actual.items():
            assert any(sp.simplify(value - root) == 0 and multiplicity == k for root, k in expected.items())
    return cases


def check_v09_polynomial_tools(sp):
    """interpolate against interpolate() on 60 seeded sets of points with distinct rational abscissas."""
    rng, cases = Random(90907), 60
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000907")
    sx = sp.Symbol("x")
    for _ in range(cases):
        xs, size = [], rng.randint(1, 6)
        while len(xs) < size:
            point = mf.Rational(rng.randint(-20, 20), rng.randint(1, 4))
            if point not in xs:
                xs.append(point)
        ys = [mf.Rational(rng.randint(-20, 20), rng.randint(1, 4)) for _ in xs]
        expected = sp.interpolate([(_rational(sp, a), _rational(sp, b)) for a, b in zip(xs, ys)], sx)
        assert _poly(sp, mf.algebra.interpolate(tuple(xs), tuple(ys), x), sx) == sp.Poly(expected, sx, domain=sp.QQ)
    return cases


def check_v09_radicals(sp):
    """denest_sqrt on 60 seeded radicands (u + v*sqrt(d))**2, each value equal to sqrtdenest's (a zero difference
    has minimal polynomial x)."""
    rng, cases = Random(90908), 60
    sx = sp.Symbol("x")
    for _ in range(cases):
        d = rng.choice((2, 3, 5, 6, 7, 10, 11, 13))
        u, v = (rng.choice((-1, 1)) * sp.Rational(rng.randint(1, 9), rng.randint(1, 3)) for _ in range(2))
        a, b = u**2 + d*v**2, 2*u*v
        actual = mf.algebra.denest_sqrt(mf.Rational(int(a.p), int(a.q)), mf.Rational(int(b.p), int(b.q)), d)
        value = sp.sqrt(_rational(sp, actual.x)) + actual.sign*sp.sqrt(_rational(sp, actual.y))
        assert sp.minpoly(value - sp.sqrtdenest(sp.sqrt(a + b*sp.sqrt(d))), sx) == sx, (a, b, d)
    return cases


def check_v09_piecewise(sp):
    """solve_abs and solve_sqrt against solveset over the reals on 40 seeded problems each."""
    from mathforge import functions as fn
    rng, cases = Random(90909), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000909")
    sx = sp.Symbol("x", real=True)
    relations = {"=": sp.Eq, "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}
    variable = fn.as_real_expr(x)
    for _ in range(cases):
        a, b = (mf.Rational(rng.randint(-6, 6), rng.randint(1, 2)) for _ in range(2))
        c, relation = mf.Rational(rng.randint(-2, 12), rng.randint(1, 2)), rng.choice(tuple(relations))
        twice = rng.random() < 0.5
        lhs = fn.abs(variable - a) + fn.abs(variable - b) if twice else fn.abs(variable - a)
        actual = mf.equations.solve_abs(fn.real_function(lhs, (x,)), fn.real_function(fn.as_real_expr(c), (x,)), x,
                                        relation)
        sa, sb, sc = (_rational(sp, value) for value in (a, b, c))
        problem = relations[relation](sp.Abs(sx - sa) + (sp.Abs(sx - sb) if twice else 0), sc)
        _same_real_set(sp, _real_set(sp, actual, sx), sp.solveset(problem, sx, sp.S.Reals), (sa, sb))
        actual = mf.equations.solve_sqrt(fn.real_function(fn.sqrt(variable - a), (x,)),
                                         fn.real_function(fn.as_real_expr(c - 4), (x,)), x, relation)
        problem = relations[relation](sp.sqrt(sx - sa), sc - 4)
        _same_real_set(sp, _real_set(sp, actual, sx), sp.solveset(problem, sx, sp.S.Reals), (sa,))
    return cases


def check_v09_interval_algebra(sp):
    """set_union, set_intersection, set_difference and set_complement against Union, Intersection and Complement
    on 60 seeded pairs of finite unions of intervals with rational ends."""
    from mathforge.intervals import Interval, IntervalSet
    rng, cases = Random(90910), 60
    sx = sp.Symbol("x")

    def interval_set():
        ends = sorted({mf.Rational(rng.randint(-12, 12), rng.randint(1, 2)) for _ in range(2 * rng.randint(0, 3))})
        ends = ends[:len(ends) // 2 * 2]
        if ends and rng.random() < 0.2:
            ends[0] = None
        if ends and rng.random() < 0.2:
            ends[-1] = None
        return IntervalSet(tuple(Interval(lower, upper, lower is not None and rng.random() < 0.5,
                                          upper is not None and rng.random() < 0.5)
                                 for lower, upper in zip(ends[::2], ends[1::2])))

    for _ in range(cases):
        a, b = interval_set(), interval_set()
        sa, sb = _real_set(sp, a, sx), _real_set(sp, b, sx)
        _same_real_set(sp, _real_set(sp, mf.sets.set_union(a, b), sx), sp.Union(sa, sb))
        _same_real_set(sp, _real_set(sp, mf.sets.set_intersection(a, b), sx), sp.Intersection(sa, sb))
        _same_real_set(sp, _real_set(sp, mf.sets.set_difference(a, b), sx), sp.Complement(sa, sb))
        _same_real_set(sp, _real_set(sp, mf.sets.set_complement(a), sx), sp.Complement(sp.S.Reals, sa))
    return cases


def check_v09_digits(sp):
    """digits of 60 seeded rationals, square roots of rationals and real cube roots against floor()."""
    from mathforge.algebraic import RealAlgebraicRoot
    rng, cases = Random(90911), 60
    sx = sp.Symbol("x")
    for case in range(cases):
        q = mf.Rational(rng.randint(-10**6, 10**6), rng.randint(1, 10**4))
        value = (q, mf.sqrt(mf.Rational(rng.randint(1, 10**4), rng.randint(1, 50))),
                 RealAlgebraicRoot((-rng.choice((2, 3, 5, 7, 9, 10, 12, 20)), 0, 0, 1), 0))[case % 3]
        if case % 3 == 1 and rng.random() < 0.5:
            value = -value
        places = rng.randint(0, 40)
        exact = _scalar(sp, value, sx)
        scaled = sp.Abs(exact) * 10**places
        actual = mf.numeric.digits(value, places)
        assert (actual.sign, actual.magnitude, actual.places, actual.exact) == (
            int(sp.sign(exact)), int(sp.floor(scaled)), places, bool(scaled.is_integer)), (value, places)
    return cases


def check_v09_sums(sp):
    """sum_polynomial, geometric_sum, geometric_series and polynomial_geometric_series against summation on 40
    seeded cases each."""
    rng, cases = Random(90912), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000912")
    sx, sk = sp.symbols("x k")
    for _ in range(cases):
        coefficients = [mf.Rational(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(rng.randint(0, 3))]
        polynomial = mf.Polynomial((*coefficients, mf.Rational(rng.choice((-1, 1)) * rng.randint(1, 9))), x)
        term = _poly(sp, polynomial, sx).as_expr().subs(sx, sk)
        expected = sp.Poly(sp.summation(term, (sk, 1, sx)), sx, domain=sp.QQ)
        assert _poly(sp, mf.series.sum_polynomial(polynomial), sx) == expected, polynomial
        first = mf.Rational(rng.randint(-20, 20), rng.randint(1, 5))
        ratio = mf.Rational(rng.choice((-1, 1)) * rng.randint(1, 30), rng.randint(1, 10))
        count = rng.randint(0, 30)
        sf, sr = _rational(sp, first), _rational(sp, ratio)
        expected = sp.summation(sf * sr**sk, (sk, 0, count - 1))
        assert _rational(sp, mf.series.geometric_sum(first, ratio, count)) == expected, (first, ratio, count)
        small = mf.Rational(rng.choice((-1, 1)) * rng.randint(1, 9), 10)
        ss = _rational(sp, small)
        assert _rational(sp, mf.series.geometric_series(first, small)) == sp.summation(sf * ss**sk, (sk, 0, sp.oo))
        expected = sp.simplify(sp.expand_func(sp.summation(term * ss**sk, (sk, 0, sp.oo))))
        assert _rational(sp, mf.series.polynomial_geometric_series(polynomial, small)) == expected, (polynomial, small)
    return cases


def check_v09_power_series(sp):
    """series_coefficients of 40 seeded functions exp(a*x)*cos(b*x) + sin(c*x) at 0 against series()."""
    from mathforge import functions as fn
    rng, cases = Random(90913), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000913")
    sx = sp.Symbol("x")
    variable = fn.as_real_expr(x)
    for _ in range(cases):
        a, b, c = (mf.Rational(rng.randint(-5, 5), rng.randint(1, 3)) for _ in range(3))
        order = rng.randint(0, 10)
        function = fn.real_function(fn.exp(variable * a) * fn.cos(variable * b) + fn.sin(variable * c), (x,))
        sa, sb, sc = (_rational(sp, value) for value in (a, b, c))
        expansion = sp.series(sp.exp(sa*sx) * sp.cos(sb*sx) + sp.sin(sc*sx), sx, 0, order + 1).removeO()
        actual = mf.series.series_coefficients(function, x, 0, order).coefficients
        assert [_rational(sp, value) for value in actual] == [expansion.coeff(sx, k) for k in range(order + 1)]
    return cases


def check_v09_trig_powers(sp):
    """trig_power_antiderivative (differentiated back) and trig_power_integral against integrate() on 40 seeded
    products sin(a*x)**m * cos(a*x)**n between multiples of pi/12."""
    from mathforge import functions as fn
    from mathforge import render_text
    rng, cases = Random(90914), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000914")
    sx = sp.Symbol("x")
    for _ in range(cases):
        a, m, n = rng.randint(1, 3), rng.randint(0, 4), rng.randint(0, 4)
        if m == n == 0:
            m = 2
        u = fn.as_real_expr(x) * a
        factors = [*((fn.sin(u)**m,) if m else ()), *((fn.cos(u)**n,) if n else ())]
        function = fn.real_function(factors[0] * factors[1] if len(factors) == 2 else factors[0], (x,))
        integrand = sp.sin(a*sx)**m * sp.cos(a*sx)**n
        antiderivative = sp.sympify(render_text(mf.integrals.trig_power_antiderivative(function, x)), {"x": sx})
        assert sp.expand((sp.diff(antiderivative, sx) - integrand).rewrite(sp.exp)) == 0, (a, m, n)
        lower, upper = sp.Rational(rng.randint(-12, 12), 12), sp.Rational(rng.randint(-12, 24), 12)
        actual = mf.integrals.trig_power_integral(function, x, mf.Rational(int(lower.p), int(lower.q)),
                                                  mf.Rational(int(upper.p), int(upper.q)))
        expected = sp.expand(sp.integrate(integrand, (sx, lower*sp.pi, upper*sp.pi)))
        assert _rational(sp, actual.pi_coefficient) == expected.coeff(sp.pi), (a, m, n, lower, upper)
        rest = expected - expected.coeff(sp.pi)*sp.pi
        assert sp.minpoly(_scalar(sp, actual.algebraic, sx) - rest, sx) == sx, (a, m, n, lower, upper)
    return cases


def check_v09_powers(sp):
    """matrix_power against Matrix**k on 60 seeded square rational matrices."""
    rng, cases = Random(90915), 60
    for _ in range(cases):
        size, k = rng.randint(1, 4), rng.randint(0, 12)
        rows = tuple(tuple(mf.Rational(rng.randint(-4, 4), rng.randint(1, 3)) for _ in range(size))
                     for _ in range(size))
        expected = sp.Matrix([[_rational(sp, value) for value in row] for row in rows])**k
        actual = mf.linalg.matrix_power(mf.Matrix(rows), k)
        assert [[_rational(sp, value) for value in row] for row in actual.rows] == expected.tolist(), (rows, k)
    return cases


def check_v09_eigen(sp):
    """eigenvectors against eigenvects() on 40 seeded triangular integer matrices conjugated by a unimodular one,
    so that every eigenvalue is rational: the same values, algebraic multiplicities and eigenspaces."""
    rng, cases = Random(90916), 40
    for _ in range(cases):
        size = rng.randint(1, 4)
        triangular = sp.Matrix(size, size, lambda i, j: rng.randint(-2, 2) if i == j else
                               (rng.randint(-3, 3) if i < j else 0))
        change = sp.Matrix(size, size, lambda i, j: 1 if i == j else (rng.randint(-2, 2) if i > j else 0))
        oracle = change * triangular * change.inv()
        rows = tuple(tuple(int(value) for value in oracle.row(i)) for i in range(size))
        actual = {_rational(sp, space.value): space for space in mf.linalg.eigenvectors(mf.Matrix(rows)).eigenspaces}
        expected = oracle.eigenvects()
        assert sorted(actual) == sorted(value for value, _, _ in expected), rows
        for value, multiplicity, vectors in expected:
            space = actual[value]
            basis = sp.Matrix([[_rational(sp, entry) for entry in vector] for vector in space.basis])
            assert space.multiplicity == multiplicity and basis.rows == len(vectors), rows
            assert basis.col_join(sp.Matrix.hstack(*vectors).T).rank() == len(vectors), rows
    return cases


def check_v09_products(sp):
    """dot, cross, projection and triple_product against Matrix.dot, cross, project and det on 60 seeded cases."""
    rng, cases = Random(90917), 60
    for _ in range(cases):
        size = rng.randint(1, 6)
        a, b = (tuple(mf.Rational(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(size)) for _ in range(2))
        sa, sb = (sp.Matrix([_rational(sp, value) for value in vector]) for vector in (a, b))
        assert _rational(sp, mf.vector.dot(a, b)) == sa.dot(sb), (a, b)
        if any(value.numerator for value in b):
            assert [_rational(sp, value) for value in mf.vector.projection(a, b)] == list(sa.project(sb)), (a, b)
        u, v, w = (tuple(mf.Rational(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(3)) for _ in range(3))
        su, sv, sw = (sp.Matrix([_rational(sp, value) for value in vector]) for vector in (u, v, w))
        assert [_rational(sp, value) for value in mf.vector.cross(u, v)] == list(su.cross(sv)), (u, v)
        assert _rational(sp, mf.vector.triple_product(u, v, w)) == sp.Matrix.hstack(su, sv, sw).det(), (u, v, w)
    return cases


def check_v09_plane(sp):
    """distance against Point.distance, polygon_area against Polygon.area (signed), line_intersection against
    Line.intersection and orientation against the sign of a determinant on 60 seeded cases."""
    rng, cases = Random(90918), 60
    sx = sp.Symbol("x")

    def point(size=2):
        return tuple(mf.Rational(rng.randint(-9, 9), rng.randint(1, 3)) for _ in range(size))

    def oracle(p):
        return sp.Point(*(_rational(sp, value) for value in p))

    for _ in range(cases):
        size = rng.randint(2, 4)
        p, q = point(size), point(size)
        _same_algebraic(sp, mf.geometry.distance(p, q), oracle(p).distance(oracle(q)), sx)
        corners = (point(), point(), point())
        if not sp.Point.is_collinear(*map(oracle, corners)):
            area = sp.Polygon(*map(oracle, corners)).area
            assert _rational(sp, mf.geometry.polygon_area(mf.Matrix(corners))) == area, corners
        p1, p2, q1 = point(), point(), point()
        while p2 == p1:
            p2 = point()
        shape = rng.random()
        if shape < 0.5:
            q2 = point()
            while q2 == q1:
                q2 = point()
        else:  # A parallel line, or with q1 on the first line the same line.
            if shape < 0.75:
                q1 = tuple(s + mf.Rational(rng.randint(-3, 3)) * (t - s) for s, t in zip(p1, p2))
            step = mf.Rational(rng.choice((-2, -1, 1, 2)))
            q2 = tuple(s + step * (t - r) for s, r, t in zip(q1, p1, p2))
        found = sp.Line(oracle(p1), oracle(p2)).intersection(sp.Line(oracle(q1), oracle(q2)))
        actual = mf.geometry.line_intersection(p1, p2, q1, q2)
        if not found:
            assert actual.kind == "parallel" and actual.point is None, (p1, p2, q1, q2)
        elif isinstance(found[0], sp.Line):
            assert actual.kind == "coincident" and actual.point is None, (p1, p2, q1, q2)
        else:
            assert actual.kind == "point" and oracle(actual.point) == found[0], (p1, p2, q1, q2)
        r = point() if rng.random() < 0.7 else tuple(s + mf.Rational(rng.randint(-2, 2)) * (t - s)
                                                    for s, t in zip(p1, p2))
        determinant = sp.Matrix([[_rational(sp, t - s) for s, t in zip(p1, p2)],
                                 [_rational(sp, t - s) for s, t in zip(p1, r)]]).det()
        assert mf.geometry.orientation(p1, p2, r) == int(sp.sign(determinant)), (p1, p2, r)
    return cases


def check_v09_bivariate(sp):
    """correlation and linear_regression against the centred-sum formulas on 60 seeded data sets."""
    rng, cases = Random(90919), 60
    sx = sp.Symbol("x")
    for _ in range(cases):
        size, xs, ys = rng.randint(2, 8), (), ()
        while len(set(xs)) < 2 or len(set(ys)) < 2:  # Neither variable may have zero variance.
            xs, ys = (tuple(mf.Rational(rng.randint(-20, 20), rng.randint(1, 3)) for _ in range(size))
                      for _ in range(2))
        sxs, sys_ = [_rational(sp, value) for value in xs], [_rational(sp, value) for value in ys]
        mx, my = sum(sxs) / size, sum(sys_) / size
        sxx, syy = sum((a - mx)**2 for a in sxs), sum((b - my)**2 for b in sys_)
        sxy = sum((a - mx) * (b - my) for a, b in zip(sxs, sys_))
        _same_algebraic(sp, mf.stats.correlation(xs, ys), sxy / sp.sqrt(sxx * syy), sx)
        fit = mf.stats.linear_regression(xs, ys)
        assert (_rational(sp, fit.slope), _rational(sp, fit.intercept)) == (sxy / sxx, my - sxy / sxx * mx), (xs, ys)
    return cases


def check_v09_counting(sp):
    """partitions, derangements, bell and stirling1 (unsigned) against partition, subfactorial, bell and
    stirling(kind=1) for n = 0 .. 60."""
    from sympy.functions.combinatorial.numbers import stirling as oracle_stirling
    rng, cases = Random(90920), 61
    for n in range(cases):
        assert mf.discrete.partitions(n) == int(sp.partition(n)), n
        assert mf.discrete.derangements(n) == int(sp.subfactorial(n)), n
        assert mf.discrete.bell(n) == int(sp.bell(n)), n
        k = rng.randint(0, n + 1)
        assert mf.discrete.stirling1(n, k) == int(oracle_stirling(n, k, kind=1)), (n, k)
    return cases


def _random_polynomial(rng, x, degree):
    """A polynomial of exactly that degree with small rational coefficients."""
    coefficients = [mf.Rational(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(degree)]
    return mf.Polynomial((*coefficients, mf.Rational(rng.choice((-1, 1)) * rng.randint(1, 9), rng.randint(1, 3))), x)


def check_v09_sequences(sp):
    """sturm_sequence against Poly.sturm on 40 seeded square-free polynomials (both sequences divided by the leading
    coefficient of their first member, since SymPy's starts from the monic square-free part), and
    cyclotomic_polynomial against cyclotomic_poly for n = 1 .. 120."""
    rng, cases = Random(90921), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000921")
    sx = sp.Symbol("x")
    done = 0
    while done < cases:
        polynomial = _random_polynomial(rng, x, rng.randint(0, 6))
        oracle = _poly(sp, polynomial, sx)
        if oracle.degree() > 0 and sp.gcd(oracle, oracle.diff(sx)).degree() > 0:
            continue  # SymPy's Sturm sequence starts from the square-free part.
        done += 1
        actual = [_poly(sp, member, sx) for member in mf.algebra.sturm_sequence(polynomial)]
        expected = oracle.sturm()
        assert [member * (1 / actual[0].LC()) for member in actual] == [
            member * (1 / expected[0].LC()) for member in expected], polynomial
    for n in range(1, 121):
        expected = sp.Poly(sp.cyclotomic_poly(n, sx), sx, domain=sp.QQ)
        assert _poly(sp, mf.algebra.cyclotomic_polynomial(n, x), sx) == expected, n
    return cases + 120


def check_v09_approximation(sp):
    """rational_approximation of 60 seeded rationals (against the best fraction found by trying every denominator,
    with the smaller denominator on a tie) and square roots and real cube roots (against Fraction.limit_denominator
    of a 200-bit approximation, far closer than any two candidates' distances differ)."""
    from fractions import Fraction
    from mathforge.algebraic import RealAlgebraicRoot
    rng, cases = Random(90922), 60
    sx = sp.Symbol("x")
    for case in range(cases):
        if case % 3 == 0:
            value = mf.Rational(rng.randint(-10**4, 10**4), rng.randint(1, 500))
        elif case % 3 == 1:
            value = mf.sqrt(mf.Rational(rng.randint(1, 10**4), rng.randint(1, 50)))
            value = -value if rng.random() < 0.5 else value
        else:
            value = RealAlgebraicRoot((-rng.choice((2, 3, 5, 7, 9, 10, 12, 20)), 0, 0, 1), 0)
        if type(value) is mf.Rational:  # Perfect squares among the square roots as well.
            bound = rng.randint(1, 40)
            exact = Fraction(value.numerator, value.denominator)
            candidates = {Fraction(p, q) for q in range(1, bound + 1)
                          for p in (exact.numerator * q // exact.denominator,
                                    exact.numerator * q // exact.denominator + 1)}
            expected = min(candidates, key=lambda c: (abs(exact - c), c.denominator, c))
        else:
            bound = rng.randint(1, 10**4)
            close = sp.Rational(sp.N(_scalar(sp, value, sx), 80))
            expected = Fraction(int(close.p), int(close.q)).limit_denominator(bound)
        actual = mf.numeric.rational_approximation(value, bound)
        assert Fraction(actual.numerator, actual.denominator) == expected, (value, bound)
    return cases


def check_v09_symmetric(sp):
    """set_symmetric_difference against the union of the two Complements on 60 seeded pairs of finite unions of
    intervals with rational ends."""
    from mathforge.intervals import Interval, IntervalSet
    rng, cases = Random(90923), 60
    sx = sp.Symbol("x")

    def interval_set():
        ends = sorted({mf.Rational(rng.randint(-12, 12), rng.randint(1, 2)) for _ in range(2 * rng.randint(0, 3))})
        ends = ends[:len(ends) // 2 * 2]
        if ends and rng.random() < 0.2:
            ends[0] = None
        if ends and rng.random() < 0.2:
            ends[-1] = None
        return IntervalSet(tuple(Interval(lower, upper, lower is not None and rng.random() < 0.5,
                                          upper is not None and rng.random() < 0.5)
                                 for lower, upper in zip(ends[::2], ends[1::2])))

    for _ in range(cases):
        a, b = interval_set(), interval_set()
        sa, sb = _real_set(sp, a, sx), _real_set(sp, b, sx)
        expected = sp.Union(sp.Complement(sa, sb), sp.Complement(sb, sa))
        _same_real_set(sp, _real_set(sp, mf.sets.set_symmetric_difference(a, b), sx), expected)
    return cases


def check_v09_laurent(sp):
    """residue against residue() on 40 seeded rational functions, at a pole (of order 1 to 3) or at another point."""
    rng, cases = Random(90924), 40
    x = mf.Symbol("x", uid="00000000-0000-4000-8000-000000000924")
    sx = sp.Symbol("x")
    for _ in range(cases):
        point = mf.Rational(rng.randint(-6, 6), rng.randint(1, 3))
        numerator = _random_polynomial(rng, x, rng.randint(0, 3))
        others = _random_polynomial(rng, x, rng.randint(0, 2))
        oracle = (sx - _rational(sp, point))**rng.randint(0, 3) * _poly(sp, others, sx).as_expr()
        denominator = sp.Poly(oracle, sx, domain=sp.QQ)
        if rng.random() < 0.25:
            point = mf.Rational(rng.randint(-6, 6), rng.randint(1, 3))
        if denominator.eval(_rational(sp, point)) == 0 and _poly(sp, numerator, sx).eval(_rational(sp, point)) == 0:
            numerator = mf.Polynomial((numerator.coefficients[0] + 1, *numerator.coefficients[1:]), x)
        coefficients = tuple(mf.Rational(int(c.p), int(c.q)) for c in reversed(denominator.all_coeffs()))
        function = mf.rational_function(numerator, mf.Polynomial(coefficients, x), variable=x)
        expected = sp.residue(_poly(sp, numerator, sx).as_expr() / oracle, sx, _rational(sp, point))
        assert _rational(sp, mf.series.residue(function, x, point)) == expected, (numerator, oracle, point)
    return cases


def check_v09_discrete(sp):
    """discrete_log against discrete_log() and n_order() modulo 60 seeded primes below 10**6 (None where SymPy finds
    no logarithm), and two_squares on n = 0 .. 400 and 60 seeded larger n against r2 from factorint (a
    representation x**2 + y**2 = n with 0 <= x <= y)."""
    from sympy.ntheory import discrete_log as oracle_log
    rng, cases = Random(90925), 60
    for _ in range(cases):
        p = int(sp.randprime(3, 10**rng.randint(2, 6)))
        base, target = rng.randint(1, p - 1), rng.randint(1, p - 1)
        actual = mf.ntheory.discrete_log(base, target, p)
        assert actual.order == int(sp.n_order(base, p)), (base, target, p)
        try:
            expected = int(oracle_log(p, target, base))
        except ValueError:
            expected = None
        assert actual.exponent == expected, (base, target, p)

    def r2(n):
        if n == 0:
            return 1
        count = 4
        for prime, exponent in sp.factorint(n).items():
            if prime % 4 == 3 and exponent % 2:
                return 0
            if prime % 4 == 1:
                count *= exponent + 1
        return count

    numbers = [*range(401), *(rng.randint(401, 10**12) for _ in range(cases))]
    for n in numbers:
        actual = mf.ntheory.two_squares(n)
        assert actual.count == r2(n), n
        if actual.count:
            assert 0 <= actual.a <= actual.b and actual.a**2 + actual.b**2 == n, n
        else:
            assert actual.a is None and actual.b is None, n
    return cases + len(numbers)


def check_v09_forms(sp):
    """definiteness against the signs of the (exact, all real) roots of the characteristic polynomial on 40 seeded
    symmetric rational matrices, some singular, and gram_schmidt against GramSchmidt on 40 seeded matrices with
    linearly independent rows."""
    rng, cases = Random(90926), 40
    sl = sp.Symbol("lambda")
    kinds = ("positive_definite", "negative_definite", "positive_semidefinite", "negative_semidefinite")
    for case in range(cases):
        size = rng.randint(1, 4)
        if case % 2:  # A Gram matrix B^T D B, singular when B has fewer rows than columns.
            factor = sp.Matrix(rng.randint(1, size), size, lambda i, j: rng.randint(-3, 3))
            middle = sp.diag(*(rng.choice((-1, 1, 2)) * sp.Rational(1, rng.randint(1, 3))
                               for _ in range(factor.rows)))
            oracle = factor.T * middle * factor
        else:
            oracle = sp.zeros(size, size)
            for i in range(size):
                for j in range(i, size):
                    oracle[i, j] = oracle[j, i] = sp.Rational(rng.randint(-6, 6), rng.randint(1, 3))
        rows = tuple(tuple(mf.Rational(int(oracle[i, j].p), int(oracle[i, j].q)) for j in range(size))
                     for i in range(size))
        roots = sp.Poly(oracle.charpoly(sl).as_expr(), sl).real_roots()
        assert len(roots) == size, rows
        positive = sum(1 for root in roots if root.is_positive)
        negative = sum(1 for root in roots if root.is_negative)
        actual = mf.linalg.definiteness(mf.Matrix(rows))
        assert (actual.positive, actual.negative, actual.zero) == (positive, negative, size - positive - negative)
        expected = (kinds[0] if positive == size else kinds[1] if negative == size else kinds[2] if negative == 0
                    else kinds[3] if positive == 0 else "indefinite")
        assert actual.kind == expected, rows
    independent = 0
    for _ in range(cases):
        width = rng.randint(1, 4)
        count = rng.randint(1, width)
        oracle = sp.Matrix(count, width, lambda i, j: sp.Rational(rng.randint(-6, 6), rng.randint(1, 3)))
        if oracle.rank() < count:
            continue
        independent += 1
        rows = tuple(tuple(mf.Rational(int(value.p), int(value.q)) for value in oracle.row(i)) for i in range(count))
        expected = sp.GramSchmidt([oracle.row(i).T for i in range(count)], False)
        actual = mf.linalg.gram_schmidt(mf.Matrix(rows))
        assert [[_rational(sp, value) for value in row] for row in actual.rows] == [list(v) for v in expected], rows
    return cases + independent


def check_v09_lattice(sp):
    """hermite_normal_form and smith_normal_form on 60 seeded integer matrices up to 4 x 4, some rank-deficient.

    The row-style Hermite form of M is the column-style form (Cohen's, which normalforms.hermite_normal_form computes,
    dropping zero columns) of M with its columns reversed and transposed, transposed back and reversed in both
    directions, then padded with zero rows. The invariant factors equal SymPy's smith_normal_form diagonal (in
    absolute value) and the quotients of the determinantal divisors (the gcds of the k x k minors)."""
    from itertools import combinations
    from sympy.matrices.normalforms import hermite_normal_form as oracle_hnf, smith_normal_form as oracle_snf
    rng, cases = Random(90927), 60
    for _ in range(cases):
        rows_count, cols_count = rng.randint(1, 4), rng.randint(1, 4)
        oracle = sp.Matrix(rows_count, cols_count, lambda i, j: rng.randint(-9, 9))
        if rng.random() < 0.25 and rows_count > 1:  # A repeated row multiple: rank-deficient.
            oracle[rows_count - 1, :] = rng.randint(-2, 2) * oracle[0, :]
        rows = tuple(tuple(int(value) for value in oracle.row(i)) for i in range(rows_count))
        actual = mf.linalg.hermite_normal_form(mf.Matrix(rows))
        if oracle.rank():
            column_form = oracle_hnf(oracle.extract(list(range(rows_count)), list(range(cols_count))[::-1]).T)
            expected = [[int(value) for value in reversed(column)] for column in reversed(column_form.T.tolist())]
        else:
            expected = []
        expected += [[0] * cols_count for _ in range(rows_count - len(expected))]
        assert [[_rational(sp, value) for value in row] for row in actual.rows] == expected, rows
        actual = mf.linalg.smith_normal_form(mf.Matrix(rows))
        size = min(rows_count, cols_count)
        diagonal = oracle_snf(oracle, domain=sp.ZZ)
        found = sorted(abs(int(diagonal[i, i])) for i in range(size))
        assert sorted(actual) == found, rows
        divisors = [1]
        for k in range(1, size + 1):
            divisors.append(abs(int(sp.gcd([oracle.extract(list(r), list(c)).det()
                                        for r in combinations(range(rows_count), k)
                                        for c in combinations(range(cols_count), k)]))))
        expected = [divisors[k] // divisors[k - 1] if divisors[k] else 0 for k in range(1, size + 1)]
        assert list(actual) == expected, rows
    return cases


def check_v09_hull(sp):
    """convex_hull against convex_hull() on 60 seeded sets of up to 12 rational points (some collinear or repeated):
    the same vertices, listed counterclockwise from the lowest (then leftmost) one."""
    rng, cases = Random(90928), 60
    for case in range(cases):
        count = rng.randint(1, 12)
        if case % 5 == 0:  # On one line.
            start, step = (rng.randint(-5, 5), rng.randint(-5, 5)), (rng.randint(-2, 2), rng.randint(-2, 2))
            points = [(mf.Rational(start[0] + t * step[0]), mf.Rational(start[1] + t * step[1]))
                      for t in (rng.randint(-4, 4) for _ in range(count))]
        else:
            points = [(mf.Rational(rng.randint(-6, 6), rng.randint(1, 2)), mf.Rational(rng.randint(-6, 6),
                                                                                      rng.randint(1, 2)))
                      for _ in range(count)]
        hull = sp.convex_hull(*(sp.Point(_rational(sp, a), _rational(sp, b)) for a, b in points))
        expected = ({hull} if isinstance(hull, sp.Point) else set(hull.points) if isinstance(hull, sp.Segment)
                    else set(hull.vertices))
        vertices = mf.geometry.convex_hull(mf.Matrix(points)).rows
        actual = [sp.Point(_rational(sp, a), _rational(sp, b)) for a, b in vertices]
        assert len(actual) == len(expected) and set(actual) == expected, points
        assert actual[0] == min(actual, key=lambda point: (point.y, point.x)), points
        if len(actual) > 2:
            assert sp.Polygon(*actual).area > 0, points  # Signed area: counterclockwise.
    return cases


def check_v09_order(sp):
    """quantile (type 7) and mode against their formulas over SymPy Rationals on 60 seeded data sets."""
    from collections import Counter
    rng, cases = Random(90929), 60
    for _ in range(cases):
        data = tuple(mf.Rational(rng.randint(-6, 6), rng.randint(1, 2)) for _ in range(rng.randint(1, 9)))
        ordered = sorted(_rational(sp, value) for value in data)
        p = sp.Rational(rng.randint(0, 12), 12)
        h = (len(ordered) - 1) * p
        j = int(sp.floor(h))
        expected = ordered[j] if h == j else ordered[j] + (h - j) * (ordered[j + 1] - ordered[j])
        assert _rational(sp, mf.stats.quantile(data, mf.Rational(int(p.p), int(p.q)))) == expected, (data, p)
        counts = Counter(ordered)
        top = max(counts.values())
        expected = sorted(value for value, count in counts.items() if count == top)
        assert [_rational(sp, value) for value in mf.stats.mode(data)] == expected, data
    return cases


def check_v09_markov(sp):
    """stationary_distribution against the normalized null vector of M^T - I on 40 seeded chains with positive
    rational transition probabilities (so the stationary distribution is unique)."""
    rng, cases = Random(90930), 40
    for _ in range(cases):
        size = rng.randint(1, 5)
        weights = [[rng.randint(1, 9) for _ in range(size)] for _ in range(size)]
        oracle = sp.Matrix(size, size, lambda i, j: sp.Rational(weights[i][j], sum(weights[i])))
        rows = tuple(tuple(mf.Rational(int(oracle[i, j].p), int(oracle[i, j].q)) for j in range(size))
                     for i in range(size))
        (vector,) = (oracle.T - sp.eye(size)).nullspace()
        expected = [value / sum(vector) for value in vector]
        actual = mf.stats.stationary_distribution(mf.Matrix(rows))
        assert [_rational(sp, value) for value in actual] == expected, rows
    return cases


def check_v09_exact_tests(sp):
    """fisher_exact against the two-sided sum of hypergeometric masses (binomial) on 60 seeded 2 x 2 tables."""
    rng, cases = Random(90931), 60
    for _ in range(cases):
        table = [[rng.randint(0, 12) for _ in range(2)] for _ in range(2)]
        if not any(map(any, table)):
            table[0][0] = 1
        (a, b), (c, d) = table
        rows_, first, total = (a + b, c + d), a + c, a + b + c + d
        mass = {k: sp.binomial(rows_[0], k) * sp.binomial(rows_[1], first - k) / sp.binomial(total, first)
                for k in range(max(0, first - rows_[1]), min(rows_[0], first) + 1)}
        expected = sum(value for value in mass.values() if value <= mass[a])
        assert _rational(sp, mf.stats.fisher_exact(mf.Matrix(table))) == expected, table
    return cases


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
    v08_cases = check_v08(sp)
    print(f"{v08_cases} v0.8 continued fractions of seeded rationals agree with SymPy 1.14.0.")
    v09_cases = (check_v09_modular(sp), check_v09_prime_search(sp), check_v09_diophantine(sp),
                 check_v09_quadratic_irrational(sp), check_v09_arithmetic_functions(sp))
    print(f"{' + '.join(map(str, v09_cases))} v0.9 modular, prime search, linear Diophantine, quadratic irrational and "
          "arithmetic function cases agree with SymPy 1.14.0.")
    algebra_cases = (check_v09_roots(sp), check_v09_polynomial_tools(sp), check_v09_radicals(sp),
                     check_v09_piecewise(sp), check_v09_interval_algebra(sp), check_v09_digits(sp))
    print(f"{' + '.join(map(str, algebra_cases))} v0.9 root, interpolation, denesting, piecewise equation, interval "
          "set and decimal digit cases agree with SymPy 1.14.0.")
    analysis_cases = (check_v09_sums(sp), check_v09_power_series(sp), check_v09_trig_powers(sp))
    print(f"{' + '.join(map(str, analysis_cases))} v0.9 sum, power series and trigonometric integral cases agree with "
          "SymPy 1.14.0.")
    geometry_cases = (check_v09_powers(sp), check_v09_eigen(sp), check_v09_products(sp), check_v09_plane(sp),
                      check_v09_bivariate(sp), check_v09_counting(sp))
    print(f"{' + '.join(map(str, geometry_cases))} v0.9 matrix power, eigenvector, vector product, plane geometry, "
          "bivariate statistics and counting cases agree with SymPy 1.14.0.")
    extended_cases = (check_v09_sequences(sp), check_v09_approximation(sp), check_v09_symmetric(sp),
                      check_v09_laurent(sp), check_v09_discrete(sp), check_v09_forms(sp), check_v09_lattice(sp),
                      check_v09_hull(sp), check_v09_order(sp), check_v09_markov(sp), check_v09_exact_tests(sp))
    print(f"{' + '.join(map(str, extended_cases))} v0.9 Sturm/cyclotomic, rational approximation, symmetric "
          "difference, residue, discrete logarithm/two squares, definiteness/Gram-Schmidt, Hermite/Smith form, convex "
          "hull, quantile/mode, stationary distribution and Fisher test cases agree with SymPy 1.14.0.")


if __name__ == "__main__":
    main()
