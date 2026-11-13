# Using the v0.9 mathematics

MathForge 0.9 adds 65 operations. This page describes them, one section per area; the tested
[v0.9 workflow](../examples/v09_workflow.py) runs a selection of them end to end. Every
operation here is a v6 family, like `continued_fraction` (see [Find operations by area](areas.md)): a producer computes
the value and its evidence, and an independent checker verifies the claim. The operations are reached through their
area module, such as `mf.ntheory` or `mf.algebra`, and through `mathforge.operations`, never through the `mathforge`
namespace itself, and their results use JSON format version 6. The MCP area tool of the area serves them as well (see
the [MCP guide](mcp.md)).

An area function returns the value and raises `OperationError` when there is none. Where an operation has no rule for
an input in its scope, its structured twin in `mathforge.operations` returns a completed `Result` with the outcome
`NO_CONCLUSION`, no value and an `Unevaluated` record in `deferred` that gives the reason; no claim is made.

## Number theory

Seventeen operations join `mf.ntheory`. Every argument is an exact `int`, except the `Rational` or `int` that
`convergents` takes, and none has a default; a `bool`, a `float` or a `str` is invalid input.

| Family | Operations | Values |
| --- | --- | --- |
| Modular arithmetic | `powmod`, `multiplicative_order`, `primitive_root`, `legendre_symbol`, `jacobi_symbol`, `sqrt_mod` | Integers; `sqrt_mod` gives a sorted tuple |
| Nearest primes | `next_prime`, `prev_prime` | Integers |
| Linear Diophantine equations | `linear_diophantine` | A `DiophantineSolution` |
| Quadratic irrationals | `quadratic_continued_fraction`, `convergents`, `pell_fundamental` | A `PeriodicContinuedFraction`, a tuple of `Rational`s, a pair of integers |
| Arithmetic functions | `moebius`, `divisor_sigma`, `integer_nth_root` | Integers |
| Discrete logarithms and sums of two squares | `discrete_log`, `two_squares` | A `DiscreteLogarithm`, a `SumOfTwoSquares` |

### Modular arithmetic

```python
import mathforge as mf

assert mf.ntheory.powmod(7, 123456, 1000003) == pow(7, 123456, 1000003)
assert mf.ntheory.powmod(3, -1, 7) == 5  # A negative power raises the inverse.
assert mf.ntheory.multiplicative_order(10, 13) == 6
assert mf.ntheory.primitive_root(23) == 5
assert mf.ntheory.legendre_symbol(2, 7) == 1 and mf.ntheory.legendre_symbol(3, 7) == -1
assert mf.ntheory.jacobi_symbol(1001, 9907) == -1 and mf.ntheory.jacobi_symbol(3, 9) == 0
assert mf.ntheory.sqrt_mod(10, 13) == (6, 7) and mf.ntheory.sqrt_mod(3, 7) == ()
assert mf.ntheory.sqrt_mod(0, 13) == (0,)
```

`powmod(base, exponent, modulus)` is the residue in `[0, modulus)` for a modulus of at least 1; a negative exponent
needs `gcd(base, modulus) = 1`. `multiplicative_order(a, n)` is the least `d >= 1` with `a**d ≡ 1 (mod n)`, for
`gcd(a, n) = 1`. `primitive_root(p)` is the smallest generator modulo a prime `p`, and 1 for `p = 2`.
`legendre_symbol(a, p)` takes an odd prime `p`, and `jacobi_symbol(a, n)` an odd `n >= 1`. `sqrt_mod(a, p)` lists
every square root modulo a prime `p` in increasing order: `(r, p - r)` for a quadratic residue, `(0,)` when `p` divides
`a`, `()` for a non-residue and `(a % 2,)` for `p = 2`. Inputs outside these domains are invalid input:

```python
for failed in (mf.operations.powmod(2, -1, 4), mf.operations.multiplicative_order(2, 4),
               mf.operations.primitive_root(9), mf.operations.legendre_symbol(5, 2),
               mf.operations.jacobi_symbol(2, 8), mf.operations.sqrt_mod(1, 15)):
    assert failed.execution_status is mf.ExecutionStatus.INVALID_INPUT
```

### Nearest primes

```python
assert mf.ntheory.next_prime(13) == 17 and mf.ntheory.next_prime(-5) == 2
assert mf.ntheory.prev_prime(100) == 97 and mf.ntheory.prev_prime(3) == 2
assert mf.operations.prev_prime(2).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`next_prime(n)` is the smallest prime above `n`, which is 2 for every `n < 2`. `prev_prime(n)` is the largest prime
below `n`, so it needs `n >= 3`. The checker visits every integer between `n` and the result and requires each to be
composite: below 10,000 by trial division; from 10,000 on by a prime factor up to 97, by a failed Fermat test to base 2,
or, for the base-2 pseudoprimes that remain, by a witness in the evidence: a base `w` with `w**(m - 1) ≢ 1 (mod m)`, or
a proper divisor, which the producer gives for a Carmichael number.

```python
gap = mf.operations.next_prime(3828000)  # It passes the Carmichael number 3828001 = 101 * 151 * 251.
assert gap.value == 3828007
assert gap.steps[0].outputs[1] == ((), (("factor", 3828001, 37901),))
```

The result itself is proved prime: by trial division below 10,000, by the strong probable-prime tests to the thirteen
prime bases 2 to 41 below psi_13 = 3,317,044,064,679,887,385,961,981, since no composite below it passes them all, and
from psi_13 on by a Pocklington certificate in the evidence. That certificate needs the complete factorization of
`r - 1`. When the work budget cannot provide it, the request ends with `resource_limit_exceeded` rather than a value,
as `next_prime(10**50)` does under the default limits:

```python
mersenne = mf.operations.next_prime(2**89 - 2)  # 2**89 - 1 lies above psi_13.
assert mersenne.value == 2**89 - 1 and mersenne.steps[0].outputs[1][0][0] == "pocklington"
assert mf.verify(mersenne).verified
assert mf.operations.next_prime(10**50).error.code == "resource_limit_exceeded"
```

### Linear Diophantine equations

```python
solution = mf.ntheory.linear_diophantine(6, 9, 15)
assert (solution.x, solution.y, solution.dx, solution.dy) == (1, 1, 3, -2)
assert mf.render_text(solution) == "(1 + 3*t, 1 - 2*t) for every integer t"
assert mf.render_text(mf.ntheory.linear_diophantine(6, 9, 10)) == "no integer solution"
assert mf.operations.linear_diophantine(0, 0, 0).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`linear_diophantine(a, b, c)` returns every integer solution of `a*x + b*y = c` as one `DiophantineSolution`: the pairs
`(x + t*dx, y + t*dy)` for every integer `t`, in the canonical form that each solution set has exactly once, with
`dx > 0` and `0 <= x < dx`, or with `dx = 0`, `dy = 1` and `y = 0` when `b = 0`. Four `None` fields say that the
equation has no integer solution. `a = b = 0` is invalid input.

### Quadratic irrationals and Pell's equation

```python
sqrt2 = mf.ntheory.quadratic_continued_fraction(0, 1, 2, 1)  # (0 + 1*sqrt(2))/1
assert (sqrt2.preperiod, sqrt2.period) == ((1,), (2,)) and mf.render_text(sqrt2) == "[1; (2)]"
assert mf.render_text(mf.ntheory.quadratic_continued_fraction(1, 1, 5, 2)) == "[(1)]"  # The golden ratio.
assert mf.render_text(mf.ntheory.quadratic_continued_fraction(0, -1, 2, 1)) == "[-2; 1, 1, (2)]"
Q = mf.Rational
assert mf.ntheory.convergents(Q(415, 93)) == (Q(4), Q(9, 2), Q(58, 13), Q(415, 93))
assert mf.ntheory.pell_fundamental(13) == (649, 180)
assert mf.ntheory.pell_fundamental(61) == (1766319049, 226153980)
```

`quadratic_continued_fraction(p, q, d, r)` expands `(p + q*sqrt(d))/r`, for a `d >= 2` that is no perfect square and
nonzero `q` and `r`, as a `PeriodicContinuedFraction`: the shortest preperiod, then the minimal period, which repeats
forever. `convergents(value)` lists the convergents of the finite continued fraction of an exact rational.
`pell_fundamental(d)` is the least solution of `x**2 - d*y**2 = 1` in positive integers, for a `d >= 2` that is no
perfect square.

The checker verifies expansions of at most 99,000 terms, preperiod and period together, whose terms have at most
`max_integer_bits` bits in all. Beyond these caps, and beyond what a verifiable Result holds under
`max_evidence_items`, an expansion is `resource_limit_exceeded`. So is a Pell solution whose `x` or `y` has more than
`max_integer_bits` bits: for `d = 10**9 + 7` the solution has 21,198 bits, for `d = 1749209053` it has 66,447, more
than the default 65,536, and wider limits admit it.

```python
assert mf.ntheory.pell_fundamental(10**9 + 7)[0].bit_length() == 21198
assert mf.operations.pell_fundamental(1749209053).error.code == "resource_limit_exceeded"
wide = mf.ComputationLimits(max_integer_bits=70_000)
assert mf.operations.pell_fundamental(1749209053, limits=wide).value[0].bit_length() == 66447
```

### Arithmetic functions

```python
assert mf.ntheory.moebius(30) == -1 and mf.ntheory.moebius(12) == 0 and mf.ntheory.moebius(1) == 1
assert mf.ntheory.divisor_sigma(12, 1) == 28 and mf.ntheory.divisor_sigma(12, 0) == 6
assert mf.ntheory.integer_nth_root(26, 3) == 2 and mf.ntheory.integer_nth_root(10**20, 2) == 10**10
```

`moebius(n)` takes `n >= 1`, and `divisor_sigma(n, k)`, the sum of the `k`-th powers of the divisors of `n`, takes
`n >= 1` and `k >= 0`. Their evidence is the prime factorization of `n`, or, for a Moebius value of 0, a square that
divides `n`. Factoring is their limit, as it is for `multiplicative_order` and `primitive_root`, which factor `n` and
`p - 1`: under the default limits the product of the primes `2**89 - 1` and `2**107 - 1` is beyond the work budget.
`divisor_sigma` also needs `k * n.bit_length()` to stay within `max_integer_bits`. `integer_nth_root(n, k)` is the
integer part of the `k`-th root of `n >= 0`, for `k >= 1`.

```python
M89, M107 = 2**89 - 1, 2**107 - 1
assert mf.operations.moebius(M89 * M107).error.code == "resource_limit_exceeded"
assert mf.operations.divisor_sigma(3, 32769).error.code == "resource_limit_exceeded"  # 2 * 32769 bits > 65,536
```

### Discrete logarithms and sums of two squares

```python
log = mf.ntheory.discrete_log(2, 3, 11)
assert (log.exponent, log.order) == (8, 10) and mf.render_text(log) == "k = 8 (mod 10)"
assert mf.ntheory.discrete_log(2, 3, 7).exponent is None  # The powers of 2 modulo 7 are 1, 2 and 4.
squares = mf.ntheory.two_squares(25)
assert (squares.count, squares.a, squares.b) == (12, 3, 4)
assert mf.render_text(squares) == "12 representations, e.g. 25 = 3^2 + 4^2"
assert mf.ntheory.two_squares(0).count == 1 and mf.render_text(mf.ntheory.two_squares(3)) == "no representation"
```

`discrete_log(base, target, p)` solves `base**k ≡ target (mod p)` for a prime `p` (a composite `p` is invalid input,
decided with a proof) and `base % p != 0`. The `DiscreteLogarithm` holds `order`, the multiplicative order of `base`,
and `exponent`, the least solution `k >= 0`, or `None` when there is none; every solution is `exponent + t*order`.
The evidence is the certified factorization of the order and the prime certificate of `p`. The producer runs
Pohlig-Hellman with baby-step giant-step on each prime power of the order, so an order with a large prime factor is
`resource_limit_exceeded` (a long power of a small prime, such as `2**2208`, also exceeds the budget), while a target
outside the powers of `base` is answered at once. `two_squares(n)` counts the pairs `(x, y)` of integers with
`x**2 + y**2 = n >= 0` and gives one representation `a**2 + b**2` with `0 <= a <= b` as an example; its evidence is
the certified factorization of `n`, so factoring is its limit, as for `moebius`.

```python
safe = 1208925819614629174708367  # 2*q + 1 with q prime, about 2**80
assert mf.operations.discrete_log(2, 3, safe).error.code == "resource_limit_exceeded"
assert mf.ntheory.discrete_log(2, 5, safe).exponent is None
assert mf.operations.discrete_log(2, 3, 15).execution_status is mf.ExecutionStatus.INVALID_INPUT
assert mf.operations.two_squares((2**61 - 1)**3).error.code == "resource_limit_exceeded"
```

### Results and their evidence

Each function above has a structured twin in `mathforge.operations` that returns the whole `Result`: the request,
the value with its evidence, and the independent checks, which `mf.verify` repeats without running the producer.

```python
result = mf.operations.primitive_root(23)
assert result.value == 5 and result.request.version == 6
factors, refutations = result.steps[0].outputs[1]
assert factors == ((2, 1, ()), (11, 1, ())) and refutations == (2, 2, 2)
assert mf.verify(result).verified and mf.loads(mf.dumps(result)) == result
```

The evidence of `primitive_root(23)` is the factorization of `23 - 1 = 2 * 11`, each prime with its certificate (`()`
below psi_13, where the checker needs none), and for each of 2, 3 and 4 a prime `q` of 22 with `h**(22/q) ≡ 1 (mod 23)`,
which shows that `h` is no generator. The checker proves the value with Lucas's test: `5**22 ≡ 1` and
`5**(22/q) ≢ 1` for `q = 2` and `q = 11`, so 5 has the order 22, 23 is prime and 5 generates its units.

## Algebra

Six operations join `mf.algebra`: `count_real_roots`, `complex_roots`, `interpolate`, `denest_sqrt`, `sturm_sequence`
and `cyclotomic_polynomial`.

```python
x = mf.symbol("x")
P = lambda expression: mf.Polynomial.from_expression(expression, x)

assert mf.algebra.count_real_roots(P(x**2 - 2)) == 2
assert mf.algebra.count_real_roots(P(x**2 - 2), 0, 2) == 1
assert mf.algebra.count_real_roots(P(x**3 - x), -1, 1) == 3
assert mf.algebra.count_real_roots(P(x**3 - x), "-inf", 0) == 2
assert mf.algebra.count_real_roots(P((x - 1)**2), 0, 2) == 1
```

`count_real_roots(polynomial, lower="-inf", upper="+inf")` counts the distinct real roots of a nonzero rational
polynomial in the closed interval `[lower, upper]`; an end is an `int`, a `Rational`, `"-inf"` (lower only) or
`"+inf"` (upper only), and `lower > upper` is invalid input.

```python
roots = mf.algebra.complex_roots(P((x**2 + 1)**2 * (x - 2)))
assert [(root.root.real, root.root.imag, root.multiplicity) for root in roots] == \
    [(Q(0), Q(-1), 2), (Q(0), Q(1), 2), (Q(2), Q(0), 1)]
assert len(mf.algebra.complex_roots(P(x**3 - 1))) == 3 and mf.algebra.complex_roots(P(5 + 0*x)) == ()
cubic = mf.operations.complex_roots(P(x**3 - 2))
assert cubic.value is None and cubic.outcome is mf.Outcome.NO_CONCLUSION
assert cubic.deferred.reason.detail == "an irreducible factor of degree three or more"
```

`complex_roots(polynomial)` lists every complex root with its multiplicity as the `ComplexRoot` records of
`mf.quadratic_roots`, sorted by real part, then imaginary part. It needs every irreducible factor over the rationals
to have degree one or two; otherwise the result is deferred, as for `x**3 - 2`. Its evidence is the factorization,
which the checker multiplies out.

```python
assert mf.render_text(mf.algebra.interpolate((0, 1, 2), (1, 3, 7), x)) == "x^2 + x + 1"
assert mf.render_text(mf.algebra.interpolate((Q(1, 2), 2), (0, 1), x)) == "2/3*x - 1/3"
assert mf.operations.interpolate((1, 1), (1, 2), x).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`interpolate(xs, ys, variable)` is the polynomial of least degree through the points `(xs[i], ys[i])`, for distinct
rational abscissas; repeated abscissas, lists of different lengths and no point are invalid input. The checker
evaluates the value at every node. Many distinct long denominators make the producer's work grow: under the default
limits about 70 abscissas with 16-bit denominators, or 50 with 30-bit ones, fit the budget.

```python
assert mf.render_text(mf.algebra.denest_sqrt(3, 2, 2)) == "sqrt(2) + 1"  # sqrt(3 + 2*sqrt(2))
assert mf.render_text(mf.algebra.denest_sqrt(7, -4, 3)) == "2 - sqrt(3)"
root = mf.algebra.denest_sqrt(5, 2, 6)
assert (root.x, root.y, root.sign) == (Q(3), Q(2), 1) and mf.render_text(root) == "sqrt(3) + sqrt(2)"
assert mf.operations.denest_sqrt(1, 1, 2).outcome is mf.Outcome.NO_CONCLUSION
assert mf.operations.denest_sqrt(-3, 2, 2).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`denest_sqrt(a, b, d)` writes `sqrt(a + b*sqrt(d))` as a `DenestedRoot`, `sqrt(x) + sign*sqrt(y)` with rationals
`x > y > 0`, for `b != 0` and an integer `d >= 2` that is no perfect square. Such a form exists exactly when
`a**2 - b**2*d` is the square of a rational; otherwise the result is deferred. A negative radicand `a + b*sqrt(d)` is
invalid input. The checker tests `x + y == a`, `4*x*y == b**2*d` and the sign.

```python
sturm = mf.algebra.sturm_sequence(P(x**3 - x))
assert mf.render_text(sturm) == "(x^3 - x, 3*x^2 - 1, 2/3*x, 1)"
assert len(mf.algebra.sturm_sequence(P(5 + 0*x))) == 1
assert mf.render_text(mf.algebra.cyclotomic_polynomial(12, x)) == "x^4 - x^2 + 1"
assert "- 2*x^41" in mf.render_text(mf.algebra.cyclotomic_polynomial(105, x))
assert mf.operations.cyclotomic_polynomial(4621, x).error.code == "resource_limit_exceeded"  # phi = 4620
```

`sturm_sequence(polynomial)` is the canonical Sturm sequence of a nonzero rational polynomial, without scaling:
`p`, `p'`, and then the negated remainders `-rem(p(i-1), p(i))` up to the last nonzero one. Its evidence is the
quotients of the divisions, which the checker multiplies back. Coefficient growth is charged to the work budget: dense
integer polynomials of degree about 55 can be answered by the producer and still end as `resource_limit_exceeded`
when the check does not fit. `cyclotomic_polynomial(n, variable)` is the `n`-th cyclotomic polynomial for `n >= 1`;
`n > 100000` or `phi(n) > 1024`, the degree bound of a `Polynomial`, is `resource_limit_exceeded`. The checker tests
`x**d - 1 == prod(Phi_e for e | d)` for every divisor `d` of `n`, which determines every factor uniquely.

## Equations

`mf.equations` gains `solve_abs` and `solve_sqrt`. Both take `(lhs, rhs, variable, relation="=")`, with the two sides
as `RealFunction`s of the one variable and `relation` one of `"="`, `"<"`, `"<="`, `">"` and `">="`. The value is
`EmptySet`, `UniversalSet` or an `IntervalSet` whose ends are rationals or real algebraic numbers.

```python
F = lambda text: mf.parse_function(text, symbols={"x": x})
solution = mf.equations.solve_abs(F("abs(x - 1)"), F("3"), x)
assert [(item.lower, item.upper) for item in solution.intervals] == [(Q(-2), Q(-2)), (Q(4), Q(4))]
assert mf.render_text(mf.equations.solve_abs(F("abs(x - 1) + abs(x + 2)"), F("3"), x)) == "[-2, 1]"
assert mf.render_text(mf.equations.solve_abs(F("abs(x^2 - 2)"), F("x"), x, "<")) == "(1, 2)"
assert mf.equations.solve_abs(F("abs(x)"), F("-1"), x) == mf.EmptySet()
assert mf.equations.solve_abs(F("abs(x)"), F("0"), x, ">=") == mf.UniversalSet()
```

`solve_abs` solves `lhs - rhs` built from polynomials and absolute values of polynomials: sums, products,
nonnegative integer powers and rational constants, with no nested absolute value and no division by a non-constant.
`solve_sqrt` solves `A(x)*sqrt(P(x)) + B(x)` with polynomials `A`, `B`, `P` and one radicand, on the domain `P >= 0`.
Anything outside these classes is invalid input that names the construct:

```python
assert mf.render_text(mf.equations.solve_sqrt(F("sqrt(x + 2)"), F("x"), x)) == "[2, 2]"
assert mf.render_text(mf.equations.solve_sqrt(F("sqrt(x - 1)"), F("x - 3"), x, "<")) == "(5, inf)"
assert mf.equations.solve_sqrt(F("sqrt(x)"), F("-1"), x) == mf.EmptySet()
outside = mf.operations.solve_sqrt(F("sqrt(x) + sin(x)"), F("1"), x)
assert outside.error.message == "The function sin lies outside the class of solve_sqrt"
assert mf.render_text(mf.equations.solve_abs(F("sqrt(x^2)"), F("x"), x)) == "[0, inf)"  # The parser makes abs(x).
```

The checker decides the claim on a certified cut of the real line into cells at rational points and at the real roots
of one polynomial (`mathforge.certify.cells`), by exact evaluation in every cell and at every cut point. The producer
factors over the integers, so a square-free problem polynomial of degree above 64 exceeds the default limits, as
`|x**70 - 1| < 1` does. An irrational end may be named through a reducible polynomial that the checker accepts: the
set is right, but such a value need not equal the producer's own, which compares by polynomial and root index.

## Sets

`mf.sets` gains `set_union`, `set_intersection`, `set_difference` and `set_complement` on finite unions of real
intervals. The value is the canonical `IntervalSet`, and the empty set is `IntervalSet(())`.

```python
I = mf.Interval
S = lambda *intervals: mf.IntervalSet(intervals)
a, b = S(I(Q(0), Q(2), True, True)), S(I(Q(1), Q(3), True, False))  # [0, 2] and [1, 3)
assert mf.render_text(mf.sets.set_union(a, b)) == "[0, 3)"
assert mf.render_text(mf.sets.set_intersection(a, b)) == "[1, 2]"
assert mf.render_text(mf.sets.set_difference(a, b)) == "[0, 1)"
assert mf.render_text(mf.sets.set_complement(S(I(Q(0), Q(1), True, False)))) == "(-inf, 0) union [1, inf)"
sqrt2 = mf.RealAlgebraicRoot((-2, 0, 1), 1)
cut = mf.sets.set_difference(S(I(mf.RealAlgebraicRoot((-2, 0, 1), 0), sqrt2, True, True)), S(I(Q(0), Q(1))))
assert [(item.lower, item.upper) for item in cut.intervals][1] == (Q(1), sqrt2)  # [-sqrt 2, 0] and [1, sqrt 2]
assert mf.sets.set_intersection(S(I(Q(0), Q(1))), S(I(Q(1), Q(2)))) == mf.IntervalSet(())
```

The checker reads the memberships of the inputs at every cell and cut point of the ends of the inputs and of the
claim. It refuses a true claim whose ends are named by a polynomial that no input uses; the producer never makes one.
`set_symmetric_difference(a, b)` is checked in the same way:

```python
assert mf.render_text(mf.sets.set_symmetric_difference(a, b)) == "[0, 1) union (2, 3)"
assert mf.sets.set_symmetric_difference(a, a) == mf.IntervalSet(())
point = S(I(Q(0), Q(0), True, True))
assert mf.render_text(mf.sets.set_symmetric_difference(S(I(None, None)), point)) == "(-inf, 0) union (0, inf)"
```

`IntervalSet` also has the unchecked methods `complement()`, `difference(other)` and `symmetric_difference(other)` and
the operators `|`, `&`, `-`, `^` and `~`, like its `union` and `intersection` methods:

```python
assert (a | b) == mf.sets.set_union(a, b) and (a - b) == mf.sets.set_difference(a, b)
assert (a ^ b) == mf.sets.set_symmetric_difference(a, b)
assert ~S(I(Q(0), Q(1), True, False)) == mf.sets.set_complement(S(I(Q(0), Q(1), True, False)))
```

## Decimal digits

`mf.numeric.digits(value, places)` truncates an exact real number toward zero to `places` decimal places, from 0 to
10,000. The value is an `int`, a `Rational`, a `RealAlgebraicRoot` or a closed expression built from rationals and
square roots of rationals. The result is a `TruncatedDecimal(sign, magnitude, places, exact)`: the value lies in
`sign * [magnitude, magnitude + 1) / 10**places`, and `exact` says that it equals `sign * magnitude / 10**places`.

```python
assert mf.render_text(mf.numeric.digits(mf.sqrt(2), 30)) == "1.414213562373095048801688724209..."
assert mf.render_text(mf.numeric.digits(Q(1, 4), 3)) == "0.250"  # exact
assert mf.render_text(mf.numeric.digits(Q(-1, 8), 2)) == "-0.12..."
cube_root = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)  # The real root of x**3 - 2.
assert mf.render_text(mf.numeric.digits(cube_root, 20)) == "1.25992104989487316476..."
third = mf.numeric.digits(Q(2, 3), 5)
assert (third.sign, third.magnitude, third.places, third.exact) == (1, 66666, 5, False)
assert mf.operations.digits(1, 10_001).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

The checker verifies a rational by multiplication, a real algebraic number by Sturm counts at the ends of the
decimal cell, and an expression by a rational enclosure inside the cell or, on a decimal boundary, by its exact
rational value. Expressions whose algebra is dense, such as the 16th power of a sum of ten square roots, exceed the
default limits.

## Rational approximation

`mf.numeric.rational_approximation(value, max_denominator)` finds the rational closest to an exact real number among
those with a denominator of at most `N = max_denominator`, for `1 <= N <= 2**64`; `value` is what `digits` accepts.
On a tie, a value exactly midway between two candidates, the smaller denominator wins, then the smaller fraction.

```python
assert mf.numeric.rational_approximation(mf.sqrt(2), 10) == Q(7, 5)
assert mf.numeric.rational_approximation(Q(314159265, 100000000), 1000) == Q(355, 113)
assert mf.numeric.rational_approximation(cube_root, 1000) == Q(635, 504)
assert mf.numeric.rational_approximation(Q(1, 4), 2) == Q(0)  # 0 and 1/2 are equally close.
best = mf.operations.rational_approximation(mf.sqrt(2), 100)
assert best.value == Q(140, 99) and best.steps[0].outputs[1] == (Q(140, 99), Q(99, 70))
assert mf.operations.rational_approximation(1, 0).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

The evidence is `()` when the value is itself a fraction with a small enough denominator, and otherwise its two
neighbours `l < value < r` in the Farey sequence of order `N`; the checker tests the Farey conditions and decides each
comparison with the value exactly, by the methods of the `digits` checker. A value that no enclosure within
`max_integer_bits` separates from a fraction, such as a high power of `1 + sqrt(2)`, is `resource_limit_exceeded`.

## Sums and power series

Six operations join `mf.series`.

```python
assert mf.render_text(mf.series.sum_polynomial(P(x**2))) == "1/3*x^3 + 1/2*x^2 + 1/6*x"
assert mf.series.geometric_sum(1, 2, 10) == Q(1023)
assert mf.series.geometric_series(1, Q(1, 2)) == Q(2)
assert mf.series.polynomial_geometric_series(P(x), Q(1, 2)) == Q(2)  # The sum of k/2**k.
assert mf.operations.geometric_series(1, 1).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`sum_polynomial(polynomial)` is the polynomial `S` in the same variable with `S(0) = 0` and `S(n) - S(n - 1) = p(n)`,
the sum of `p(k)` for `k = 1 .. n`. `geometric_sum(first, ratio, count)` adds `first * ratio**k` for
`k = 0 .. count - 1`, and `geometric_series(first, ratio)` sums the infinite series for `|ratio| < 1`.
`polynomial_geometric_series(polynomial, ratio)` sums `p(k) * ratio**k` over `k = 0, 1, 2, ...` for `|ratio| < 1`.
A sum whose value would exceed `max_integer_bits` is `resource_limit_exceeded`.

```python
assert mf.render_text(mf.series.series_coefficients(F("exp(x)"), x, 0, 4)) == \
    "1 + x + 1/2*x^2 + 1/6*x^3 + 1/24*x^4 + O(x^5)"
logarithm = mf.series.series_coefficients(F("log(x)"), x, 1, 3)
assert logarithm.coefficients == (Q(0), Q(1), Q(-1, 2), Q(1, 3)) and logarithm.center == Q(1)
assert mf.render_text(logarithm) == "(x - 1) - 1/2*(x - 1)^2 + 1/3*(x - 1)^3 + O((x - 1)^4)"
assert mf.operations.series_coefficients(F("exp(x)"), x, 1, 3).outcome is mf.Outcome.NO_CONCLUSION
assert mf.operations.series_coefficients(F("exp(x)"), x, 0, 4097).error.code == "resource_limit_exceeded"
```

`series_coefficients(function, variable, center, order)` gives the coefficients `c_0, ..., c_order` of the power
series of a real function at a rational center as a `PowerSeries`. Every part of the function must have a rational
series there: exp, the trigonometric and hyperbolic functions, atan and asin of an argument that vanishes at the
center, log of an argument equal to 1, sqrt of a positive rational square, abs of a nonzero argument, integer powers
and quotients with a nonzero denominator. Anything else, such as `exp(x)` at 1, is deferred. Orders run up to 4,096.
The checker recomputes the coefficients by the first-order recurrences of the Taylor checker and charges each step by
the length of its coefficients, so values whose check would not fit the budget, such as some square roots at orders
in the thousands, end as `resource_limit_exceeded`.

```python
R = lambda numerator, denominator: mf.rational_function(numerator, denominator, variable=x)
assert mf.series.residue(R(1, x**2 - 1), x, 1) == Q(1, 2)
assert mf.series.residue(R(x + 1, x**2), x, 0) == Q(1)
assert mf.series.residue(R(1, x**3), x, 0) == Q(0) and mf.series.residue(R(1, x - 2), x, 3) == Q(0)
pole = mf.operations.residue(R(1, x**2 - 1), x, 1)
assert pole.steps[0].outputs[1] == (1, (Q(1), Q(1)), (Q(1, 2),))  # Order 1, cofactor x + 1, principal part.
```

`residue(function, variable, point)` is the coefficient of `(x - a)**-1` in the Laurent expansion of a
`RationalFunction` at a rational point `a`, and `0` where `a` is no pole. The evidence is the pole order `k`, the
cofactor `R` with `D = (x - a)**k * R` for the denominator `D`, and the principal part; the checker multiplies it out.
An integer longer than `max_integer_bits` in an argument, the value or the evidence is `resource_limit_exceeded`.

## Trigonometric integrals

`mf.integrals` gains `trig_power_antiderivative` and `trig_power_integral` for polynomials in `sin(a*x)` and
`cos(a*x)` with one rational `a`.

```python
assert mf.render_text(mf.integrals.trig_power_antiderivative(F("sin(x)^2"), x)) == \
    "((1/2*x) + (-1/2*(sin(x)*cos(x))))"
assert mf.render_text(mf.integrals.trig_power_antiderivative(F("sin(2*x)*cos(2*x)^2"), x)) == \
    "(-1/6*(cos((2*x)))^3)"
assert mf.render_text(mf.integrals.trig_power_integral(F("sin(x)^2"), x, 0, 1)) == "pi/2"
assert mf.render_text(mf.integrals.trig_power_integral(F("sin(x)^3"), x, 0, 1)) == "4/3"
area = mf.integrals.trig_power_integral(F("sin(x)^2"), x, 0, Q(1, 3))
assert area.pi_coefficient == Q(1, 6) and mf.render_text(area) == "-sqrt(3)/8 + pi/6"
```

The antiderivative is `lambda*x + T(sin(a*x), cos(a*x))` with a polynomial `T`, no constant term and no multiple
angles; the checker differentiates it formally and reduces with `sin**2 = 1 - cos**2`. `trig_power_integral(function,
variable, lower_pi, upper_pi)` integrates from `lower_pi*pi` to `upper_pi*pi` and returns a `PiLinear`, an algebraic
part in `Q(sqrt 2, sqrt 3)` plus a rational multiple of pi. The angles `a*lower_pi*pi` and `a*upper_pi*pi` must be
multiples of `pi/12`, where sine and cosine are exact:

```python
assert mf.operations.trig_power_integral(F("sin(x)^2"), x, 0, Q(1, 24)).execution_status \
    is mf.ExecutionStatus.INVALID_INPUT
assert mf.operations.trig_power_antiderivative(F("x*sin(x)"), x).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

## Linear algebra

`mf.linalg` gains `matrix_power`, `eigenvectors`, `definiteness`, `gram_schmidt`, `hermite_normal_form` and
`smith_normal_form`.

```python
M = mf.Matrix(((1, 1), (1, 0)))
assert mf.linalg.matrix_power(M, 10) == mf.Matrix(((89, 55), (55, 34)))
assert mf.linalg.matrix_power(mf.Matrix(((2, 0), (0, 3))), 0) == mf.Matrix(((1, 0), (0, 1)))
assert M ** 10 == mf.linalg.matrix_power(M, 10)  # The same value, without the independent check.
negative = mf.operations.matrix_power(M, -1)
assert negative.error.message == "matrix_power requires k >= 0: use inverse for negative powers"
```

`matrix_power(M, k)` raises a square rational matrix to an integer power `k >= 0`, the identity for `k = 0`. The
checker recomputes the power modulo word primes whose product exceeds twice the bound `||s*M||_inf**k` on the
entries, with another addition chain. A power whose bound has more than `max_integer_bits` bits is
`resource_limit_exceeded`. `Matrix ** k` returns the producer's value as unchecked arithmetic, like `@`.

```python
spaces = mf.linalg.eigenvectors(mf.Matrix(((2, 1), (1, 2))))
assert [(space.value, space.multiplicity, space.basis) for space in spaces.eigenspaces] == \
    [(Q(1), 1, ((Q(-1), Q(1)),)), (Q(3), 1, ((Q(1), Q(1)),))]
assert spaces.residual == (Q(1),)
irrational = mf.linalg.eigenvectors(mf.Matrix(((0, 2), (1, 0))))
assert irrational.eigenspaces == () and irrational.residual == (Q(-2), Q(0), Q(1))  # x**2 - 2
jordan = mf.linalg.eigenvectors(mf.Matrix(((1, 1), (0, 1)))).eigenspaces[0]
assert (jordan.value, jordan.multiplicity, jordan.basis) == (Q(1), 2, ((Q(1), Q(0)),))
```

`eigenvectors(M)` finds every rational eigenvalue with its algebraic multiplicity and the canonical basis of its
eigenspace (the RREF nullspace basis of `M - value*I`), in increasing order. `residual` is the monic factor of the
characteristic polynomial without rational roots, lowest degree first, and `(1,)` when every eigenvalue is rational.
The evidence holds the characteristic polynomial's certificate, its factorization and one echelon certificate per
eigenvalue. Hensel lifting is charged to the budget, so long-entry matrices can end as `resource_limit_exceeded`.

```python
inertia = mf.linalg.definiteness(mf.Matrix(((2, 1), (1, 2))))
assert (inertia.positive, inertia.negative, inertia.zero, inertia.kind) == (2, 0, 0, "positive_definite")
assert mf.render_text(mf.linalg.definiteness(mf.Matrix(((1, 2), (2, 1))))) == "indefinite (1, 1, 0)"
assert mf.linalg.definiteness(mf.Matrix(((1, 1), (1, 1)))).kind == "positive_semidefinite"
assert mf.render_text(mf.linalg.gram_schmidt(mf.Matrix(((3, 1), (2, 2))))) == "[[3, 1], [-2/5, 6/5]]"
dependent = mf.operations.gram_schmidt(mf.Matrix(((1, 2), (2, 4))))
assert dependent.error.message.endswith("row 1 lies in the span of the rows before it")
```

`definiteness(M)` gives the `Inertia` of a symmetric rational matrix, the numbers of positive, negative and zero
eigenvalues, and its `kind`: `positive_definite`, `negative_definite`, `positive_semidefinite` (the zero matrix
included), `negative_semidefinite` or `indefinite`. The evidence is a congruence `S^T M S = D` with `D` diagonal and
`S` invertible, by Sylvester's law of inertia. `gram_schmidt(M)` orthogonalizes the linearly independent rows of a
rational matrix without normalizing them, `q_i = m_i - sum(mu_ij q_j)`, with the coefficients `mu` as evidence. Both
use fraction-free elimination, whose cost grows with the length of the entries: a 48 x 48 matrix with 100-bit
entries, or the 64 x 64 Hilbert matrix, is `resource_limit_exceeded` under the default limits although its check would
fit.

```python
assert mf.render_text(mf.linalg.hermite_normal_form(mf.Matrix(((2, 4), (3, 6))))) == "[[1, 2], [0, 0]]"
assert mf.linalg.hermite_normal_form(mf.Matrix(((2, 3), (5, 7)))) == mf.Matrix(((1, 0), (0, 1)))
assert mf.linalg.smith_normal_form(mf.Matrix(((2, 4), (6, 8)))) == (2, 4)
assert mf.linalg.smith_normal_form(mf.Matrix(((1, 2, 3), (4, 5, 6)))) == (1, 3)
fraction = mf.operations.hermite_normal_form(mf.Matrix(((Q(1, 2), 1), (1, 1))))
assert fraction.error.message == "hermite_normal_form requires integer entries, not a fraction"
```

`hermite_normal_form(M)` is the row-style Hermite normal form `H = U M` of an integer matrix, with `U` unimodular:
row echelon form, positive pivots, the entries above each pivot in `[0, pivot)` and zero rows last. The evidence is `U`
and its inverse. `smith_normal_form(M)` gives the invariant factors `d_1 | d_2 | ...`, `min(rows, cols)` of them with
zeros last, with the two unimodular transforms and their inverses as evidence. A dense 64 x 64 matrix with entries of
more than about 10 bits exceeds the producer's budget for the Hermite form.

## Vectors

`mf.vector` gains `dot`, `cross`, `projection` and `triple_product` on exact rational vectors.

```python
assert mf.vector.dot((1, 2, 3), (4, 5, 6)) == Q(32)
assert mf.vector.cross((1, 2, 3), (4, 5, 6)) == (Q(-3), Q(6), Q(-3))
assert mf.vector.projection((1, 2), (3, 0)) == (Q(1), Q(0))
assert mf.vector.triple_product((1, 0, 0), (0, 1, 0), (0, 0, 1)) == Q(1)
for failed in (mf.operations.cross((1, 2), (3, 4)), mf.operations.dot((1,), (1, 2)),
               mf.operations.projection((1, 2), (0, 0))):
    assert failed.execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`dot` takes two nonempty vectors of one length, `cross` and `triple_product` vectors of length three, and
`projection(a, b)` projects `a` onto a nonzero `b`. The checkers use identities: polarization for `dot`, Lagrange's
identity and the orientation for `cross`, orthogonality for `projection`.

## Plane geometry

`mf.geometry` gains `distance`, `polygon_area`, `line_intersection`, `orientation` and `convex_hull`. Points are
sequences of rationals; `polygon_area` and `convex_hull` take a `Matrix` with one row `(x, y)` per point.

```python
assert mf.geometry.distance((0, 0), (3, 4)) == Q(5)
assert mf.render_text(mf.geometry.distance((0, 0), (1, 1))) == "sqrt(2)"
assert mf.geometry.polygon_area(mf.Matrix(((0, 0), (4, 0), (0, 3)))) == Q(6)
assert mf.geometry.polygon_area(mf.Matrix(((0, 0), (0, 1), (1, 1), (1, 0)))) == Q(-1)  # Clockwise.
meet = mf.geometry.line_intersection((0, 0), (1, 1), (0, 1), (1, 0))
assert meet.kind == "point" and meet.point == (Q(1, 2), Q(1, 2))
assert mf.geometry.line_intersection((0, 0), (1, 0), (0, 1), (1, 1)).kind == "parallel"
assert mf.geometry.line_intersection((0, 0), (1, 1), (2, 2), (3, 3)).kind == "coincident"
assert mf.geometry.orientation((0, 0), (1, 0), (0, 1)) == 1 and mf.geometry.orientation((0, 0), (1, 1), (2, 2)) == 0
```

`distance(p, q)` takes two points of one dimension and returns a `Rational` when the squared distance is a rational
square, else its `sqrt`. `polygon_area` is the signed shoelace area, positive counterclockwise; a self-intersecting
polygon can give 0. `line_intersection(p1, p2, q1, q2)` intersects the line through `p1` and `p2` with the line
through `q1` and `q2` in the plane, as a `LineIntersection` whose `point` is set only for the kind `"point"`;
`p1 == p2` or `q1 == q2` is invalid input. `orientation(p, q, r)` is the sign of the turn from `p` through `q` to
`r`. Very large polygons on a line through the origin are answered by the producer but can exceed the checker's
budget, which then ends as `resource_limit_exceeded`.

```python
hull = mf.geometry.convex_hull(mf.Matrix(((0, 0), (2, 0), (1, 1), (2, 2), (0, 2))))
assert mf.render_text(hull) == "[[0, 0], [2, 0], [2, 2], [0, 2]]"
assert mf.render_text(mf.geometry.convex_hull(mf.Matrix(((0, 0), (1, 1), (2, 2))))) == "[[0, 0], [2, 2]]"
assert mf.render_text(mf.geometry.convex_hull(mf.Matrix(((3, 3), (3, 3))))) == "[[3, 3]]"
```

`convex_hull(points)` takes an `n x 2` `Matrix` of rational points, repeated points allowed, and returns the hull's
vertices counterclockwise from the lowest (then leftmost) point; points on an edge are no vertices, collinear points
give the two extreme ones and a single point itself. The checker tests the strict convex turns and the containment of
every other point. 2,048 points in convex position fit the default budget; many interior points with long coordinates
cost the checker up to about 1.7 times the producer's work and can end as `resource_limit_exceeded`.

## Statistics

`mf.stats` gains `correlation` and `linear_regression` on paired rational data, `quantile` and `mode`,
`stationary_distribution` and `fisher_exact`.

```python
assert mf.stats.correlation((1, 2, 3), (2, 4, 6)) == Q(1)
assert mf.stats.correlation((1, 2, 3, 4), (1, 3, 2, 4)) == Q(4, 5)
fit = mf.stats.linear_regression((0, 1, 2), (1, 2, 4))
assert (fit.slope, fit.intercept) == (Q(3, 2), Q(5, 6))
assert mf.render_text(fit) == "y = 3/2*x + 5/6"
assert mf.operations.correlation((1, 2, 3), (5, 5, 5)).execution_status is mf.ExecutionStatus.INVALID_INPUT
```

`correlation(xs, ys)` is the Pearson coefficient, a `Rational` when its square is a rational square and otherwise
an expression with a square root. `linear_regression(xs, ys)` fits the least-squares line as a `LinearFit`; its
checker tests the normal equations on the residuals. Both need at least two pairs of equal length; `correlation`
needs nonzero variance in both variables and `linear_regression` in the abscissas. A common denominator beyond
`max_integer_bits` is `resource_limit_exceeded`.

```python
assert mf.stats.quantile((1, 2, 3, 4), Q(1, 2)) == Q(5, 2) and mf.stats.quantile((1, 2, 3, 4), Q(1, 4)) == Q(7, 4)
assert mf.stats.quantile((3, 1, 2), 0) == Q(1)
assert mf.stats.mode((1, 1, 2, 2, 3)) == (Q(1), Q(2))
assert mf.operations.quantile((1, 2), Q(3, 2)).error.message == "quantile requires p with 0 <= p <= 1"
chain = mf.Matrix(((Q(1, 2), Q(1, 2)), (Q(1, 4), Q(3, 4))))
assert mf.stats.stationary_distribution(chain) == (Q(1, 3), Q(2, 3))
reducible = mf.operations.stationary_distribution(mf.Matrix(((1, 0), (0, 1))))  # Two closed classes.
assert reducible.execution_status is mf.ExecutionStatus.INVALID_INPUT
assert mf.stats.fisher_exact(mf.Matrix(((1, 9), (11, 3)))) == Q(41, 14858)
assert mf.stats.fisher_exact(mf.Matrix(((3, 1), (1, 3)))) == Q(17, 35)
```

`quantile(data, p)` is the type-7 quantile of nonempty rational data for `0 <= p <= 1`: with the sorted data and
`h = (n - 1) p`, the order statistic at `h` or the linear interpolation between its neighbours; the evidence names the
order statistics used. `mode(data)` lists every value of maximal multiplicity in increasing order. Data with about
8,300 distinct most frequent fractions (about 12,500 integers) exceed `max_evidence_items` under the verifier and end
as `resource_limit_exceeded`.
`stationary_distribution(M)` is the unique `pi` with `pi M = pi` and `sum(pi) = 1` for a square row-stochastic
rational matrix; a chain with more than one closed communicating class is invalid input. Its evidence is a nonsingular
`(n - 1) x (n - 1)` minor of `s (I - M)` with a prime that certifies it. Dense 64-state chains with long entries
(61-bit weights) exceed the producer's budget. `fisher_exact(table)` is the exact two-sided p-value of Fisher's test
on a 2 x 2 table of nonnegative counts, the sum of the hypergeometric masses at most the observed one (compared
exactly); a total above 100,000 is `resource_limit_exceeded`, and balanced tables with a total from about 52,000 can
exceed the checker's budget the same way.

## Counting

`mf.discrete` gains `partitions`, `derangements`, `bell` and `stirling1`.

```python
assert mf.discrete.partitions(100) == 190569292
assert mf.discrete.derangements(10) == 1334961
assert mf.discrete.bell(10) == 115975
assert mf.discrete.stirling1(4, 2) == 11 and mf.discrete.stirling1(5, 3) == 35
assert mf.operations.partitions(5001).error.code == "resource_limit_exceeded"
```

`partitions(n)` counts the integer partitions of `n`, `derangements(n)` the permutations of `n` items without a fixed
point, `bell(n)` the set partitions of `n` items and `stirling1(n, k)` the permutations of `n` items with `k` cycles
(the unsigned Stirling numbers of the first kind). Each takes integers `n >= 0`; the limits are `n <= 5000` for
`partitions`, 5900 for `derangements`, 1500 for `bell` and 1000 for `stirling1`, and a `k` longer than
`max_integer_bits` is refused although every `k > n` gives 0. The evidence of `partitions` is the whole table
`p(0), ..., p(n)`, which the checker tests with Euler's pentagonal recurrence and a divisor-sum identity.
