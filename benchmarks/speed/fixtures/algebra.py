"""Speed fixtures and value-corpus edge cases of the ``algebra`` area.

The area holds the exact complex scalars (``complex_*``, ``quadratic_roots``), the univariate and multivariate
polynomial operations, ``real_roots``, ``compare_real``, ``rational_function``, ``factor``, ``resultant``,
``discriminant``, ``apart`` and the real algebraic number operations (``algebraic``, ``algebraic_div``,
``algebraic_pow``, ``minimal_polynomial``). ``expr.py`` and ``equations.py`` import the helpers of this module.

Reference rule: a SymPy-level public function or object method that a user calls for the operation and that returns
the same mathematical object (``Poly`` methods, ``minimal_polynomial``, ``real_roots``, ``factor_list``, ``apart``,
``solve`` and the like, ``Matrix`` and ``DomainMatrix`` methods); where several such calls exist, the fastest. Where
SymPy has no call for the operation, the reference is the fastest expression a user writes with SymPy objects
(``a + b`` or ``Add(a, b)``, ``expand(a*b)`` or ``expand_complex(a*b)``, ``Abs(a)**2`` or ``re(a)**2 + im(a)**2``,
``sign(a - b)``): a shorter spelling does not win a tie, and a faster one replaces it. The reference is always the
candidate with the smallest median. Ground arithmetic is excluded: domain elements (``QQ_I`` and ``ZZ_I`` Gaussian
numbers, ZZ/QQ methods, ``PythonMPQ``, ``ANP``/``AlgebraicField`` elements) and the ``dup_*``, ``dmp_*`` and ``gf_*``
helpers. The reference of every fixture lists the candidates that were measured, with their medians (SymPy caches
cleared before each call); a reference that was re-pointed to a faster call was measured again with the suite's own
sampling, together with the call it replaced: 31 to 127 interleaved samples of k calls, caches reset before each.

* Polynomial operations time ``Poly`` methods on prebuilt ``Poly`` objects over QQ (ZZ for ``resultant`` and
  ``discriminant``, whose value is an integer).
* A complex result is a Cartesian value: the SymPy call must determine the real and imaginary parts (``expand``,
  ``expand_complex``, ``sqrt`` of a perfect square), so a lazy ``(a + b*I)*(c + d*I)`` does not count as a reference.
* MathForge returns an irrational real algebraic number as a root descriptor (minimal polynomial and index), and the
  spec pins ``algebraic_eval`` to ``minimal_polynomial`` because that is SymPy's canonical name for such a number.
  A fixture whose value has irrational parts therefore also times ``minimal_polynomial(part, t, compose=False)`` of
  every irrational part (``compose=False`` is the faster public mode on nearly every input measured). The timed call
  returns the names together with the prebuilt numbers they name, so that ``agree`` can also check which root is meant.
* The Gaussian domain elements (``QQ_I``) are ground arithmetic and so are not references; their medians are still
  listed in the references of the complex micro fixtures.
* There is no non-square ``complex_sqrt`` fixture: SymPy has no public call that returns the algebraic Cartesian parts
  of ``sqrt(3 + 5*I)`` (``expand_complex`` gives ``cos(atan(5/3)/2)`` forms that ``minimal_polynomial`` rejects).

SymPy is imported only inside the ``sympy`` builders and the ``agree`` functions. Builders are deterministic: the
inputs of variant ``i`` depend on ``i`` alone, through the integer hash ``_mix`` and the digit scramble ``_spread``,
never through ``random``. A SymPy builder binds the SymPy callables outside the timed call.

Edge cases follow the corpus rules: per operation one input for each v0.7 error class it can raise, one input far past
a default limit (an integer of 140,001 bits against 65,536, an expression of 300,000 nodes against 100,000) and the
``True``, ``1.5``, ``"1"`` and ``None`` type edges in the first argument.
"""

from __future__ import annotations

import functools

import mathforge as mf
from benchmarks.speed.fixture import EdgeCase, Fixture, symbol

X = symbol("x", 1)
Y = symbol("y", 2)
Z = symbol("z", 3)

_MASK = (1 << 64) - 1
_MICRO = 1000  # variants of a micro fixture
_HUGE = 1 << 140_000  # about 2.1x max_integer_bits (65,536): far past the limit, not just past it


# ---------------------------------------------------------------------------------------------------------------
# Deterministic inputs
# ---------------------------------------------------------------------------------------------------------------

def _mix(*values: int) -> int:
    """A 64-bit integer hash of the arguments (splitmix style): the same in every process and Python version."""
    h = 0x9E3779B97F4A7C15
    for value in values:
        h = ((h ^ (value & _MASK)) * 0xBF58476D1CE4E5B9) & _MASK
        h ^= h >> 31
    return h


def _draw(low: int, high: int, *key: int) -> int:
    """An integer in [low, high] chosen by the key."""
    return low + _mix(*key) % (high - low + 1)


def _spread(i: int, *bases: int, tag: int = 0) -> tuple[int, ...]:
    """One digit per base, from a scramble of i: distinct i below the product of the bases give distinct tuples.

    The multiplier 48271 is prime, so ``i -> i * 48271 + tag * 7919`` is a bijection modulo the product of the bases.
    """
    total = 1
    for base in bases:
        total *= base
    n = (i * 48271 + tag * 7919) % total
    digits = []
    for base in bases:
        n, digit = divmod(n, base)
        digits.append(digit)
    return tuple(digits)


def _mathforge(function, inputs):
    """MathForge builder: variant i -> (function, inputs(i), no keywords)."""
    return lambda i: (function, inputs(i), {})


def _sympy(inputs, call):
    """SymPy builder: variant i -> the zero-argument call ``call(sympy, *inputs(i))`` with its inputs prebuilt.

    ``inputs`` and ``call`` stay readable on the builder, so that a measurement can time other candidates on the
    same prebuilt inputs.
    """
    def build(i):
        import sympy as sp
        return functools.partial(call, sp, *inputs(i))
    build.inputs, build.call = inputs, call
    return build


# ---------------------------------------------------------------------------------------------------------------
# Comparing MathForge values with SymPy values (not timed)
# ---------------------------------------------------------------------------------------------------------------

def _sp_real(value):
    """A MathForge real algebraic scalar (Rational or RealAlgebraicRoot) as a SymPy number; roots in CRootOf order."""
    import sympy as sp
    if type(value) is mf.Rational:
        return sp.Rational(value.numerator, value.denominator)
    t = sp.Symbol("t")
    return sp.CRootOf(sp.Poly(list(reversed(value.integer_coefficients)), t), value.real_index)


def _same_number(a, b) -> bool:
    """Exact for rationals; 60 correct digits otherwise (distinct algebraic numbers of these sizes differ by more)."""
    import sympy as sp
    a, b = sp.sympify(a), sp.sympify(b)
    if a.is_Rational and b.is_Rational:
        return a == b
    return bool(abs(sp.N(a - b, 90)) < sp.Float(10) ** -60)


def _sp_complex(value):
    """A MathForge ComplexAlgebraic as ``real + I*imag`` in SymPy."""
    import sympy as sp
    return _sp_real(value.real) + sp.I * _sp_real(value.imag)


def _same_complex(mf_value, sympy_value) -> bool:
    """The MathForge ComplexAlgebraic and the SymPy expression (or (real, imaginary) pair) are the same number."""
    import sympy as sp
    if type(mf_value) is not mf.ComplexAlgebraic:
        return False
    if isinstance(sympy_value, tuple):
        sympy_value = sympy_value[0] + sp.I * sympy_value[1]
    difference = sp.sympify(sympy_value) - _sp_complex(mf_value)
    return sp.expand(difference) == 0 or bool(abs(sp.N(difference, 90)) < sp.Float(10) ** -60)


def _same_real(mf_value, sympy_value) -> bool:
    return type(mf_value) in (mf.Rational, mf.RealAlgebraicRoot) and _same_number(_sp_real(mf_value), sympy_value)


def _identify(sp, part, t):
    """SymPy's canonical name of a real algebraic number: itself when rational, else its minimal polynomial in t."""
    return part if part.is_Rational else sp.minimal_polynomial(part, t, compose=False)


def _name_matches(mf_part, name) -> bool:
    """The SymPy name (a rational, or a minimal polynomial in t) is the one the MathForge part descriptor stands for."""
    import sympy as sp
    t = sp.Symbol("t")
    if type(mf_part) is mf.Rational:
        number, name = sp.Rational(mf_part.numerator, mf_part.denominator), sp.sympify(name)
        return name == number or sp.expand(name - (t - number)) == 0  # the number, or its minimal polynomial t - number
    mine = sp.Poly(list(reversed(mf_part.integer_coefficients)), t, domain="QQ").monic().as_expr()
    theirs = sp.Poly(name, t, domain="QQ").monic().as_expr()
    return sp.expand(mine - theirs) == 0


def _same_named_complex(mf_value, found) -> bool:
    """A ComplexAlgebraic against (name of real part, name of imaginary part, real part, imaginary part)."""
    name_real, name_imag, real, imag = found
    return (type(mf_value) is mf.ComplexAlgebraic
            and _same_number(_sp_real(mf_value.real), real) and _same_number(_sp_real(mf_value.imag), imag)
            and _name_matches(mf_value.real, name_real) and _name_matches(mf_value.imag, name_imag))


def _same_roots(mf_value, sympy_value) -> bool:
    """A tuple of ComplexRoot against SymPy roots, given as a dict {root: multiplicity} or as a list with repeats."""
    import sympy as sp
    expected = [(_sp_complex(record.root), record.multiplicity) for record in mf_value]
    if isinstance(sympy_value, dict):
        found = list(sympy_value.items())
    else:
        found = []
        for root in sympy_value:
            for index, (known, count) in enumerate(found):
                if _same_number(root, known):
                    found[index] = (known, count + 1)
                    break
            else:
                found.append((root, 1))
    if len(found) != len(expected):
        return False
    return all(any(count == multiplicity and _same_number(root, other) for other, count in found)
               for root, multiplicity in expected)


def _same_named_roots(mf_value, found) -> bool:
    """A tuple of distinct ComplexRoot against [(root, name of real part, name of imaginary part), ...]."""
    if len(found) != len(mf_value) or any(record.multiplicity != 1 for record in mf_value):
        return False
    for record in mf_value:
        for root, name_real, name_imag in found:
            if _same_number(_sp_complex(record.root), root):
                if not (_name_matches(record.root.real, name_real) and _name_matches(record.root.imag, name_imag)):
                    return False
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# Complex scalars
# ---------------------------------------------------------------------------------------------------------------
# A real part is a tuple of terms (numerator, denominator, radicand) that stands for the sum of
# numerator/denominator * sqrt(radicand); radicand 1 makes a rational term. A complex operand is (real, imaginary).

def _terms_mf(terms):
    value = mf.Rational(0)
    for n, d, r in terms:
        value = value + mf.Rational(n, d) * mf.sqrt(r)
    return value


def _terms_sp(terms):
    import sympy as sp
    value = sp.Integer(0)
    for n, d, r in terms:
        value = value + sp.Rational(n, d) * sp.sqrt(r)
    return value


@functools.cache
def _zm(operand):
    """The MathForge ComplexAlgebraic of an operand description, built once."""
    real, imag = operand
    return mf.complex_algebraic(_terms_mf(real), _terms_mf(imag))


def _zs(operand):
    """The SymPy complex number of an operand description."""
    import sympy as sp
    real, imag = operand
    return _terms_sp(real) + sp.I * _terms_sp(imag)


def _gauss_pair(i: int, k: int, divisor: bool = False) -> tuple[int, int]:
    """Micro operand k (0 or 1) of variant i as (a, b): the Gaussian integer a + b*i with a, b in [-9, 9].

    A divisor has a real part in [1, 9], so it is never zero. Distinct i below 19**4 give distinct operand pairs.
    """
    digits = _spread(i, 19, 19, 9 if divisor else 19, 19)
    a = digits[2] + 1 if divisor and k == 1 else digits[2 * k] - 9
    return a, digits[2 * k + 1] - 9


def _gauss(i: int, k: int, divisor: bool = False):
    a, b = _gauss_pair(i, k, divisor)
    return (((a, 1, 1),), ((b, 1, 1),))


def _fraction_gauss(i: int, k: int, divisor: bool = False):
    """Realistic operand k of variant i: n1/d1 + (n2/d2) i with |n| <= 999 and 2 <= d <= 99; a divisor has n1 > 0."""
    def part(which):
        n = _draw(1, 999, 2, i, k, which, 0)
        if not (divisor and which == 0) and _draw(0, 1, 2, i, k, which, 1):
            n = -n
        return ((n, _draw(2, 99, 2, i, k, which, 2), 1),)
    return part(0), part(1)


_SURDS = ((2, 3), (5, 7), (3, 11))  # radicands (p, q) of the three surd variants


def _surd_left(i: int):
    """sqrt(p) + i"""
    return (((1, 1, _SURDS[i][0]),), ((1, 1, 1),))


def _surd_right(i: int):
    """1 + sqrt(q)*i"""
    return (((1, 1, 1),), ((1, 1, _SURDS[i][1]),))


def _binary_fixtures(function, call, micro_reference, realistic_reference, *, divisor=False, realistic_call=None):
    """Micro (Gaussian integers) and realistic (fractions) fixtures of a binary operation with a Cartesian value. The
    call of each size is the fastest one measured at that size: ``call`` for the micro fixture, ``realistic_call`` (by
    default the same) for the realistic one."""
    return (
        Fixture(function.__name__, "micro", "gaussian_integers",
                _mathforge(function, lambda i: (_zm(_gauss(i, 0)), _zm(_gauss(i, 1, divisor)))),
                _sympy(lambda i: (_zs(_gauss(i, 0)), _zs(_gauss(i, 1, divisor))), call), _same_complex,
                micro_reference, variants=_MICRO),
        Fixture(function.__name__, "realistic", "gaussian_fractions",
                _mathforge(function, lambda i: (_zm(_fraction_gauss(i, 0)), _zm(_fraction_gauss(i, 1, divisor)))),
                _sympy(lambda i: (_zs(_fraction_gauss(i, 0)), _zs(_fraction_gauss(i, 1, divisor))),
                       realistic_call or call),
                _same_complex, realistic_reference, variants=len(_SURDS)),
    )


def _unary_fixtures(function, call, agree, micro_reference, realistic_reference):
    return (
        Fixture(function.__name__, "micro", "gaussian_integer",
                _mathforge(function, lambda i: (_zm(_gauss(i, 0)),)),
                _sympy(lambda i: (_zs(_gauss(i, 0)),), call), agree, micro_reference, variants=_MICRO),
        Fixture(function.__name__, "realistic", "gaussian_fraction",
                _mathforge(function, lambda i: (_zm(_fraction_gauss(i, 0)),)),
                _sympy(lambda i: (_zs(_fraction_gauss(i, 0)),), call), agree, realistic_reference,
                variants=len(_SURDS)),
    )


def _identified_parts(sp, a, b, t):
    return _identify(sp, a, t), _identify(sp, b, t), a, b


def _named_product(sp, a, b, t):
    re, im = sp.expand(a * b).as_real_imag()
    return _identify(sp, re, t), _identify(sp, im, t), re, im


def _named_quotient(sp, a, b, t):
    re, im = sp.expand_complex(a / b).as_real_imag()
    return _identify(sp, re, t), _identify(sp, im, t), re, im


def _identify_composed(sp, part, t):
    """_identify with minimal_polynomial's default (composing) mode, which is the faster one on a few inputs."""
    return part if part.is_Rational else sp.minimal_polynomial(part, t)


def _named_power_composed(sp, a, n, t):
    re, im = sp.expand(a ** n).as_real_imag()
    return _identify_composed(sp, re, t), _identify_composed(sp, im, t), re, im


def _surd_inputs(i):
    import sympy as sp
    return _zs(_surd_left(i)), _zs(_surd_right(i)), sp.Symbol("t")


def _surd_fixture(function, call, reference):
    return Fixture(function.__name__, "realistic", "quadratic_surds",
                   _mathforge(function, lambda i: (_zm(_surd_left(i)), _zm(_surd_right(i)))),
                   _sympy(_surd_inputs, call), _same_named_complex, reference, variants=len(_SURDS))


def _complex_algebraic_fixtures():
    def surd_parts_mf(i):
        return _terms_mf(((1, 1, 1), (1, 1, _SURDS[i][0]))), _terms_mf(((1, 1, _SURDS[i][1]),))

    def surd_parts_sp(i):
        import sympy as sp
        return (_terms_sp(((1, 1, 1), (1, 1, _SURDS[i][0]))), _terms_sp(((1, 1, _SURDS[i][1]),)),
                sp.Symbol("t"))

    def gauss_sp(i):
        import sympy as sp
        a, b = _gauss_pair(i, 0)
        return sp.Integer(a), sp.Integer(b)

    return (
        Fixture("complex_algebraic", "micro", "gaussian_integer",
                _mathforge(mf.complex_algebraic, lambda i: _gauss_pair(i, 0)),
                _sympy(gauss_sp, lambda sp, a, b: a + b * sp.I), _same_complex,
                ("used: a + b*I 170 us; others: Add(a, Mul(b, I)) 173 us; medians of 150 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("complex_algebraic", "realistic", "quadratic_surd_parts",
                _mathforge(mf.complex_algebraic, surd_parts_mf), _sympy(surd_parts_sp, _identified_parts),
                _same_named_complex,
                ("used: minimal_polynomial(part, compose=False) of each part 4.16 ms; others: "
                 "minimal_polynomial(part) of each part 5.28 ms; a + b*I (parts not identified) 225 us (excluded: "
                 "parts are not identified); medians of 9 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), variants=len(_SURDS)),
    )


def _square_operand(p: int, q: int):
    """(p + q i) squared as an operand: (p*p - q*q) + 2*p*q i."""
    return (((p * p - q * q, 1, 1),), ((2 * p * q, 1, 1),))


def _micro_square(i: int):
    """Variant i of the micro square roots: z = (p + q i)**2 with p in [1, 20] and q in [-25, 25], so that sqrt(z)
    is p + q i; distinct i below 1,020 give distinct z."""
    return _square_operand(1 + i % 20, (i // 20) % 51 - 25)


_SCALED_SQUARES = ((1234, -987), (4321, 2468), (7777, 1313))


def _complex_sqrt_fixtures():
    return (
        Fixture("complex_sqrt", "micro", "gaussian_square",
                _mathforge(mf.complex_sqrt, lambda i: (_zm(_micro_square(i)),)),
                _sympy(lambda i: (_zs(_micro_square(i)),), lambda sp, a: sp.sqrt(a)), _same_complex,
                ("used: sqrt(a) 1.04 ms; others: sqrt(a).as_real_imag() 1.19 ms; expand_complex(sqrt(a)) 1.49 ms; "
                 "sqrtdenest(sqrt(a)) 1.25 ms; medians of 150 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), variants=_MICRO),
        Fixture("complex_sqrt", "realistic", "three_plus_four_i",
                _mathforge(mf.complex_sqrt, lambda i: (_zm((((3, 1, 1),), ((4, 1, 1),))),)),
                _sympy(lambda i: (_zs((((3, 1, 1),), ((4, 1, 1),))),), lambda sp, a: sp.sqrt(a)), _same_complex,
                ("used: sqrt(a) 663 us; others: sqrt(a).as_real_imag() 789 us; expand_complex(sqrt(a)) 854 us; "
                 "sqrtdenest(sqrt(a)) 752 us; medians of 9 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), v07="complex_sqrt"),
        Fixture("complex_sqrt", "realistic", "scaled_gaussian_square",
                _mathforge(mf.complex_sqrt, lambda i: (_zm(_square_operand(*_SCALED_SQUARES[i])),)),
                _sympy(lambda i: (_zs(_square_operand(*_SCALED_SQUARES[i])),), lambda sp, a: sp.sqrt(a)),
                _same_complex,
                ("used: sqrt(a) 950 us; others: sqrt(a).as_real_imag() 1.02 ms; expand_complex(sqrt(a)) 1.28 ms; "
                 "sqrtdenest(sqrt(a)) 1.02 ms; medians of 9 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), variants=len(_SCALED_SQUARES)),
    )


_POWER_BASES = ((2, 3), (3, -2), (1, 4))
_SCALED_POWER_BASES = ((3, 4), (1, 2), (2, -1))
_UNIT_CIRCLE = (((((1, 2, 1),), ((1, 2, 3),)), 7),   # (1 + i sqrt 3)/2, the 7th power
                ((((1, 2, 2),), ((1, 2, 2),)), 9),   # (sqrt 2 + i sqrt 2)/2, the 9th power
                ((((1, 2, 3),), ((1, 2, 1),)), 5))   # (sqrt 3 + i)/2, the 5th power


def _gauss_base(pair):
    return (((pair[0], 1, 1),), ((pair[1], 1, 1),))


def _complex_pow_fixtures():
    def unit_inputs(i):
        import sympy as sp
        operand, n = _UNIT_CIRCLE[i]
        return _zs(operand), n, sp.Symbol("t")

    return (
        Fixture("complex_pow", "micro", "gaussian_small_power",
                _mathforge(mf.complex_pow, lambda i: (_zm(_gauss(i, 0)), 2 + i % 4)),
                _sympy(lambda i: (_zs(_gauss(i, 0)), 2 + i % 4), lambda sp, a, n: sp.expand(a ** n)),
                _same_complex,
                ("used: expand(a**n) 438 us; others: expand_complex(a**n) 727 us; (a**n).as_real_imag() 489 us "
                 "(excluded: splits a number into its parts, not the expression a user writes for this operation); "
                 "a**n 87 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); QQ_I a**n 22 "
                 "us (excluded: QQ_I domain element: ground arithmetic, not a SymPy-level call); medians of 150 "
                 "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("complex_pow", "realistic", "gaussian_tenth_power",
                _mathforge(mf.complex_pow, lambda i: (_zm(_gauss_base(_POWER_BASES[i])), 10)),
                _sympy(lambda i: (_zs(_gauss_base(_POWER_BASES[i])), 10), lambda sp, a, n: sp.expand(a ** n)),
                _same_complex,
                ("used: expand(a**n) 576 us; others: expand_complex(a**n) 948 us; (a**n).as_real_imag() 591 us "
                 "(excluded: splits a number into its parts, not the expression a user writes for this operation); "
                 "a**n 90 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); medians of 60 "
                 "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_POWER_BASES)),
        Fixture("complex_pow", "realistic", "scaled_hundredth_power",
                _mathforge(mf.complex_pow, lambda i: (_zm(_gauss_base(_SCALED_POWER_BASES[i])), 100)),
                _sympy(lambda i: (_zs(_gauss_base(_SCALED_POWER_BASES[i])), 100),
                       lambda sp, a, n: sp.expand(a ** n)),
                _same_complex,
                ("used: expand(a**n) 599 us; others: expand_complex(a**n) 891 us; (a**n).as_real_imag() 579 us "
                 "(excluded: splits a number into its parts, not the expression a user writes for this operation); "
                 "a**n 94 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); medians of 60 "
                 "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_SCALED_POWER_BASES)),
        Fixture("complex_pow", "realistic", "unit_circle_surds",
                _mathforge(mf.complex_pow, lambda i: (_zm(_UNIT_CIRCLE[i][0]), _UNIT_CIRCLE[i][1])),
                _sympy(unit_inputs, _named_power_composed), _same_named_complex,
                ("used: expand(a**n), as_real_imag, minimal_polynomial 3.82 ms; others: expand(a**n), as_real_imag, "
                 "minimal_polynomial(compose=False) 4.16 ms; first pass, an earlier machine state: "
                 "expand_complex(a**n), as_real_imag, minimal_polynomial(compose=False) 10.87 ms; medians of 31 "
                 "samples of k = 1 calls (the suite's sampling: caches reset before each sample), SymPy 1.14.0 "
                 "(python ground types)"),
                variants=len(_UNIT_CIRCLE)),
    )


def _quadratic_triple(i: int) -> tuple[int, int, int]:
    """Micro variant i: a*x**2 + b*x + c with the complex roots p +- q i (a in [1, 6], p in [-8, 8], q in [1, 10]);
    distinct i below 1,020 give distinct triples."""
    a, p, q = 1 + i % 6, (i // 6) % 17 - 8, 1 + (i // 102) % 10
    return a, -2 * a * p, a * (p * p + q * q)


_IRRATIONAL_QUADRATICS = ((1, 1, 1), (2, 3, 7), (1, -4, 1))


def _quadratic_poly(triple):
    import sympy as sp
    return sp.Poly(list(triple), sp.Symbol("x"))


def _quadratic_named_inputs(i):
    import sympy as sp
    return _quadratic_poly(_IRRATIONAL_QUADRATICS[i]), sp.Symbol("t")


def _quadratic_named_roots(sp, poly, t):
    roots = sp.polys.polyroots.roots_quadratic(poly)
    return [(root, *(_identify(sp, part, t) for part in root.as_real_imag())) for root in roots]


def _quadratic_roots_fixtures():
    return (
        Fixture("quadratic_roots", "micro", "gaussian_roots",
                _mathforge(mf.quadratic_roots, _quadratic_triple),
                _sympy(lambda i: (_quadratic_poly(_quadratic_triple(i)),),
                       lambda sp, poly: sp.polys.polyroots.roots_quadratic(poly)),
                _same_roots,
                ("used: roots_quadratic(P) 1.04 ms; others: roots(P) 2.00 ms; solve(P.as_expr(), x) 18.44 ms; "
                 "solveset(P.as_expr(), x) 14.00 ms; P.all_roots() 8.30 ms (excluded: CRootOf objects, not "
                 "Cartesian closed forms); medians of 150 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), variants=_MICRO),
        Fixture("quadratic_roots", "realistic", "irrational_roots",
                _mathforge(mf.quadratic_roots, lambda i: _IRRATIONAL_QUADRATICS[i]),
                _sympy(_quadratic_named_inputs, _quadratic_named_roots),
                _same_named_roots,
                ("used: roots_quadratic(P), minimal_polynomial(compose=False) 5.29 ms; others: roots(P), "
                 "minimal_polynomial(compose=False) 5.92 ms; solve(P.as_expr(), x), "
                 "minimal_polynomial(compose=False) 29.59 ms; roots_quadratic(P), minimal_polynomial 5.95 ms; "
                 "medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_IRRATIONAL_QUADRATICS)),
    )


COMPLEX_FIXTURES = (
    *_complex_algebraic_fixtures(),
    *_binary_fixtures(mf.complex_add, lambda sp, a, b: sp.Add(a, b),
             ("used: Add(a, b) 30.5 us; others: a + b 30.9 us; medians of 127 samples of k = 74 calls (the suite's "
              "sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
             ("used: a + b 139 us; others: Add(a, b) 141 us; medians of 60 calls, SymPy 1.14.0 (python ground "
              "types), caches cleared before each call"),
             realistic_call=lambda sp, a, b: a + b),
    *_binary_fixtures(mf.complex_sub, lambda sp, a, b: a - b,
             ("used: a - b 43.1 us; others: Add(a, Mul(-1, b)) 43.6 us; medians of 127 samples of k = 54 calls (the "
              "suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
             ("used: a - b 217 us; others: Add(a, Mul(-1, b)) 217 us; medians of 9 calls, SymPy 1.14.0 (python "
              "ground types), caches cleared before each call")),
    *_binary_fixtures(mf.complex_mul, lambda sp, a, b: sp.expand(a * b),
             ("used: expand(a*b) 1.07 ms; others: expand_complex(a*b) 1.62 ms; (a*b).as_real_imag() 1.20 ms "
              "(excluded: splits a number into its parts, not the expression a user writes for this operation); "
              "a*b 128 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); QQ_I a * b 19 "
              "us (excluded: QQ_I domain element: ground arithmetic, not a SymPy-level call); medians of 150 "
              "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
             ("used: expand_complex(a*b) 601 us; others: expand(a*b) 700 us; first pass, an earlier machine state: "
              "a*b 145 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); "
              "(a*b).as_real_imag() 944 us (excluded: splits a number into its parts, not the expression a user "
              "writes for this operation); medians of 31 samples of k = 3 calls (the suite's sampling: caches reset "
              "before each sample), SymPy 1.14.0 (python ground types); excluded candidates keep their first-pass "
              "medians"),
             realistic_call=lambda sp, a, b: sp.expand_complex(a * b)),
    _surd_fixture(mf.complex_mul, _named_product,
            ("used: expand(a*b), as_real_imag, minimal_polynomial(compose=False) 7.96 ms; others: "
             "(a*b).as_real_imag(), minimal_polynomial(compose=False) 7.99 ms; expand(a*b), as_real_imag, "
             "minimal_polynomial 10.32 ms; medians of 9 calls, SymPy 1.14.0 (python ground types), caches "
             "cleared before each call")),
    *_binary_fixtures(mf.complex_div, lambda sp, a, b: sp.expand_complex(a / b),
             ("used: expand_complex(a/b) 2.04 ms; others: (a/b).as_real_imag() 1.62 ms (excluded: splits a "
              "number into its parts, not the expression a user writes for this operation); cancel(a/b) 4.32 "
              "ms; radsimp(a/b) 1.35 ms (excluded: lazy: leaves the operation unevaluated, no Cartesian value); "
              "a/b 621 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); QQ_I a / b 32 "
              "us (excluded: QQ_I domain element: ground arithmetic, not a SymPy-level call); medians of 150 "
              "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
             ("used: expand_complex(a/b) 1.83 ms; others: (a/b).as_real_imag() 1.37 ms (excluded: splits a "
              "number into its parts, not the expression a user writes for this operation); cancel(a/b) 5.66 "
              "ms; radsimp(a/b) 1.91 ms (excluded: lazy: leaves the operation unevaluated, no Cartesian value); "
              "a/b 651 us (excluded: lazy: leaves the operation unevaluated, no Cartesian value); medians of 9 "
              "calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                      divisor=True),
    _surd_fixture(mf.complex_div, _named_quotient,
            ("used: expand_complex(a/b), as_real_imag, minimal_polynomial(compose=False) 12.39 ms; others: "
             "(a/b).as_real_imag(), minimal_polynomial(compose=False) 15.66 ms; expand_complex(a/b), "
             "as_real_imag, minimal_polynomial 15.95 ms; cancel(a/b), as_real_imag, "
             "minimal_polynomial(compose=False) 20.77 ms; medians of 9 calls, SymPy 1.14.0 (python ground "
             "types), caches cleared before each call")),
    *_unary_fixtures(mf.complex_conjugate, lambda sp, a: sp.conjugate(a), _same_complex,
             ("used: conjugate(a) 262 us; others: a.conjugate() 271 us; medians of 150 calls, SymPy 1.14.0 "
              "(python ground types), caches cleared before each call"),
             ("used: conjugate(a) 71.1 us; others: a.conjugate() 71.4 us; medians of 127 samples of k = 3 calls (the "
              "suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)")),
    *_unary_fixtures(mf.complex_real, lambda sp, a: sp.re(a), _same_real,
             ("used: re(a) 718 us; others: a.as_real_imag()[0] 254 us (excluded: splits a number into its "
              "parts, not the expression a user writes for this operation); medians of 150 calls, SymPy 1.14.0 "
              "(python ground types), caches cleared before each call"),
             ("used: re(a) 523 us; others: a.as_real_imag()[0] 251 us (excluded: splits a number into its "
              "parts, not the expression a user writes for this operation); medians of 9 calls, SymPy 1.14.0 "
              "(python ground types), caches cleared before each call")),
    *_unary_fixtures(mf.complex_imag, lambda sp, a: sp.im(a), _same_real,
             ("used: im(a) 685 us; others: a.as_real_imag()[1] 244 us (excluded: splits a number into its "
              "parts, not the expression a user writes for this operation); medians of 150 calls, SymPy 1.14.0 "
              "(python ground types), caches cleared before each call"),
             ("used: im(a) 614 us; others: a.as_real_imag()[1] 270 us (excluded: splits a number into its "
              "parts, not the expression a user writes for this operation); medians of 9 calls, SymPy 1.14.0 "
              "(python ground types), caches cleared before each call")),
    *_unary_fixtures(mf.complex_norm_squared, lambda sp, a: sp.re(a) ** 2 + sp.im(a) ** 2, _same_real,
             ("used: re(a)**2 + im(a)**2 423 us; others: expand(a*conjugate(a)) 497 us; Abs(a)**2 1.66 ms; first "
              "pass, an earlier machine state: as_real_imag, then x*x + y*y 224 us (excluded: a hand-written formula;"
              " Abs(a)**2 is the expression a user writes); medians of 31 samples of k = 2 calls (the suite's "
              "sampling: caches reset before each sample), SymPy 1.14.0 (python ground types); excluded candidates "
              "keep their first-pass medians"),
             ("used: re(a)**2 + im(a)**2 544 us; others: expand(a*conjugate(a)) 632 us; Abs(a)**2 2.48 ms; first "
              "pass, an earlier machine state: as_real_imag, then x*x + y*y 242 us (excluded: a hand-written formula;"
              " Abs(a)**2 is the expression a user writes); medians of 31 samples of k = 1 calls (the suite's "
              "sampling: caches reset before each sample), SymPy 1.14.0 (python ground types); excluded candidates "
              "keep their first-pass medians")),
    *_complex_sqrt_fixtures(),
    *_complex_pow_fixtures(),
    *_quadratic_roots_fixtures(),
)


# ---------------------------------------------------------------------------------------------------------------
# Polynomials
# ---------------------------------------------------------------------------------------------------------------
# A polynomial description is its tuple of integer coefficients, constant term first.

def _coeffs(tag: int, i: int, degree: int, bound: int = 9) -> tuple[int, ...]:
    """Integer coefficients of a polynomial of the given degree for variant i; the leading coefficient is nonzero."""
    lower = [_draw(-bound, bound, tag, i, k) for k in range(degree)]
    lead = _draw(1, bound, tag, i, degree)
    return (*lower, -lead if _draw(0, 1, tag, i, degree, 1) else lead)


def _times(*polynomials: tuple[int, ...]) -> tuple[int, ...]:
    """The product of integer polynomials."""
    result = (1,)
    for q in polynomials:
        out = [0] * (len(result) + len(q) - 1)
        for a, u in enumerate(result):
            for b, v in enumerate(q):
                out[a + b] += u * v
        result = tuple(out)
    return result


def _power(p: tuple[int, ...], n: int) -> tuple[int, ...]:
    return _times(*([p] * n))


def _scaled(c: int, p: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(c * u for u in p)


def _xs():
    """The SymPy symbol x."""
    import sympy as sp
    return sp.Symbol("x")


@functools.cache
def _pm(coefficients: tuple[int, ...]):
    """The MathForge Polynomial in X of integer coefficients, built once."""
    return mf.Polynomial(coefficients, X)


def _ps(coefficients: tuple[int, ...], domain: str = "QQ"):
    """The SymPy Poly in x of integer coefficients, over QQ unless said otherwise."""
    import sympy as sp
    return sp.Poly(list(reversed(coefficients)), sp.Symbol("x"), domain=domain)


def _sp_poly(value):
    """A MathForge Polynomial as a SymPy Poly over QQ."""
    import sympy as sp
    return sp.Poly([sp.Rational(c.numerator, c.denominator) for c in reversed(value.coefficients)],
                   sp.Symbol("x"), domain="QQ")


def _same_poly(mf_value, sympy_value) -> bool:
    """A MathForge Polynomial and a SymPy Poly, PurePoly or expression in x are the same polynomial over QQ."""
    import sympy as sp
    if type(mf_value) is not mf.Polynomial:
        return False
    other = sympy_value if isinstance(sympy_value, sp.Poly) else sp.Poly(sympy_value, sp.Symbol("x"))
    return _sp_poly(mf_value).all_coeffs() == other.set_domain("QQ").all_coeffs()


def _same_pair(mf_value, sympy_value) -> bool:
    """(quotient, remainder) against SymPy's (quotient, remainder)."""
    return (type(mf_value) is tuple and len(mf_value) == len(sympy_value) == 2
            and all(_same_poly(a, b) for a, b in zip(mf_value, sympy_value)))


def _same_gcdex(mf_value, sympy_value) -> bool:
    """(gcd, s, t) against SymPy's gcdex (s, t, gcd): the cofactors of minimal degree are unique."""
    gcd, s, t = mf_value
    s2, t2, gcd2 = sympy_value
    return _same_poly(gcd, gcd2) and _same_poly(s, s2) and _same_poly(t, t2)


def _factor_pairs(sympy_value):
    """SymPy's (coefficient, [(factor, multiplicity), ...]) with monic factors: the leading coefficients of its
    primitive integer factors move into the coefficient. A factorization is unique up to such units."""
    import sympy as sp
    coefficient, factors = sympy_value
    coefficient, pairs = sp.Rational(coefficient), []
    for factor, multiplicity in factors:
        factor = sp.Poly(factor, sp.Symbol("x"), domain="QQ")
        coefficient *= sp.Rational(factor.LC()) ** multiplicity
        pairs.append((factor.monic().all_coeffs(), multiplicity))
    return coefficient, sorted(pairs, key=lambda pair: (pair[1], pair[0]))


def _same_factors(mf_value, sympy_value) -> bool:
    """A PolynomialFactorization or SquareFreeDecomposition against SymPy's (coefficient, [(factor, mult)])."""
    import sympy as sp
    coefficient, found = _factor_pairs(sympy_value)
    if sp.Rational(mf_value.coefficient.numerator, mf_value.coefficient.denominator) != coefficient:
        return False
    own = sorted(((_sp_poly(f.polynomial).all_coeffs(), f.multiplicity) for f in mf_value.factors),
                 key=lambda pair: (pair[1], pair[0]))
    return own == found


def _same_scalar(mf_value, sympy_value) -> bool:
    return type(mf_value) is mf.Rational and _same_number(_sp_real(mf_value), sympy_value)


def _sp_expr(value, symbols):
    """A MathForge expression as a SymPy expression; ``symbols`` maps MathForge Symbols to SymPy ones."""
    import sympy as sp
    kind = type(value)
    if kind is mf.Rational:
        return sp.Rational(value.numerator, value.denominator)
    if kind is mf.Symbol:
        return symbols[value]
    if kind is mf.Add:
        return sp.Add(*(_sp_expr(v, symbols) for v in value.terms))
    if kind is mf.Mul:
        return sp.Mul(*(_sp_expr(v, symbols) for v in value.factors))
    if kind is mf.Pow:
        return _sp_expr(value.base, symbols) ** value.exponent
    if kind is mf.Sqrt:
        return sp.sqrt(_sp_expr(value.radicand, symbols))
    if kind is mf.RealAlgebraicRoot:
        return _sp_real(value)
    raise TypeError(f"not a MathForge expression: {kind.__name__}")


def _sp_symbols():
    import sympy as sp
    return {X: sp.Symbol("x"), Y: sp.Symbol("y"), Z: sp.Symbol("z")}


def _same_expression(mf_value, sympy_value) -> bool:
    """A MathForge expression and a SymPy expression (or Poly) are equal polynomials/expressions."""
    import sympy as sp
    if isinstance(sympy_value, sp.Poly):
        sympy_value = sympy_value.as_expr()
    return sp.expand(_sp_expr(mf_value, _sp_symbols()) - sympy_value) == 0


# --- expand and expand_multivariate -----------------------------------------------------------------------------

def _binomial_pair(i: int):
    """Micro variant i of expand: (a*x + b)*(c*x + d), a and c in [1, 6], b and d in [-9, 9]; distinct for
    i < 12,996."""
    a, b, c, d = _spread(i, 6, 19, 6, 19)
    return a + 1, b - 9, c + 1, d - 9


def _expand_micro_mf(i):
    a, b, c, d = _binomial_pair(i)
    return (a * X + b) * (c * X + d), X


def _expand_micro_sp(i):
    import sympy as sp
    a, b, c, d = _binomial_pair(i)
    x = sp.Symbol("x")
    return ((a * x + b) * (c * x + d),)


_EXPAND_REALISTIC = ((1, 2), (3, 1), (2, 5))  # (x + a)**10 * (x - b)**5
_EXPAND_SCALED = ((2, 1), (1, 3), (3, 2))  # (x**3 + a*x + b)**15


def _expand_realistic_mf(i):
    a, b = _EXPAND_REALISTIC[i]
    return (X + a) ** 10 * (X - b) ** 5, X


def _expand_realistic_sp(i):
    import sympy as sp
    a, b = _EXPAND_REALISTIC[i]
    x = sp.Symbol("x")
    return ((x + a) ** 10 * (x - b) ** 5,)


def _expand_scaled_mf(i):
    a, b = _EXPAND_SCALED[i]
    return (X ** 3 + a * X + b) ** 15, X


def _expand_scaled_sp(i):
    import sympy as sp
    a, b = _EXPAND_SCALED[i]
    x = sp.Symbol("x")
    return ((x ** 3 + a * x + b) ** 15,)


def _agree_expression(mf_value, sympy_value):
    return _same_expression(mf_value, sympy_value)


def _expand_fixtures():
    call = lambda sp, e: sp.expand(e)
    without_hints = lambda sp, e: sp.expand(e, power_base=False, power_exp=False, log=False, basic=False)
    return (
        Fixture("expand", "micro", "two_binomials", _mathforge(mf.expand, _expand_micro_mf),
                _sympy(_expand_micro_sp, without_hints), _agree_expression,
                ("used: expand(e, power_base=False, power_exp=False, log=False, basic=False) 420 us; others: "
                 "e.expand() 449 us; expand(e) 453 us; Poly(e).as_expr() 539 us; first pass, an earlier machine "
                 "state: Poly(e) 972 us (excluded: a Poly, not an expression); medians of 127 samples of k = 5 calls "
                 "(the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"), variants=_MICRO),
        Fixture("expand", "realistic", "binomial_powers", _mathforge(mf.expand, _expand_realistic_mf),
                _sympy(_expand_realistic_sp, call), _agree_expression,
                ("used: expand(e) 7.39 ms; others: e.expand() 8.15 ms; expand(e, power_base=False, "
                 "power_exp=False, log=False, basic=False) 8.21 ms; Poly(e) 7.50 ms; Poly(e).as_expr() 8.08 ms; "
                 "medians of 60 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_EXPAND_REALISTIC)),
        Fixture("expand", "realistic", "scaled_cubic_power", _mathforge(mf.expand, _expand_scaled_mf),
                _sympy(_expand_scaled_sp, call), _agree_expression,
                ("used: expand(e) 13.46 ms; others: Poly(e).as_expr() 14.35 ms; first pass, an earlier machine state:"
                 " expand(e, power_base=False, power_exp=False, log=False, basic=False) 16.20 ms; e.expand() 16.24 "
                 "ms; Poly(e) 15.16 ms (excluded: a Poly, not an expression); medians of 31 samples of k = 1 calls "
                 "(the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=len(_EXPAND_SCALED)),
    )


def _bilinear_pair(i: int):
    """Micro variant i of expand_multivariate: (a*x + b*y + c)*(d*x + e*y + f), a, b, d, e in [1, 5] and c, f in
    [-4, 4]; distinct for i < 50,625."""
    a, b, c, d, e, f = _spread(i, 5, 5, 9, 5, 5, 9)
    return a + 1, b + 1, c - 4, d + 1, e + 1, f - 4


def _multivariate_micro_mf(i):
    a, b, c, d, e, f = _bilinear_pair(i)
    return (a * X + b * Y + c) * (d * X + e * Y + f), (X, Y)


def _multivariate_micro_sp(i):
    import sympy as sp
    a, b, c, d, e, f = _bilinear_pair(i)
    x, y = sp.symbols("x y")
    return ((a * x + b * y + c) * (d * x + e * y + f),)


_MULTIVARIATE_REALISTIC = ((1, 1, 1), (1, 2, 3), (2, -1, 1))  # (a*x + b*y + c*z)**8
_MULTIVARIATE_SCALED = ((2, 3), (3, -1), (-2, 5))  # (x + a*y + b*z)**12


def _multivariate_realistic_mf(i):
    a, b, c = _MULTIVARIATE_REALISTIC[i]
    return (a * X + b * Y + c * Z) ** 8, (X, Y, Z)


def _multivariate_realistic_sp(i):
    import sympy as sp
    a, b, c = _MULTIVARIATE_REALISTIC[i]
    x, y, z = sp.symbols("x y z")
    return ((a * x + b * y + c * z) ** 8,)


def _multivariate_scaled_mf(i):
    a, b = _MULTIVARIATE_SCALED[i]
    return (X + a * Y + b * Z) ** 12, (X, Y, Z)


def _multivariate_scaled_sp(i):
    import sympy as sp
    a, b = _MULTIVARIATE_SCALED[i]
    x, y, z = sp.symbols("x y z")
    return ((x + a * y + b * z) ** 12,)


def _expand_multivariate_fixtures():
    # expand with its four rewriting hints off: the faster call of the same expansion on every fixture of the operation
    call = lambda sp, e: sp.expand(e, power_base=False, power_exp=False, log=False, basic=False)
    return (
        Fixture("expand_multivariate", "micro", "two_bilinear_factors",
                _mathforge(mf.expand_multivariate, _multivariate_micro_mf), _sympy(_multivariate_micro_sp, call),
                _agree_expression,
                ("used: expand(e, power_base=False, power_exp=False, log=False, basic=False) 678 us; others: "
                 "expand(e) 740 us; first pass, an earlier machine state: e.expand() 1.96 ms; Poly(e).as_expr() 2.28 "
                 "ms; Poly(e) 2.12 ms (excluded: a Poly, not an expression); medians of 31 samples of k = 4 calls "
                 "(the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=_MICRO),
        Fixture("expand_multivariate", "realistic", "trinomial_eighth_power",
                _mathforge(mf.expand_multivariate, _multivariate_realistic_mf),
                _sympy(_multivariate_realistic_sp, call), _agree_expression,
                ("used: expand(e, power_base=False, power_exp=False, log=False, basic=False) 4.32 ms; others: "
                 "expand(e) 4.96 ms; first pass, an earlier machine state: e.expand() 12.03 ms; Poly(e).as_expr() "
                 "15.31 ms; Poly(e) 12.52 ms (excluded: a Poly, not an expression); medians of 31 samples of k = 1 "
                 "calls (the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=len(_MULTIVARIATE_REALISTIC)),
        Fixture("expand_multivariate", "realistic", "scaled_twelfth_power",
                _mathforge(mf.expand_multivariate, _multivariate_scaled_mf), _sympy(_multivariate_scaled_sp, call),
                _agree_expression,
                ("used: expand(e, power_base=False, power_exp=False, log=False, basic=False) 12.66 ms; others: "
                 "expand(e) 14.35 ms; first pass, an earlier machine state: e.expand() 24.74 ms; Poly(e).as_expr() "
                 "31.78 ms; Poly(e) 26.59 ms (excluded: a Poly, not an expression); medians of 31 samples of k = 1 "
                 "calls (the suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=len(_MULTIVARIATE_SCALED)),
    )


# --- poly_divmod, poly_gcd, poly_xgcd, square_free ---------------------------------------------------------------

def _divmod_pair(shape: tuple[int, int], tag: int):
    """Variant builder of a division: a dividend and a divisor of the given degrees."""
    return lambda i: (_coeffs(tag, i, shape[0]), _coeffs(tag + 1, i, shape[1]))


_DIVMOD_MICRO = _divmod_pair((4, 2), 11)
_DIVMOD_REALISTIC = _divmod_pair((12, 5), 13)
_DIVMOD_SCALED = _divmod_pair((40, 15), 15)


def _pair_mf(pair_of):
    return lambda i: tuple(_pm(c) for c in pair_of(i))


def _pair_sp(pair_of, domain="QQ"):
    return lambda i: tuple(_ps(c, domain) for c in pair_of(i))


def _poly_divmod_fixtures():
    call = lambda sp, f, g: f.div(g)
    return (
        Fixture("poly_divmod", "micro", "quartic_by_quadratic", _mathforge(mf.poly_divmod, _pair_mf(_DIVMOD_MICRO)),
                _sympy(_pair_sp(_DIVMOD_MICRO), call), _same_pair,
                ("used: f.div(g) 18.2 us; others: divmod(f, g) 18.2 us; first pass, an earlier machine state: "
                 "(f.quo(g), f.rem(g)) 89 us; div(f, g) 96 us; medians of 127 samples of k = 89 calls (the suite's "
                 "sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=_MICRO),
        Fixture("poly_divmod", "realistic", "degree_12_by_5",
                _mathforge(mf.poly_divmod, _pair_mf(_DIVMOD_REALISTIC)), _sympy(_pair_sp(_DIVMOD_REALISTIC), call),
                _same_pair,
                ("used: f.div(g) 82.9 us; others: divmod(f, g) 83.7 us; first pass, an earlier machine state: div(f, "
                 "g) 164 us; (f.quo(g), f.rem(g)) 201 us; medians of 127 samples of k = 3 calls (the suite's "
                 "sampling: caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=3),
        Fixture("poly_divmod", "realistic", "scaled_degree_40_by_15",
                _mathforge(mf.poly_divmod, _pair_mf(_DIVMOD_SCALED)), _sympy(_pair_sp(_DIVMOD_SCALED), call),
                _same_pair,
                ("used: f.div(g) 586 us; others: divmod(f, g) 588 us; first pass, an earlier machine state: div(f, g)"
                 " 703 us; (f.quo(g), f.rem(g)) 1.26 ms; medians of 127 samples of k = 3 calls (the suite's sampling:"
                 " caches reset before each sample), SymPy 1.14.0 (python ground types)"),
                variants=3),
    )


def _common_factor_micro(i: int):
    """Micro variant i of the gcd family: c*(x + a)*(x + b) and d*(x + a)*(x + e) with a < b < e (so the gcd is
    x + a), c and d in [1, 5]; distinct for i < 30,400."""
    a, db, de, c, d = _spread(i, 19, 8, 8, 5, 5)
    a -= 9
    b = a + 1 + db
    e = b + 1 + de
    return (_scaled(c + 1, _times((a, 1), (b, 1))), _scaled(d + 1, _times((a, 1), (e, 1))))


def _common_factor(tag: int, common: int, first: int, second: int):
    """Variant builder of two polynomials h*u and h*v with random h, u, v of the given degrees."""
    def pair_of(i):
        h = _coeffs(tag, i, common)
        return _times(h, _coeffs(tag + 1, i, first)), _times(h, _coeffs(tag + 2, i, second))
    return pair_of


_GCD_REALISTIC = _common_factor(21, 5, 6, 4)
_GCD_SCALED = _common_factor(24, 12, 14, 12)
_XGCD_REALISTIC = _common_factor(27, 3, 5, 4)
_XGCD_SCALED = _common_factor(30, 8, 12, 10)


def _poly_gcd_fixtures():
    call = lambda sp, f, g: f.gcd(g)
    return (
        Fixture("poly_gcd", "micro", "common_linear_factor", _mathforge(mf.poly_gcd, _pair_mf(_common_factor_micro)),
                _sympy(_pair_sp(_common_factor_micro), call), _same_poly,
                ("used: f.gcd(g) 112 us; others: gcd(f, g) 191 us; f.cofactors(g)[0] 120 us; medians of 150 calls, "
                 "SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("poly_gcd", "realistic", "common_factor_degree_5",
                _mathforge(mf.poly_gcd, _pair_mf(_GCD_REALISTIC)), _sympy(_pair_sp(_GCD_REALISTIC), call),
                _same_poly,
                ("used: f.gcd(g) 124 us; others: f.cofactors(g)[0] 124 us; first pass, an earlier machine state: "
                 "gcd(f, g) 262 us; medians of 127 samples of k = 3 calls (the suite's sampling: caches reset before "
                 "each sample), SymPy 1.14.0 (python ground types)"), variants=3),
        Fixture("poly_gcd", "realistic", "scaled_common_factor_degree_12",
                _mathforge(mf.poly_gcd, _pair_mf(_GCD_SCALED)), _sympy(_pair_sp(_GCD_SCALED), call), _same_poly,
                ("used: f.gcd(g) 280 us; others: gcd(f, g) 356 us; f.cofactors(g)[0] 312 us; medians of 9 calls, "
                 "SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=3),
    )


def _poly_xgcd_fixtures():
    call = lambda sp, f, g: f.gcdex(g)
    return (
        Fixture("poly_xgcd", "micro", "common_linear_factor",
                _mathforge(mf.poly_xgcd, _pair_mf(_common_factor_micro)),
                _sympy(_pair_sp(_common_factor_micro), call), _same_gcdex,
                ("used: f.gcdex(g) 89 us; others: gcdex(f, g) 127 us; medians of 150 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("poly_xgcd", "realistic", "common_factor_degree_3",
                _mathforge(mf.poly_xgcd, _pair_mf(_XGCD_REALISTIC)), _sympy(_pair_sp(_XGCD_REALISTIC), call),
                _same_gcdex,
                ("used: f.gcdex(g) 324 us; others: gcdex(f, g) 357 us; medians of 9 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=3),
        Fixture("poly_xgcd", "realistic", "scaled_common_factor_degree_8",
                _mathforge(mf.poly_xgcd, _pair_mf(_XGCD_SCALED)), _sympy(_pair_sp(_XGCD_SCALED), call),
                _same_gcdex,
                ("used: f.gcdex(g) 2.20 ms; others: gcdex(f, g) 2.20 ms; medians of 9 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=3),
    )


def _square_free_micro(i: int):
    """Micro variant i: c*(x + a)**2*(x + b) with a < b and c in [1, 6]; distinct for i < 1,026."""
    a, db, c = _spread(i, 19, 9, 6)
    a -= 9
    return _scaled(c + 1, _times(_power((a, 1), 2), (a + 1 + db, 1)))


def _powers_of_random(tag: int, i: int, degrees: tuple[int, ...]) -> tuple[int, ...]:
    """A * B**2 * C**3 * ... of random polynomials A, B, C, ... of the given degrees."""
    return _times(*(_power(_coeffs(tag + k, i, degree), k + 1) for k, degree in enumerate(degrees)))


def _poly_one(function):
    return lambda i: (_pm(function(i)),)


def _square_free_fixtures():
    call = lambda sp, f: f.sqf_list()
    realistic = lambda i: _powers_of_random(61, i, (3, 2, 2))
    scaled = lambda i: _powers_of_random(65, i, (6, 5, 4, 3))
    return (
        Fixture("square_free", "micro", "repeated_linear_factor",
                _mathforge(mf.square_free, _poly_one(_square_free_micro)),
                _sympy(lambda i: (_ps(_square_free_micro(i)),), call), _same_factors,
                ("used: f.sqf_list() 180 us; others: sqf_list(f) 275 us; medians of 150 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("square_free", "realistic", "powers_one_to_three",
                _mathforge(mf.square_free, _poly_one(realistic)), _sympy(lambda i: (_ps(realistic(i)),), call),
                _same_factors,
                ("used: f.sqf_list() 431 us; others: sqf_list(f) 541 us; medians of 9 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=3),
        Fixture("square_free", "realistic", "scaled_powers_one_to_four",
                _mathforge(mf.square_free, _poly_one(scaled)), _sympy(lambda i: (_ps(scaled(i)),), call),
                _same_factors,
                ("used: f.sqf_list() 1.08 ms; others: sqf_list(f) 1.13 ms; medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=3),
    )


# --- real_roots, compare_real ------------------------------------------------------------------------------------

def _same_root_records(mf_value, sympy_value) -> bool:
    """A tuple of RootRecord against SymPy roots: [(root, multiplicity), ...] or a list of roots with repeats."""
    found = []
    for item in sympy_value:
        if isinstance(item, tuple):
            found.append(item)
        elif found and _same_number(found[-1][0], item):
            found[-1] = (found[-1][0], found[-1][1] + 1)
        else:
            found.append((item, 1))
    return (len(found) == len(mf_value)
            and all(record.multiplicity == multiplicity and _same_number(_sp_real(record.root), root)
                    for record, (root, multiplicity) in zip(mf_value, found)))


def _rational_roots(i: int):
    """Micro variant i of real_roots: (x - a)*(x - b)*(x - c) with a < b < c in [-9, 17]; distinct for i < 1,216."""
    a, db, dc = _spread(i, 19, 8, 8)
    a -= 9
    b = a + 1 + db
    return a, b, b + 1 + dc


def _real_roots_micro_mf(i):
    a, b, c = _rational_roots(i)
    return (X - a) * (X - b) * (X - c), X


def _real_roots_micro_sp(i):
    import sympy as sp
    a, b, c = _rational_roots(i)
    x = sp.Symbol("x")
    return ((x - a) * (x - b) * (x - c),)


_DEGREE_TEN = ((3, 2), (4, 1), (2, 3))  # x**10 - a*x**5 + x - b


def _real_roots_ten_mf(i):
    a, b = _DEGREE_TEN[i]
    return X ** 10 - a * X ** 5 + X - b, X


def _real_roots_ten_sp(i):
    import sympy as sp
    a, b = _DEGREE_TEN[i]
    x = sp.Symbol("x")
    return (x ** 10 - a * x ** 5 + x - b,)


def _real_roots_fixtures():
    call = lambda sp, e: sp.real_roots(e, multiple=False)
    return (
        Fixture("real_roots", "micro", "three_rational_roots", _mathforge(mf.real_roots, _real_roots_micro_mf),
                _sympy(_real_roots_micro_sp, call), _same_root_records,
                ("used: real_roots(e, multiple=False) 1.95 ms; others: Poly(e).real_roots() 2.01 ms; real_roots(e, "
                 "radicals=False, multiple=False) 2.06 ms; Poly(e).intervals() 1.36 ms (excluded: isolating "
                 "intervals, not the roots); medians of 150 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"),
                variants=_MICRO),
        Fixture("real_roots", "realistic", "quintic_x5_minus_x_minus_1",
                _mathforge(mf.real_roots, lambda i: (X ** 5 - X - 1, X)),
                _sympy(lambda i: (_xs() ** 5 - _xs() - 1,), call),
                _same_root_records,
                ("used: real_roots(e, multiple=False) 1.08 ms; others: Poly(e).real_roots() 1.10 ms; real_roots(e, "
                 "radicals=False, multiple=False) 1.09 ms; Poly(e).intervals() 597 us (excluded: isolating "
                 "intervals, not the roots); medians of 60 calls, SymPy 1.14.0 (python ground types), caches "
                 "cleared before each call"), v07="quintic_roots"),
        Fixture("real_roots", "realistic", "scaled_degree_ten", _mathforge(mf.real_roots, _real_roots_ten_mf),
                _sympy(_real_roots_ten_sp, lambda sp, e: sp.real_roots(e, radicals=False, multiple=False)),
                _same_root_records,
                ("used: real_roots(e, radicals=False, multiple=False) 1.49 ms; others: real_roots(e, multiple=False) "
                 "1.49 ms; first pass, an earlier machine state: Poly(e).real_roots() 2.81 ms; Poly(e).intervals() "
                 "868 us (excluded: isolating intervals, not the roots); medians of 127 samples of k = 2 calls (the "
                 "suite's sampling: caches reset before each sample), SymPy 1.14.0 (python ground types); excluded "
                 "candidates keep their first-pass medians"),
                variants=len(_DEGREE_TEN)),
    )


_NON_SQUARES = tuple(n for n in range(2, 90) if int(n ** 0.5) ** 2 != n)[:40]


def _sqrt_comparison(i: int):
    """Micro variant i of compare_real: p/q against sqrt(r), p in [1, 60], q in [1, 20], r a non-square below 90;
    distinct for i < 48,000. The two numbers are never equal."""
    p, q, r = _spread(i, 60, 20, 40)
    return p + 1, q + 1, _NON_SQUARES[r]


_SURD_COMPARISONS = (((2, 3), (1, 5)),    # sqrt 2 + sqrt 3  against  1 + sqrt 5
                     ((5, 7), (2, 11)),   # sqrt 5 + sqrt 7  against  2 + sqrt 11
                     ((7, 13), (2, 17)))  # sqrt 7 + sqrt 13 against  2 + sqrt 17
_THREE_SURD_COMPARISONS = (((2, 3, 5), (1, 7, 6)), ((2, 5, 7), (3, 6, 10)), ((3, 5, 11), (2, 13, 7)))


def _sum_of_roots_mf(radicands, offset=0):
    value = mf.Rational(offset)
    for r in radicands:
        value = value + mf.sqrt(r)
    return value


def _compare_pair_mf(pair):
    (left, right) = pair
    return _sum_of_roots_mf(left), _sum_of_roots_mf(right[1:], right[0])


def _compare_pair_sp(pair):
    import sympy as sp
    (left, right) = pair
    return sum(sp.sqrt(r) for r in left), right[0] + sum(sp.sqrt(r) for r in right[1:])


def _same_sign(mf_value, sympy_value) -> bool:
    try:
        return type(mf_value) is int and int(sympy_value) == mf_value
    except TypeError:
        return False


def _compare_real_fixtures():
    call = lambda sp, left, right: sp.sign(left - right)

    def micro_mf(i):
        p, q, r = _sqrt_comparison(i)
        return mf.Rational(p, q), mf.sqrt(r)

    def micro_sp(i):
        import sympy as sp
        p, q, r = _sqrt_comparison(i)
        return sp.Rational(p, q), sp.sqrt(r)

    return (
        Fixture("compare_real", "micro", "rational_against_square_root", _mathforge(mf.compare_real, micro_mf),
                _sympy(micro_sp, call), _same_sign,
                ("used: sign(l - r) 899 us; others: int(bool(l > r)) - int(bool(l < r)) 526 us (excluded: two "
                 "comparisons combined by hand; sign(l - r) is the expression a user writes); (l - r).is_positive "
                 "/ is_zero 780 us (excluded: three-valued flag, not -1/0/1); medians of 150 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("compare_real", "realistic", "sums_of_two_square_roots",
                _mathforge(mf.compare_real, lambda i: _compare_pair_mf(_SURD_COMPARISONS[i])),
                _sympy(lambda i: _compare_pair_sp(_SURD_COMPARISONS[i]), call), _same_sign,
                ("used: sign(l - r) 1.72 ms; others: int(bool(l > r)) - int(bool(l < r)) 1.22 ms (excluded: two "
                 "comparisons combined by hand; sign(l - r) is the expression a user writes); (l - r).is_positive "
                 "/ is_zero 1.39 ms (excluded: three-valued flag, not -1/0/1); medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=len(_SURD_COMPARISONS)),
        Fixture("compare_real", "realistic", "scaled_sums_of_three_square_roots",
                _mathforge(mf.compare_real, lambda i: _compare_pair_mf(_THREE_SURD_COMPARISONS[i])),
                _sympy(lambda i: _compare_pair_sp(_THREE_SURD_COMPARISONS[i]), call), _same_sign,
                ("used: sign(l - r) 1.72 ms; others: int(bool(l > r)) - int(bool(l < r)) 1.64 ms (excluded: two "
                 "comparisons combined by hand; sign(l - r) is the expression a user writes); (l - r).is_positive "
                 "/ is_zero 1.64 ms (excluded: three-valued flag, not -1/0/1); medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=len(_THREE_SURD_COMPARISONS)),
    )


# --- rational_function, factor, resultant, discriminant, apart --------------------------------------------------

def _same_rational_function(mf_value, sympy_value) -> bool:
    """A RationalFunction against SymPy's reduced fraction: a (numerator, denominator) pair or one expression."""
    import sympy as sp
    if type(mf_value) is not mf.RationalFunction:
        return False
    x = sp.Symbol("x")
    expected = _sp_poly(mf_value.numerator).as_expr() / _sp_poly(mf_value.denominator).as_expr()
    if isinstance(sympy_value, tuple):
        p, q = (sp.Poly(part, x) for part in sympy_value)
        if p.gcd(q).degree() > 0:
            return False  # not in lowest terms
        value = p.as_expr() / q.as_expr()
    else:
        value = sympy_value
    return sp.cancel(expected - value) == 0


def _rational_function_mf(pair_of):
    def build(i):
        numerator, denominator = pair_of(i)
        return _pm(numerator), _pm(denominator)
    return lambda i: (mf.rational_function, build(i), {"variable": X})


def _rational_function_fixtures():
    call = lambda sp, p, q: p.cancel(q, include=True)
    realistic = _common_factor(71, 3, 4, 3)
    scaled = _common_factor(74, 8, 10, 8)
    return (
        Fixture("rational_function", "micro", "common_linear_factor", _rational_function_mf(_common_factor_micro),
                _sympy(_pair_sp(_common_factor_micro), call), _same_rational_function,
                ("used: P.cancel(Q, include=True) 95 us; others: cancel((P, Q)) n/a (raised on this input); "
                 "cancel(P.as_expr()/Q.as_expr()) 3.10 ms; medians of 150 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=_MICRO),
        Fixture("rational_function", "realistic", "common_factor_degree_3", _rational_function_mf(realistic),
                _sympy(_pair_sp(realistic), call), _same_rational_function,
                ("used: P.cancel(Q, include=True) 133 us; others: cancel((P, Q)) n/a (raised on this input); "
                 "cancel(P.as_expr()/Q.as_expr()) 6.48 ms; medians of 9 calls, SymPy 1.14.0 (python ground types), "
                 "caches cleared before each call"), variants=3),
        Fixture("rational_function", "realistic", "scaled_common_factor_degree_8", _rational_function_mf(scaled),
                _sympy(_pair_sp(scaled), call), _same_rational_function,
                ("used: P.cancel(Q, include=True) 209 us; others: cancel((P, Q)) n/a (raised on this input); "
                 "cancel(P.as_expr()/Q.as_expr()) 12.05 ms; medians of 9 calls, SymPy 1.14.0 (python ground "
                 "types), caches cleared before each call"), variants=3),
    )


def _factor_fixtures():
    call = lambda sp, f: f.factor_list()
    micro = lambda i: _coeffs(41, i, 4)
    realistic = lambda i: _times(*(_coeffs(42 + k, i, degree) for k, degree in enumerate((1, 2, 3, 4))))
    scaled = lambda i: _times(*(_coeffs(46 + k, i, degree) for k, degree in enumerate((2, 3, 4, 5, 6))))
    return (
        Fixture("factor", "micro", "random_quartic", _mathforge(mf.factor, _poly_one(micro)),
                _sympy(lambda i: (_ps(micro(i)),), call), _same_factors,
                ("used: f.factor_list() 329 us; others: factor_list(f) 382 us; factor(f) [a product expression] "
                 "1.82 ms (excluded: returns a product expression, not the list of factors); medians of 150 calls, "
                 "SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("factor", "realistic", "product_of_four_factors", _mathforge(mf.factor, _poly_one(realistic)),
                _sympy(lambda i: (_ps(realistic(i)),), call), _same_factors,
                ("used: f.factor_list() 1.75 ms; others: factor_list(f) 1.82 ms; factor(f) [a product expression] "
                 "8.95 ms (excluded: returns a product expression, not the list of factors); medians of 9 calls, "
                 "SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=3),
        Fixture("factor", "realistic", "scaled_product_of_five_factors", _mathforge(mf.factor, _poly_one(scaled)),
                _sympy(lambda i: (_ps(scaled(i)),), call), _same_factors,
                ("used: f.factor_list() 5.14 ms; others: factor_list(f) 5.50 ms; factor(f) [a product expression] "
                 "23.91 ms (excluded: returns a product expression, not the list of factors); medians of 9 calls, "
                 "SymPy 1.14.0 (python ground types), caches cleared before each call"), variants=3),
    )


def _resultant_fixtures():
    call = lambda sp, f, g: f.resultant(g)
    micro, realistic, scaled = (_divmod_pair((3, 2), 51), _divmod_pair((8, 6), 53), _divmod_pair((20, 15), 55))
    return (
        Fixture("resultant", "micro", "cubic_and_quadratic", _mathforge(mf.resultant, _pair_mf(micro)),
                _sympy(_pair_sp(micro, "ZZ"), call), _same_scalar,
                ("used: f.resultant(g) 44 us; others: resultant(f, g) 72 us; medians of 150 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("resultant", "realistic", "degree_8_and_6", _mathforge(mf.resultant, _pair_mf(realistic)),
                _sympy(_pair_sp(realistic, "ZZ"), call), _same_scalar,
                ("used: f.resultant(g) 66 us; others: resultant(f, g) 88 us; medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"),
                variants=3),
        Fixture("resultant", "realistic", "scaled_degree_20_and_15", _mathforge(mf.resultant, _pair_mf(scaled)),
                _sympy(_pair_sp(scaled, "ZZ"), call), _same_scalar,
                ("used: f.resultant(g) 157 us; others: resultant(f, g) 176 us; medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=3),
    )


def _discriminant_fixtures():
    call = lambda sp, f: f.discriminant()
    micro, realistic, scaled = (lambda i: _coeffs(57, i, 3), lambda i: _coeffs(58, i, 8), lambda i: _coeffs(59, i, 20))
    return (
        Fixture("discriminant", "micro", "random_cubic", _mathforge(mf.discriminant, _poly_one(micro)),
                _sympy(lambda i: (_ps(micro(i), "ZZ"),), call), _same_scalar,
                ("used: f.discriminant() 42 us; others: discriminant(f) 72 us; medians of 150 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("discriminant", "realistic", "random_degree_8", _mathforge(mf.discriminant, _poly_one(realistic)),
                _sympy(lambda i: (_ps(realistic(i), "ZZ"),), call), _same_scalar,
                ("used: f.discriminant() 72 us; others: discriminant(f) 101 us; medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=3),
        Fixture("discriminant", "realistic", "scaled_random_degree_20",
                _mathforge(mf.discriminant, _poly_one(scaled)), _sympy(lambda i: (_ps(scaled(i), "ZZ"),), call),
                _same_scalar,
                ("used: f.discriminant() 208 us; others: discriminant(f) 238 us; medians of 9 calls, SymPy 1.14.0 "
                 "(python ground types), caches cleared before each call"), variants=3),
    )


@functools.cache
def _rf(numerator: tuple[int, ...], denominator: tuple[int, ...]):
    """The MathForge RationalFunction numerator/denominator in X, built once."""
    return mf.rational_function(_pm(numerator), _pm(denominator), variable=X)


def _same_partial_fractions(mf_value, sympy_value) -> bool:
    """A PartialFractionDecomposition against SymPy's partial fraction expression: the same rational function."""
    import sympy as sp
    expected = _sp_poly(mf_value.polynomial).as_expr()
    for term in mf_value.terms:
        expected += _sp_poly(term.numerator).as_expr() / _sp_poly(term.factor).as_expr() ** term.power
    return sp.cancel(expected - sympy_value) == 0


def _apart_micro(i: int):
    """Micro variant i of apart: (a*x + b)/((x + c)*(x + d)) with a in [1, 5], b in [-9, 9], c < d; distinct for
    i < 14,440. Returns (numerator, denominator) coefficients."""
    a, b, c, dd = _spread(i, 5, 19, 19, 8)
    c -= 9
    return (b - 9, a + 1), _times((c, 1), (c + 1 + dd, 1))


_APART_REALISTIC = ((1, 1, 1), (2, 3, 1), (3, 2, 5))  # (x + c)/((x - a)**2 * (x**2 + b)) as (a, b, c)
_APART_SCALED = ((1, 1), (2, 2), (3, 3))  # degree 5 over degree 9, see _apart_scaled


def _apart_realistic(i: int):
    a, b, c = _APART_REALISTIC[i]
    return (c, 1), _times(_power((-a, 1), 2), (b, 0, 1))


def _apart_scaled(i: int):
    """(x**5 + a*x**2 + 2*x + 7 + b)/((x-1)*(x+2)*(x**2+x+1)*(x-3)*(x**2+b)*(x+4)): degree 5 over degree 9."""
    a, b = _APART_SCALED[i]
    denominator = _times((-1, 1), (2, 1), (1, 1, 1), (-3, 1), (b, 0, 1), (4, 1))
    return (7 + b, 2, a, 0, 0, 1), denominator


def _apart_inputs_mf(description):
    return lambda i: (_rf(*description(i)),)


def _apart_inputs_sp(description):
    def build(i):
        numerator, denominator = description(i)
        return _ps(numerator).as_expr() / _ps(denominator).as_expr(), _xs()
    return build


def _apart_fixtures():
    call = lambda sp, f, x: sp.apart(f, x)
    return (
        Fixture("apart", "micro", "two_simple_poles", _mathforge(mf.apart, _apart_inputs_mf(_apart_micro)),
                _sympy(_apart_inputs_sp(_apart_micro), call), _same_partial_fractions,
                ("used: apart(f, x) 4.32 ms; others: apart(f) 5.91 ms; medians of 150 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=_MICRO),
        Fixture("apart", "realistic", "double_pole_and_quadratic",
                _mathforge(mf.apart, _apart_inputs_mf(_apart_realistic)),
                _sympy(_apart_inputs_sp(_apart_realistic), call), _same_partial_fractions,
                ("used: apart(f, x) 9.09 ms; others: apart(f) 9.09 ms; medians of 60 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=len(_APART_REALISTIC)),
        Fixture("apart", "realistic", "scaled_degree_9_denominator",
                _mathforge(mf.apart, _apart_inputs_mf(_apart_scaled)),
                _sympy(_apart_inputs_sp(_apart_scaled), call), _same_partial_fractions,
                ("used: apart(f, x) 26.17 ms; others: apart(f) 28.14 ms; medians of 60 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=len(_APART_SCALED)),
    )


# --- real algebraic numbers: algebraic, algebraic_div, algebraic_pow, minimal_polynomial -------------------------
# An expression description is (offset, radicands): offset + sqrt(r1) + sqrt(r2) + ..., all radicands non-squares.

def _expression_mf(description):
    offset, radicands = description
    value = mf.Rational(offset)
    for r in radicands:
        value = value + mf.sqrt(r)
    return value


def _expression_sp(description):
    import sympy as sp
    offset, radicands = description
    return sp.Integer(offset) + sum(sp.sqrt(r) for r in radicands)


def _t():
    import sympy as sp
    return sp.Symbol("t")


def _same_algebraic(mf_value, found) -> bool:
    """A Rational or RealAlgebraicRoot against (minimal polynomial in t, the number it names)."""
    name, number = found
    return (type(mf_value) in (mf.Rational, mf.RealAlgebraicRoot) and _same_number(_sp_real(mf_value), number)
            and _name_matches(mf_value, name))


def _surd_micro(i: int):
    """Micro variant i: p/q + r*sqrt(s), p in [-9, 9], q and r in [1, 9], s one of 20 non-squares; distinct for
    i < 30,780."""
    p, q, r, s = _spread(i, 19, 9, 9, 20)
    return mf.Rational(p - 9, q + 1), r + 1, _NON_SQUARES[s]


def _surd_micro_mf(i):
    p, r, s = _surd_micro(i)
    return p + r * mf.sqrt(s)


def _surd_micro_sp(i):
    import sympy as sp
    p, r, s = _surd_micro(i)
    return sp.Rational(p.numerator, p.denominator) + r * sp.sqrt(s)


_TWO_ROOTS = ((0, (2, 3)), (0, (3, 5)), (0, (5, 7)))
_THREE_ROOTS = ((0, (2, 3, 5)), (0, (2, 3, 7)), (0, (3, 5, 7)))


def _minpoly_call(sp, e, t):
    return sp.minimal_polynomial(e, t, compose=False), e


def _algebraic_fixtures():
    return (
        Fixture("algebraic", "micro", "quadratic_surd", _mathforge(mf.algebraic, lambda i: (_surd_micro_mf(i),)),
                _sympy(lambda i: (_surd_micro_sp(i), _t()), _minpoly_call), _same_algebraic,
                ("used: minimal_polynomial(e, t, compose=False) 3.88 ms; others: minimal_polynomial(e, t) 4.98 ms; "
                 "minimal_polynomial(e, t, polys=True) 4.59 ms; AlgebraicNumber(e).minpoly n/a (raised on this "
                 "input); nsimplify(e) [unnormalised] 10.23 ms (excluded: gives an expression, not an identified "
                 "number); medians of 150 calls, SymPy 1.14.0 (python ground types), caches cleared before each "
                 "call"), variants=_MICRO),
        Fixture("algebraic", "realistic", "sum_of_two_square_roots",
                _mathforge(mf.algebraic, lambda i: (_expression_mf(_TWO_ROOTS[i]),)),
                _sympy(lambda i: (_expression_sp(_TWO_ROOTS[i]), _t()), _minpoly_call), _same_algebraic,
                ("used: minimal_polynomial(e, t, compose=False) 5.35 ms; others: minimal_polynomial(e, t) 6.73 ms; "
                 "minimal_polynomial(e, t, polys=True) 6.63 ms; AlgebraicNumber(e).minpoly n/a (raised on this "
                 "input); nsimplify(e) [unnormalised] 3.12 ms (excluded: gives an expression, not an identified "
                 "number); medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_TWO_ROOTS)),
        Fixture("algebraic", "realistic", "scaled_sum_of_three_square_roots",
                _mathforge(mf.algebraic, lambda i: (_expression_mf(_THREE_ROOTS[i]),)),
                _sympy(lambda i: (_expression_sp(_THREE_ROOTS[i]), _t()), _minpoly_call), _same_algebraic,
                ("used: minimal_polynomial(e, t, compose=False) 12.19 ms; others: minimal_polynomial(e, t) 15.31 "
                 "ms; minimal_polynomial(e, t, polys=True) 14.46 ms; AlgebraicNumber(e).minpoly n/a (raised on "
                 "this input); nsimplify(e) [unnormalised] 12.23 ms (excluded: gives an expression, not an "
                 "identified number); medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared "
                 "before each call"), variants=len(_THREE_ROOTS)),
    )


def _quotient_micro(i: int):
    """Micro variant i: p/(q + r*sqrt(s)), p in [1, 9], q in [-9, 9], r in [1, 9], s a non-square; distinct for
    i < 30,780."""
    p, q, r, s = _spread(i, 9, 19, 9, 20)
    return p + 1, q - 9, r + 1, _NON_SQUARES[s]


_QUOTIENTS = (((0, (2,)), (1, (3,))),          # sqrt 2 / (1 + sqrt 3)
              ((1, ()), (0, (2, 3))),          # 1 / (sqrt 2 + sqrt 3)
              ((1, (5,)), (0, (2, 7))))        # (1 + sqrt 5) / (sqrt 2 + sqrt 7)
_SCALED_QUOTIENTS = (((1, ()), (0, (2, 3, 5))),       # 1 / (sqrt 2 + sqrt 3 + sqrt 5)
                     ((0, (7,)), (1, (2, 3))),        # sqrt 7 / (1 + sqrt 2 + sqrt 3)
                     ((1, (5,)), (0, (2, 3, 5))))     # (1 + sqrt 5) / (sqrt 2 + sqrt 3 + sqrt 5)


def _quotient_call(sp, left, right, t):
    e = left / right
    return sp.minimal_polynomial(e, t, compose=False), e


def _algebraic_div_fixtures():
    def micro_mf(i):
        p, q, r, s = _quotient_micro(i)
        return p, q + r * mf.sqrt(s)

    def micro_sp(i):
        import sympy as sp
        p, q, r, s = _quotient_micro(i)
        return sp.Integer(p), q + r * sp.sqrt(s), _t()

    def pair_mf(table):
        return lambda i: (_expression_mf(table[i][0]), _expression_mf(table[i][1]))

    def pair_sp(table):
        return lambda i: (_expression_sp(table[i][0]), _expression_sp(table[i][1]), _t())

    return (
        Fixture("algebraic_div", "micro", "rational_over_quadratic_surd",
                _mathforge(mf.algebraic_div, micro_mf), _sympy(micro_sp, _quotient_call), _same_algebraic,
                ("used: minimal_polynomial(l/r, t, compose=False) 5.56 ms; others: minimal_polynomial(l/r, t) 8.13 "
                 "ms; radsimp(l/r) [unnormalised] 6.29 ms (excluded: gives an expression, not an identified "
                 "number); medians of 150 calls, SymPy 1.14.0 (python ground types), caches cleared before each "
                 "call"), variants=_MICRO),
        Fixture("algebraic_div", "realistic", "quotients_of_square_root_sums",
                _mathforge(mf.algebraic_div, pair_mf(_QUOTIENTS)), _sympy(pair_sp(_QUOTIENTS), _quotient_call),
                _same_algebraic,
                ("used: minimal_polynomial(l/r, t, compose=False) 6.21 ms; others: minimal_polynomial(l/r, t) 8.22 "
                 "ms; radsimp(l/r) [unnormalised] 7.98 ms (excluded: gives an expression, not an identified "
                 "number); medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_QUOTIENTS)),
        Fixture("algebraic_div", "realistic", "scaled_quotients_of_three_root_sums",
                _mathforge(mf.algebraic_div, pair_mf(_SCALED_QUOTIENTS)),
                _sympy(pair_sp(_SCALED_QUOTIENTS), _quotient_call), _same_algebraic,
                ("used: minimal_polynomial(l/r, t, compose=False) 13.19 ms; others: minimal_polynomial(l/r, t) "
                 "16.54 ms; radsimp(l/r) [unnormalised] 23.27 ms (excluded: gives an expression, not an identified "
                 "number); medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_SCALED_QUOTIENTS)),
    )


def _power_micro(i: int):
    """Micro variant i: (q + r*sqrt(s))**n, q in [-9, 9], r in [1, 9], s a non-square, n in [2, 5]; distinct for
    i < 13,680."""
    q, r, s, n = _spread(i, 19, 9, 20, 4)
    return q - 9, r + 1, _NON_SQUARES[s], n + 2


_POWERS = (((1, (2,)), 5), ((0, (2, 3)), -2), ((1, (3,)), 7))
_SCALED_POWERS = (((0, (2, 3)), 12), ((0, (2, 5)), 10), ((1, (2, 3)), 5))


def _power_call(sp, base, n, t):
    e = base ** n
    return sp.minimal_polynomial(e, t, compose=False), e


def _algebraic_pow_fixtures():
    def micro_mf(i):
        q, r, s, n = _power_micro(i)
        return q + r * mf.sqrt(s), n

    def micro_sp(i):
        import sympy as sp
        q, r, s, n = _power_micro(i)
        return q + r * sp.sqrt(s), n, _t()

    def table_mf(table):
        return lambda i: (_expression_mf(table[i][0]), table[i][1])

    def table_sp(table):
        return lambda i: (_expression_sp(table[i][0]), table[i][1], _t())

    return (
        Fixture("algebraic_pow", "micro", "quadratic_surd_small_power", _mathforge(mf.algebraic_pow, micro_mf),
                _sympy(micro_sp, _power_call), _same_algebraic,
                ("used: minimal_polynomial(b**n, t, compose=False) 5.27 ms; others: minimal_polynomial(b**n, t) "
                 "6.49 ms; expand(b**n) + minimal_polynomial 6.44 ms; medians of 150 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("algebraic_pow", "realistic", "surd_powers", _mathforge(mf.algebraic_pow, table_mf(_POWERS)),
                _sympy(table_sp(_POWERS), _power_call), _same_algebraic,
                ("used: minimal_polynomial(b**n, t, compose=False) 4.71 ms; others: minimal_polynomial(b**n, t) "
                 "5.74 ms; expand(b**n) + minimal_polynomial 6.00 ms; medians of 9 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"),
                variants=len(_POWERS)),
        Fixture("algebraic_pow", "realistic", "scaled_surd_powers",
                _mathforge(mf.algebraic_pow, table_mf(_SCALED_POWERS)), _sympy(table_sp(_SCALED_POWERS), _power_call),
                _same_algebraic,
                ("used: minimal_polynomial(b**n, t, compose=False) 7.64 ms; others: minimal_polynomial(b**n, t) "
                 "9.30 ms; expand(b**n) + minimal_polynomial 8.81 ms; medians of 9 calls, SymPy 1.14.0 (python "
                 "ground types), caches cleared before each call"), variants=len(_SCALED_POWERS)),
    )


def _minimal_polynomial_fixtures():
    call = lambda sp, e, x: sp.minimal_polynomial(e, x, compose=False, polys=True).monic()
    x_symbol = _xs

    return (
        Fixture("minimal_polynomial", "micro", "quadratic_surd",
                _mathforge(mf.minimal_polynomial, lambda i: (_surd_micro_mf(i), X)),
                _sympy(lambda i: (_surd_micro_sp(i), x_symbol()), call), _same_poly,
                ("used: minimal_polynomial(e, x, compose=False, polys=True).monic() 3.84 ms; others: "
                 "Poly(minimal_polynomial(e, x, compose=False), x).monic() 4.29 ms; minimal_polynomial(e, x, "
                 "polys=True).monic() 4.74 ms; minimal_polynomial(e, x, compose=False) 3.89 ms (excluded: not "
                 "monic: a rational multiple of the minimal polynomial when it has fractional coefficients); "
                 "medians of 150 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=_MICRO),
        Fixture("minimal_polynomial", "realistic", "sum_of_two_square_roots",
                _mathforge(mf.minimal_polynomial, lambda i: (_expression_mf(_TWO_ROOTS[i]), X)),
                _sympy(lambda i: (_expression_sp(_TWO_ROOTS[i]), x_symbol()), call), _same_poly,
                ("used: minimal_polynomial(e, x, compose=False, polys=True).monic() 5.52 ms; others: "
                 "Poly(minimal_polynomial(e, x, compose=False), x).monic() 5.71 ms; minimal_polynomial(e, x, "
                 "polys=True).monic() 6.42 ms; minimal_polynomial(e, x, compose=False) 5.66 ms (excluded: not "
                 "monic: a rational multiple of the minimal polynomial when it has fractional coefficients); "
                 "medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_TWO_ROOTS)),
        Fixture("minimal_polynomial", "realistic", "scaled_sum_of_three_square_roots",
                _mathforge(mf.minimal_polynomial, lambda i: (_expression_mf(_THREE_ROOTS[i]), X)),
                _sympy(lambda i: (_expression_sp(_THREE_ROOTS[i]), x_symbol()), call), _same_poly,
                ("used: minimal_polynomial(e, x, compose=False, polys=True).monic() 12.76 ms; others: "
                 "Poly(minimal_polynomial(e, x, compose=False), x).monic() 12.88 ms; minimal_polynomial(e, x, "
                 "polys=True).monic() 15.38 ms; minimal_polynomial(e, x, compose=False) 12.45 ms (excluded: not "
                 "monic: a rational multiple of the minimal polynomial when it has fractional coefficients); "
                 "medians of 9 calls, SymPy 1.14.0 (python ground types), caches cleared before each call"),
                variants=len(_THREE_ROOTS)),
    )


POLYNOMIAL_FIXTURES = (
    *_expand_fixtures(),
    *_expand_multivariate_fixtures(),
    *_poly_divmod_fixtures(),
    *_poly_gcd_fixtures(),
    *_poly_xgcd_fixtures(),
    *_square_free_fixtures(),
    *_real_roots_fixtures(),
    *_compare_real_fixtures(),
    *_rational_function_fixtures(),
    *_factor_fixtures(),
    *_resultant_fixtures(),
    *_discriminant_fixtures(),
    *_apart_fixtures(),
    *_algebraic_fixtures(),
    *_algebraic_div_fixtures(),
    *_algebraic_pow_fixtures(),
    *_minimal_polynomial_fixtures(),
)

FIXTURES = (*COMPLEX_FIXTURES, *POLYNOMIAL_FIXTURES)


# ---------------------------------------------------------------------------------------------------------------
# Edge cases: one input per v0.7 error class an operation can raise, one input far past a default limit, and the
# True / 1.5 / "1" / None type edges in the first argument. Over-limit inputs are about 2x or more past the limit
# (integers of 140,001 bits against 65,536; expressions of 300,000 nodes against 100,000).
# ---------------------------------------------------------------------------------------------------------------

def _edge(operation, label, function, args=(), kwargs=None):
    """An EdgeCase; ``args`` and ``kwargs`` may be callables, which run each time the case is built."""
    return EdgeCase(operation, label, lambda: (function, args() if callable(args) else args,
                                               (kwargs() if callable(kwargs) else kwargs) or {}))


def _type_edges(operation, function, rest=(), kwargs=None):
    """The True, 1.5, "1" and None edges in the first argument; ``rest`` holds the other positional arguments."""
    return tuple(_edge(operation, f"first_argument_{label}", function,
                       lambda value=value: (value, *(rest() if callable(rest) else rest)), kwargs)
                 for label, value in (("true", True), ("float", 1.5), ("string", "1"), ("none", None)))


def _gauss_z(a: int = 1, b: int = 2):
    return _zm((((a, 1, 1),), ((b, 1, 1),)))


def _repeated_sum():
    """x + x + ... + x with 300,000 terms: 3x max_nodes, built from one shared node."""
    return mf.Add((X,) * 300_000)


def _in_y(coefficients):
    return mf.Polynomial(coefficients, Y)


def _forged_rational_function():
    """A RationalFunction record with a 140,001-bit coefficient, built without its constructor (which refuses it)."""
    record = object.__new__(mf.RationalFunction)
    for name, value in (("numerator", mf.Polynomial((_HUGE, 1), X)), ("denominator", mf.Polynomial((1, 0, 1), X)),
                        ("excluded", (mf.Polynomial((1, 0, 1), X),))):
        object.__setattr__(record, name, value)
    return record


def _binary_edges(function, extra=()):
    """Edges of a binary complex operation: type edges in the left operand, a bad right operand, a symbol, an integer
    far past max_integer_bits."""
    name = function.__name__
    return (
        *_type_edges(name, function, lambda: (_gauss_z(),)),
        _edge(name, "right_operand_none", function, lambda: (_gauss_z(), None)),
        _edge(name, "free_symbol_operand", function, lambda: (X, _gauss_z())),
        _edge(name, "over_limit_integer", function, lambda: (_HUGE, _gauss_z())),
        *extra,
    )


def _unary_edges(function):
    name = function.__name__
    return (
        *_type_edges(name, function),
        _edge(name, "free_symbol_operand", function, lambda: (X,)),
        _edge(name, "over_limit_integer", function, lambda: (_HUGE,)),
    )


def _polynomial_pair_edges(function):
    name = function.__name__
    return (
        *_type_edges(name, function, lambda: (_pm((1, 1)),)),
        _edge(name, "second_operand_none", function, lambda: (_pm((1, 1)), None)),
        _edge(name, "different_variables", function, lambda: (_pm((1, 1)), _in_y((1, 1)))),
        _edge(name, "over_limit_integer", function, lambda: (mf.Polynomial((_HUGE, 1), X), _pm((1, 1)))),
    )


def _polynomial_edges(function):
    name = function.__name__
    return (
        *_type_edges(name, function),
        _edge(name, "expression_not_a_polynomial", function, lambda: (X + 1,)),
        _edge(name, "over_limit_integer", function, lambda: (mf.Polynomial((_HUGE, 1, 1), X),)),
    )


def _expression_edges(function, rest, name=None, over_limit_nodes=True):
    """Edges of an operation whose first argument is an expression and whose other arguments are ``rest()``."""
    name = name or function.__name__
    edges = [
        *_type_edges(name, function, rest),
        _edge(name, "over_limit_integer", function, lambda: (_HUGE * X, *rest())),
    ]
    if over_limit_nodes:
        edges.append(_edge(name, "over_limit_nodes", function, lambda: (_repeated_sum(), *rest())))
    return tuple(edges)


EDGE_CASES = (
    # complex scalars
    *_type_edges("complex_algebraic", mf.complex_algebraic, (1,)),
    _edge("complex_algebraic", "imaginary_part_float", mf.complex_algebraic, (1, 1.5)),
    _edge("complex_algebraic", "free_symbol_part", mf.complex_algebraic, lambda: (X, 0)),
    _edge("complex_algebraic", "over_limit_integer", mf.complex_algebraic, lambda: (_HUGE, 0)),
    *_binary_edges(mf.complex_add),
    *_binary_edges(mf.complex_sub),
    *_binary_edges(mf.complex_mul),
    *_binary_edges(mf.complex_div, extra=(
        _edge("complex_div", "zero_divisor", mf.complex_div, lambda: (_gauss_z(), 0)),
        _edge("complex_div", "zero_complex_divisor", mf.complex_div, lambda: (_gauss_z(), _gauss_z(0, 0))),
    )),
    *_unary_edges(mf.complex_conjugate),
    *_unary_edges(mf.complex_real),
    *_unary_edges(mf.complex_imag),
    *_unary_edges(mf.complex_norm_squared),
    *_unary_edges(mf.complex_sqrt),
    *_type_edges("complex_pow", mf.complex_pow, (2,)),
    _edge("complex_pow", "exponent_float", mf.complex_pow, lambda: (_gauss_z(), 1.5)),
    _edge("complex_pow", "exponent_none", mf.complex_pow, lambda: (_gauss_z(), None)),
    _edge("complex_pow", "zero_to_a_negative_power", mf.complex_pow, lambda: (_gauss_z(0, 0), -1)),
    _edge("complex_pow", "over_limit_integer_base", mf.complex_pow, lambda: (_HUGE, 2)),
    _edge("complex_pow", "over_limit_integer_exponent", mf.complex_pow, lambda: (_gauss_z(), _HUGE)),
    _edge("complex_pow", "over_limit_result_bits", mf.complex_pow, lambda: (_gauss_z(), 10 ** 6)),
    *_type_edges("quadratic_roots", mf.quadratic_roots, (2, 3)),
    _edge("quadratic_roots", "zero_leading_coefficient", mf.quadratic_roots, (0, 2, 3)),
    _edge("quadratic_roots", "irrational_coefficient", mf.quadratic_roots, lambda: (mf.sqrt(2), 2, 3)),
    _edge("quadratic_roots", "over_limit_integer", mf.quadratic_roots, lambda: (_HUGE, 2, 3)),
    # polynomials
    *_expression_edges(mf.expand, lambda: (X,)),
    _edge("expand", "other_variable", mf.expand, lambda: (Y + 1, X)),
    _edge("expand", "irrational_coefficient", mf.expand, lambda: (mf.sqrt(2) * X, X)),
    _edge("expand", "variable_not_a_symbol", mf.expand, lambda: (X + 1, 5)),
    _edge("expand", "degree_past_the_dense_limit", mf.expand, lambda: (X ** 2000, X)),
    *_expression_edges(mf.expand_multivariate, lambda: ((X, Y),)),
    _edge("expand_multivariate", "variable_missing_from_the_list", mf.expand_multivariate, lambda: (X + Y, (X,))),
    _edge("expand_multivariate", "irrational_coefficient", mf.expand_multivariate, lambda: (mf.sqrt(2) * X, (X, Y))),
    _edge("expand_multivariate", "duplicate_variables", mf.expand_multivariate, lambda: (X + Y, (X, X))),
    _edge("expand_multivariate", "variables_as_text", mf.expand_multivariate, lambda: (X + Y, "xy")),
    _edge("expand_multivariate", "no_variables", mf.expand_multivariate, lambda: (X + Y, ())),
    *_polynomial_pair_edges(mf.poly_divmod),
    _edge("poly_divmod", "zero_divisor", mf.poly_divmod, lambda: (_pm((1, 1)), _pm((0,)))),
    *_polynomial_pair_edges(mf.poly_gcd),
    *_polynomial_pair_edges(mf.poly_xgcd),
    *_polynomial_edges(mf.square_free),
    *_expression_edges(mf.real_roots, lambda: (X,)),
    _edge("real_roots", "other_variable", mf.real_roots, lambda: (Y + 1, X)),
    _edge("real_roots", "irrational_coefficient", mf.real_roots, lambda: (mf.sqrt(2) * X - 1, X)),
    _edge("real_roots", "zero_polynomial", mf.real_roots, lambda: (0, X)),
    _edge("real_roots", "variable_not_a_symbol", mf.real_roots, lambda: (X + 1, 5)),
    *_type_edges("compare_real", mf.compare_real, (1,)),
    _edge("compare_real", "second_operand_text", mf.compare_real, (1, "1")),
    _edge("compare_real", "free_symbol_operand", mf.compare_real, lambda: (X, 1)),
    _edge("compare_real", "over_limit_integer", mf.compare_real, lambda: (_HUGE, 1)),
    *_type_edges("rational_function", mf.rational_function, (1,), lambda: {"variable": X}),
    _edge("rational_function", "zero_denominator", mf.rational_function, (1, 0), lambda: {"variable": X}),
    _edge("rational_function", "variable_not_in_the_expression", mf.rational_function, lambda: (1, X),
          lambda: {"variable": Y}),
    _edge("rational_function", "variable_not_a_symbol", mf.rational_function, lambda: (1, X), {"variable": 5}),
    _edge("rational_function", "irrational_coefficient", mf.rational_function, lambda: (mf.sqrt(2), X),
          lambda: {"variable": X}),
    _edge("rational_function", "over_limit_integer", mf.rational_function, lambda: (_HUGE, X),
          lambda: {"variable": X}),
    *_polynomial_edges(mf.factor),
    _edge("factor", "zero_polynomial", mf.factor, lambda: (_pm((0,)),)),
    _edge("factor", "constant_polynomial", mf.factor, lambda: (_pm((5,)),)),
    *_polynomial_pair_edges(mf.resultant),
    *_polynomial_edges(mf.discriminant),
    _edge("discriminant", "constant_polynomial", mf.discriminant, lambda: (_pm((5,)),)),
    *_type_edges("apart", mf.apart),
    _edge("apart", "polynomial_not_a_rational_function", mf.apart, lambda: (X + 1,)),
    _edge("apart", "over_limit_integer", mf.apart, lambda: (_forged_rational_function(),)),
    # real algebraic numbers
    *_type_edges("algebraic", mf.algebraic),
    _edge("algebraic", "free_symbol", mf.algebraic, lambda: (X,)),
    _edge("algebraic", "over_limit_integer", mf.algebraic, lambda: (_HUGE + mf.sqrt(2),)),
    _edge("algebraic", "over_limit_nodes", mf.algebraic, lambda: (_repeated_sum(),)),
    _edge("algebraic", "over_limit_power", mf.algebraic, lambda: (mf.Pow(1 + mf.sqrt(2), 1_000_000),)),
    *_type_edges("algebraic_div", mf.algebraic_div, lambda: (mf.sqrt(2),)),
    _edge("algebraic_div", "zero_divisor", mf.algebraic_div, (1, 0)),
    _edge("algebraic_div", "divisor_that_is_zero_algebraically", mf.algebraic_div,
          lambda: (1, mf.Add((mf.Sqrt(mf.Rational(8)), mf.Mul((mf.Rational(-2), mf.Sqrt(mf.Rational(2)))))))),
    _edge("algebraic_div", "free_symbol_operand", mf.algebraic_div, lambda: (X, 1)),
    _edge("algebraic_div", "over_limit_integer", mf.algebraic_div, lambda: (_HUGE, mf.sqrt(2))),
    *_type_edges("algebraic_pow", mf.algebraic_pow, (2,)),
    _edge("algebraic_pow", "zero_to_a_negative_power", mf.algebraic_pow, (0, -1)),
    _edge("algebraic_pow", "exponent_float", mf.algebraic_pow, lambda: (mf.sqrt(2), 1.5)),
    _edge("algebraic_pow", "exponent_bool", mf.algebraic_pow, lambda: (mf.sqrt(2), True)),
    _edge("algebraic_pow", "exponent_none", mf.algebraic_pow, lambda: (mf.sqrt(2), None)),
    _edge("algebraic_pow", "free_symbol_base", mf.algebraic_pow, lambda: (X, 2)),
    _edge("algebraic_pow", "over_limit_integer_base", mf.algebraic_pow, lambda: (_HUGE, 2)),
    _edge("algebraic_pow", "over_limit_integer_exponent", mf.algebraic_pow, lambda: (mf.sqrt(2), _HUGE)),
    *_type_edges("minimal_polynomial", mf.minimal_polynomial, lambda: (X,)),
    _edge("minimal_polynomial", "free_symbol_expression", mf.minimal_polynomial, lambda: (X, X)),
    _edge("minimal_polynomial", "variable_not_a_symbol", mf.minimal_polynomial, lambda: (mf.sqrt(2), 5)),
    _edge("minimal_polynomial", "constrained_variable", mf.minimal_polynomial,
          lambda: (mf.sqrt(2), mf.Symbol("t", mf.Reals, (mf.Assumption("positive", mf.Truth.TRUE),),
                                         uid="00000000-0000-0000-0000-0000000000a1"))),
    _edge("minimal_polynomial", "over_limit_integer", mf.minimal_polynomial, lambda: (_HUGE + mf.sqrt(2), X)),
)
