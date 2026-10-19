"""Three tiny fixtures for the speed suite's own tests (tests/v08/test_speed_suite.py): not a registry area.

The suite is run on this module with ``--modules tests.v08.speed_demo_fixtures``, which makes the run a complete one:
both files are written. Two Fixtures compare a real MathForge value function with the SymPy call that returns the same
object, or with the standard library when SymPy is not installed, so that the unit suite runs without SymPy; one
ExemptFixture of ``verify`` has a timed call that is trivially faster than its budget, so that its ``budget_met`` is
true on any machine. The micro fixture has 50 variants, not the 1,000 that a real micro fixture needs: it keeps the
tests short.
"""

import time

import mathforge as mf
from benchmarks.speed.fixture import ExemptFixture, Fixture


def _reference(name):
    """sympy.<name>, or the standard library's counterpart when SymPy is absent."""
    try:
        import sympy
        return getattr(sympy, {"gcd": "igcd"}.get(name, name))
    except ImportError:
        import math
        return getattr(math, name)


def _gcd_inputs(i):
    return 84 + 6 * i, 30 + 6 * i


def _gcd_sympy(i):
    a, b = _gcd_inputs(i)
    igcd = _reference("gcd")
    return lambda: igcd(a, b)


def _factorial_sympy(i):
    n = 200 + i
    factorial = _reference("factorial")
    return lambda: factorial(n)


def _same_integer(mathforge_value, sympy_value):
    return mathforge_value == int(sympy_value)


def _verify_timed():
    return sum, ((),), {}  # Trivially quick: far under the budget below.


def _verify_budget():
    return time.sleep, (0.002,), {}


FIXTURES = (
    Fixture("integer_gcd", "micro", "demo", lambda i: (mf.integer_gcd, _gcd_inputs(i), {}), _gcd_sympy,
            _same_integer, "demo: sympy.igcd (math.gcd without SymPy)", variants=50),
    Fixture("factorial", "realistic", "demo", lambda i: (mf.factorial, (200 + i,), {}), _factorial_sympy,
            _same_integer, "demo: sympy.factorial (math.factorial without SymPy)", variants=3),
    ExemptFixture("verify", "micro", "demo", _verify_timed, _verify_budget, "demo: no SymPy counterpart"),
)
EDGE_CASES = ()
