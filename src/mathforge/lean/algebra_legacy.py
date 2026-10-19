"""Lean dispatcher of the legacy compare_real operation, whose public function routes on its first result.

``mathforge.operations.compare_real`` runs its v2 ``operations.LEGACY`` entry first and, when that result is
unsupported (its producer raised UnsupportedOperation), the v4 exact path (``_run_exact``). This lean function takes
the same two steps, after one fast path: two closed real algebraic expressions (Rational, Sqrt, Add, Mul and Pow
nodes, not both rational) whose values differ are compared by fixed-point interval enclosures refined until they
separate (mathforge.kernels.upoly.separate), and the sign is returned only after the checker re-encloses both sides
with exact rational intervals at that precision (mathforge.certify.upoly.check_sign). Equal values never separate;
they, and everything else, take the routes of the public function. Both structured routes accept every such
expression, up to their resource limits.
"""

from __future__ import annotations

from ..certify import upoly as certify
from ..kernels import upoly as kernels
from ..limits import computation, current_budget
from ..model import Add, Mul, Pow, Rational, Sqrt
from . import FALLBACK, run_legacy

_UNSUPPORTED = object()
_CLOSED = frozenset({Rational, Sqrt, Add, Mul, Pow})


def _separated(left: object, right: object) -> object:
    """The certified sign of left - right for differing closed expressions, or FALLBACK."""
    if current_budget() is not None:
        return FALLBACK
    if type(left) is int:
        left = Rational(left) if left.bit_length() <= 65_536 else None
    if type(right) is int:
        right = Rational(right) if right.bit_length() <= 65_536 else None
    if type(left) not in _CLOSED or type(right) not in _CLOSED or (type(left) is Rational and type(right) is Rational):
        return FALLBACK
    from .expr_poly import _Walk
    try:
        walk = _Walk()
        walk.add(left)
        walk.add(right)
        with computation(fresh=True):
            found = kernels.separate(left, right)
            if found is None:
                return FALLBACK  # Possibly equal: the exact routes decide.
            sign, bits = found
            if certify.check_sign(left, right, sign, bits) is not True:
                return FALLBACK
            return sign
    except Exception:
        return FALLBACK


def compare_real(left: object, right: object) -> object:
    value = _separated(left, right)
    if value is not FALLBACK:
        return value
    arguments = {"left": left, "right": right}
    value = run_legacy("compare_real", arguments, unsupported=_UNSUPPORTED)
    if value is _UNSUPPORTED:
        return run_legacy("compare_real", arguments, exact=True)
    return value


LEAN = {"compare_real": compare_real}
