"""Lean dispatcher of the legacy compare_real operation, whose public function routes on its first result.

``mathforge.operations.compare_real`` runs its v2 ``operations.LEGACY`` entry first and, when that result is
unsupported (its producer raised UnsupportedOperation), the v4 exact path (``_run_exact``). This lean function takes
the same two steps.
"""

from __future__ import annotations

from . import run_legacy

_UNSUPPORTED = object()


def compare_real(left: object, right: object) -> object:
    arguments = {"left": left, "right": right}
    value = run_legacy("compare_real", arguments, unsupported=_UNSUPPORTED)
    if value is _UNSUPPORTED:
        return run_legacy("compare_real", arguments, exact=True)
    return value


LEAN = {"compare_real": compare_real}
