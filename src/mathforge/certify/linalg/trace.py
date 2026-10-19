"""Independent checker of a trace, by v0.7's criterion: the diagonal entries, added as Rationals, give the value.

The producer adds the diagonal over a common denominator in integers; this check adds the Rationals one by one.
"""

from __future__ import annotations

from mathforge.model import Rational


def check_trace(rows, value) -> bool:
    """Whether value is the sum of the diagonal of the square matrix rows (a tuple of row tuples of Rationals,
    whose diagonal the caller has validated)."""
    if type(value) is not Rational:
        return False
    total = Rational(0)
    for index, row in enumerate(rows):
        total = total + row[index]
    return total == value
