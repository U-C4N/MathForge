"""Exact descriptive statistics for finite rational samples."""

from .errors import InvalidInput
from .limits import computation, current_budget
from .model import Rational, Sqrt, sqrt
from .probability import _integer, _rational, _values, _total

ZERO = Rational(0)


def _sample(data):
    data = _values(data)
    if not data:
        raise InvalidInput("A sample must contain at least one observation")
    return data


def _degrees(ddof, count):
    ddof = _integer(ddof, "ddof")
    if ddof >= count:
        raise InvalidInput("ddof must be smaller than the sample size")
    return count - ddof


def mean(data):
    with computation():
        data = _sample(data)
        return _total(data) / Rational(len(data))


def median(data):
    with computation():
        data = _sample(data)
        current_budget().nodes(len(data))
        ordered = sorted(data)
        middle = len(data) // 2
        return ordered[middle] if len(data) % 2 else (ordered[middle - 1] + ordered[middle]) / Rational(2)


def variance(data, ddof=0):
    with computation():
        data = _sample(data)
        denominator = _degrees(ddof, len(data))
        center = _total(data) / Rational(len(data))
        return _total((value - center)**2 for value in data) / Rational(denominator)


def stddev(data, ddof=0):
    with computation():
        return sqrt(variance(data, ddof))


def covariance(x, y, ddof=0):
    with computation():
        x, y = _sample(x), _sample(y)
        if len(x) != len(y):
            raise InvalidInput("Paired samples must have the same length")
        denominator = _degrees(ddof, len(x))
        center_x, center_y = _total(x) / Rational(len(x)), _total(y) / Rational(len(y))
        return _total((a - center_x) * (b - center_y) for a, b in zip(x, y)) / Rational(denominator)


def median_certificate(nums, dens, numerator, denominator):
    """Whether numerator/denominator is the median of the sample nums[i]/dens[i], by one counting pass.

    The sample entries are canonical rationals (positive denominators) and so is the claimed value (denominator > 0).
    Both middle ranks of the sorted sample must be accounted for by counts alone, without sorting and without
    trusting the producer's order statistics: count the entries below, equal to and above the claimed value, and keep
    the largest entry below and the smallest entry above it.

    * Odd size: the value is the median exactly when the middle rank lies inside the block of entries equal to it.
    * Even size: it is the median exactly when both middle ranks lie inside that block (the two middle order
      statistics are equal to it), or when no entry equals it, as many entries lie below as above, and it is the
      average of the largest entry below and the smallest entry above (the two middle order statistics). If the two
      middle order statistics differ, no entry lies between them, so no entry equals their average.

    This is the verdict of the quadratic check that located both order statistics by counting every candidate
    against every entry, in one pass over the sample.
    """
    size = len(nums)
    lower = equal = 0
    below_n = below_d = above_n = above_d = None  # The largest entry below the value and the smallest above it.
    for n, d in zip(nums, dens):
        side = n * denominator - numerator * d  # The sign of n/d - value, as d and denominator are positive.
        if side < 0:
            lower += 1
            if below_n is None or n * below_d > below_n * d:
                below_n, below_d = n, d
        elif side > 0:
            if above_n is None or n * above_d < above_n * d:
                above_n, above_d = n, d
        else:
            equal += 1
    middle = size // 2
    if size % 2:
        return lower <= middle < lower + equal
    if equal:
        return lower <= middle - 1 and middle < lower + equal
    if lower != middle or size - lower != middle:
        return False
    return 2 * numerator * below_d * above_d == denominator * (below_n * above_d + above_n * below_d)


ARGUMENTS = {
    "mean": {"data"}, "median": {"data"}, "variance": {"data", "ddof"},
    "stddev": {"data", "ddof"}, "covariance": {"x", "y", "ddof"},
}
TYPES = {}


def produce(op, args):
    with computation():
        if op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]:
            raise InvalidInput("Unknown statistics operation or incorrect arguments")
        return globals()[op](**args), ()


def check_operation(op, args, value, evidence):
    """Recompute mathematical moments/order statistics without producer calls."""
    with computation():
        try:
            if (op not in ARGUMENTS or type(args) is not dict or set(args) != ARGUMENTS[op]
                    or type(evidence) is not tuple or evidence != ()):
                return False
            if op == "covariance":
                x, y = _sample(args["x"]), _sample(args["y"])
                if len(x) != len(y):
                    return False
            else:
                x = _sample(args["data"])
                y = x
            if op == "median":
                if type(value) is not Rational:
                    return False
                _rational(value)
                # Account for the order statistics by counts, independently of production sorting: one pass.
                current_budget().tick(len(x))
                return median_certificate([observation.numerator for observation in x],
                                          [observation.denominator for observation in x],
                                          value.numerator, value.denominator)
            total_x, total_y, products = ZERO, ZERO, ZERO
            for a, b in zip(x, y):
                current_budget().tick()
                total_x += a
                total_y += b
                products += a*b
            if op == "mean":
                target = total_x / Rational(len(x))
            else:
                denominator = _degrees(args["ddof"], len(x))
                target = (products - total_x * total_y / Rational(len(x))) / Rational(denominator)
            if op == "stddev":
                if type(value) is Rational:
                    _rational(value)
                    return value >= ZERO and value * value == target
                if type(value) is Sqrt:
                    return type(value.radicand) is Rational and _rational(value.radicand) == target and target >= ZERO
                return False
            return type(value) is Rational and _rational(value) == target
        except (InvalidInput, TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError):
            return False
