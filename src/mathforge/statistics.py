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
                # Locate order statistics by counts, independently of production sorting.
                ranks = ((len(x)-1)//2, len(x)//2)
                selected = []
                for rank in ranks:
                    found = None
                    for candidate in x:
                        lower = equal = 0
                        for observation in x:
                            current_budget().tick()
                            lower += observation < candidate
                            equal += observation == candidate
                        if lower <= rank < lower + equal:
                            found = candidate
                            break
                    if found is None:
                        return False
                    selected.append(found)
                return value * Rational(2) == selected[0] + selected[1]
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
