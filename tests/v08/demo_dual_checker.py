"""The independent checker of the demo dual family: a common divisor that a recomputed Bezout pair reaches.

The greatest common divisor of a and b is the one nonnegative common divisor that is an integer combination of
a and b. The checker finds a Bezout pair itself, by the extended Euclidean algorithm, never through the producer.
"""


def _bezout(a, b):
    """(x, y) with a*x + b*y == gcd(a, b) up to sign."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        quotient, remainder = divmod(a, b)
        a, b = b, remainder
        x0, x1 = x1, x0 - quotient * x1
        y0, y1 = y1, y0 - quotient * y1
    return x0, y0


def check_operation(operation, args, value, evidence):
    if operation != "integer_gcd" or evidence != ():
        return False
    a, b = args.get("a"), args.get("b")
    if type(a) is not int or type(b) is not int or type(value) is not int or value < 0:
        return False
    if value == 0:
        return a == 0 and b == 0
    if a % value or b % value:
        return False
    x, y = _bezout(a, b)
    return abs(a * x + b * y) == value
