"""Independent checker of continued_fraction (family mathforge.areas.ntheory.continued_fraction).

It does not rerun the producer's Euclidean algorithm. It rebuilds the fraction from the claimed terms with
the forward recurrence of the convergents, p_k = a_k p_{k-1} + p_{k-2} and q_k = a_k q_{k-1} + q_{k-2},
starting from p_{-1} = 1, p_{-2} = 0, q_{-1} = 0, q_{-2} = 1. It accepts a canonical expansion whose last
convergent p/q equals the input. The record class is identified by name: a checker never imports producer code.
"""

from mathforge.limits import computation
from mathforge.model import Rational

RECORD = ("mathforge.areas.ntheory.continued_fraction", "ContinuedFraction")  # Module and name of the value's class.


def _canonical(terms):
    """At least one exact integer term; the later terms are positive and the last of several is at least 2."""
    return (type(terms) is tuple and len(terms) > 0 and all(type(term) is int for term in terms)
            and all(term >= 1 for term in terms[1:]) and (len(terms) == 1 or terms[-1] >= 2))


def check_operation(operation, args, value, evidence):
    source = args.get("value")
    if type(source) is int:
        source = Rational(source)
    if (operation != "continued_fraction" or evidence != () or type(source) is not Rational
            or (type(value).__module__, type(value).__qualname__) != RECORD or not _canonical(value.terms)):
        return False
    with computation() as budget:
        p, p_before, q, q_before = 1, 0, 0, 1
        for term in value.terms:
            budget.product(term, p)  # Forged terms cannot grow the products past the integer budget.
            budget.product(term, q)
            p, p_before = term * p + p_before, p
            q, q_before = term * q + q_before, q
        return Rational(p, q) == source
