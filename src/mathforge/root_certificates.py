"""Persisted signed Sturm chains and source-linked exact root isolators."""

from .algebraic import as_endpoint, interval_for, _polynomial
from .limits import computation
from .model import Rational
from .polynomial import Polynomial
from .root_isolation import (
    count_real_roots, evaluate_rational, refine_interval, sturm_sequence,
    verify_isolator, verify_root_records,
)


def root_certificate(polynomial, records):
    with computation() as budget:
        intervals = []
        for record in records:
            point = as_endpoint(record.root)
            lower, upper = interval_for(point)
            while lower != upper and (
                    evaluate_rational(polynomial, lower).numerator == 0
                    or evaluate_rational(polynomial, upper).numerator == 0
                    or count_real_roots(polynomial, lower, upper) != 1):
                budget.refine()
                lower, upper = refine_interval(_polynomial(point), (lower, upper))
            intervals.append((lower, upper))
        evidence = (sturm_sequence(polynomial), tuple(intervals))
        budget.inspect(evidence, evidence=True)
        return evidence


def _signed_primitive(values):
    """Integer coefficients divided by their positive content (the sign kept); () for zero."""
    from math import gcd
    if not values:
        return ()
    divisor = gcd(*values)
    return tuple(value // divisor for value in values)


def check_root_certificate(polynomial, records, evidence):
    """Check the chain recurrence, source, isolator index and all multiplicities.

    v0.7's predicate, decided on integers by mathforge.certify.upoly: the first member is a positive multiple of the
    square-free part, the second its derivative, each further member a positive multiple of minus the remainder of
    the two before it (pseudo-remainders with the sign of lc**k undone), the last remainder zero; then the records
    (verify_root_records) and each isolator (verify_isolator, the source polynomial nonzero at open ends and with one
    root inside)."""
    from .certify import upoly as certify
    from .trusted import upoly as tp
    with computation():
        if type(evidence) is not tuple or len(evidence) != 2:
            return False
        chain, intervals = evidence
        if (type(chain) is not tuple or not chain or type(intervals) is not tuple
                or len(intervals) != len(records)
                or any(type(p) is not Polynomial or p.variable != polynomial.variable for p in chain)
                or polynomial.is_zero):
            return False
        source = tp.primitive(tp.from_rationals(polynomial.coefficients)[0])
        square_free = certify.squarefree_part(source)
        if chain[0].is_zero or chain[0].leading_coefficient <= 0:
            return False
        members = [tp.from_rationals(member.coefficients)[0] for member in chain]
        if tp.primitive(members[0]) != square_free:
            return False
        if chain[0].degree == 0:
            if len(chain) != 1:
                return False
        else:
            first = chain[0].coefficients
            if len(chain) < 2 or chain[1].coefficients != (tuple(value * index for index, value in enumerate(first)
                                                                    if index) or (Rational(0),)):
                return False
            for index in range(2, len(chain)):
                a, b, c = members[index - 2], members[index - 1], members[index]
                if not b or not c:
                    return False
                remainder = tp.pseudo_remainder(a, b)
                if not remainder:
                    return False
                if b[-1] > 0 or (len(a) - len(b) + 1) % 2 == 0:
                    remainder = tp.neg(remainder)
                if _signed_primitive(c) != _signed_primitive(remainder):
                    return False
            if not members[-1] or tp.pseudo_remainder(members[-2], members[-1]):
                return False
        if not verify_root_records(polynomial, records):
            return False
        for record, interval in zip(records, intervals):
            if not verify_isolator(record.root, interval):
                return False
            lower, upper = interval
            if lower == upper:
                if tp.sign_at(source, lower.numerator, lower.denominator) != 0:
                    return False
            elif (tp.sign_at(source, lower.numerator, lower.denominator) == 0
                  or tp.sign_at(source, upper.numerator, upper.denominator) == 0
                  or certify.count_roots(square_free, lower, upper) != 1):
                return False
        return True
