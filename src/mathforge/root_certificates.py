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


def check_root_certificate(polynomial, records, evidence):
    """Check the chain recurrence, source, isolator index and all multiplicities."""
    with computation():
        if type(evidence) is not tuple or len(evidence) != 2:
            return False
        chain, intervals = evidence
        if (type(chain) is not tuple or not chain or type(intervals) is not tuple
                or len(intervals) != len(records)
                or any(type(p) is not Polynomial or p.variable != polynomial.variable for p in chain)
                or polynomial.is_zero):
            return False
        square_free = polynomial.exact_div(polynomial.gcd(polynomial.derivative())).monic()
        if chain[0].is_zero or chain[0].leading_coefficient <= 0 or chain[0].monic() != square_free:
            return False
        if chain[0].degree == 0:
            if len(chain) != 1:
                return False
        else:
            if len(chain) < 2 or chain[1] != chain[0].derivative():
                return False
            for index in range(2, len(chain)):
                remainder = -chain[index - 2].divmod(chain[index - 1])[1]
                if remainder.is_zero or chain[index].is_zero:
                    return False
                scale = chain[index].leading_coefficient / remainder.leading_coefficient
                if scale <= 0 or chain[index] != remainder * scale:
                    return False
            if not chain[-2].divmod(chain[-1])[1].is_zero:
                return False
        if not verify_root_records(polynomial, records):
            return False
        for record, interval in zip(records, intervals):
            if not verify_isolator(record.root, interval):
                return False
            lower, upper = interval
            if lower == upper:
                if evaluate_rational(polynomial, lower).numerator != 0:
                    return False
            elif (evaluate_rational(polynomial, lower).numerator == 0
                  or evaluate_rational(polynomial, upper).numerator == 0
                  or count_real_roots(polynomial, lower, upper) != 1):
                return False
        return True
