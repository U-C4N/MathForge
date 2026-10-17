"""Finite simple continued fractions of exact rationals: the pilot v6 family and the template for the others.

A v6 family module declares:

- SPECS, its operations. tools/generate_api.py turns each one into a wrapper of mathforge.operations and a
  value function of its area module, here mathforge.ntheory.continued_fraction.
- TYPES, the record classes of its values by tag. They are defined in this module and serialize as "v6_" + tag.
- CHECKER, the import path of its independent checker below mathforge.certify. The checker never imports
  this module.
- produce(operation, args), which returns (value, evidence) for a request whose argument names
  operations_v6.run has already checked. run() calls it inside the request's computation budget.
  produce() validates its inputs first and raises InvalidInput for those it rejects, here anything but an
  exact int or Rational. Once the inputs are validated, an InvalidInput raised while building a record is a
  producer bug that run() cannot tell from rejected input, so a family builds records only from validated
  data (the checker re-validates the canonical form anyway). Any other exception, apart from
  UnsupportedOperation and ResourceLimitError, counts as the producer's internal error.
- RENDER, optionally: {record class: render(value, renderer, child) -> str}, the signature of
  rendering_v5.render. renderer.latex selects LaTeX; renderer.label, parens, join and integer apply the shared
  budget; child(nested) renders a nested value with the aliases and the budget of the whole rendering. The
  registry checks at build time that the keys are classes of this module's TYPES and the values callables.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import InvalidInput
from ...limits import computation
from ...model import Rational
from ...registry import Argument, OperationSpec


SPECS = (OperationSpec("continued_fraction", "ntheory", (Argument("value", "Rational | int", None, "rational"),),
                       "ContinuedFraction", "Return the finite simple continued fraction of an exact rational.",
                       "exact rationals; canonical finite simple continued fraction", version=6),)


@dataclass(frozen=True)
class ContinuedFraction:
    """[a0; a1, ..., an] = a0 + 1/(a1 + 1/(... + 1/an)), in the canonical form that each rational has exactly
    once: a0 is any integer, every later term is positive, and the last of several terms is at least 2."""
    terms: tuple[int, ...]

    def __post_init__(self):
        with computation() as budget:
            if type(self.terms) is not tuple or not self.terms:
                raise InvalidInput("A continued fraction has a nonempty tuple of terms")
            for index, term in enumerate(self.terms):
                budget.tick()
                if type(term) is not int:  # Neither bool nor float is an exact integer.
                    raise InvalidInput("Continued fraction terms are exact integers")
                budget.check_int(term)
                if index and term < 1:
                    raise InvalidInput("Continued fraction terms after the first are positive")
            if len(self.terms) > 1 and self.terms[-1] < 2:
                raise InvalidInput("The last of several continued fraction terms is at least 2")


TYPES = {"continued_fraction": ContinuedFraction}
CHECKER = "mathforge.certify.ntheory.continued_fraction"


def produce(operation, args):
    """The Euclidean algorithm with floor division on p/q. The checker needs no evidence."""
    value = args["value"]
    if type(value) is int:
        p, q = value, 1
    elif type(value) is Rational:
        p, q = value.numerator, value.denominator
    else:  # A float, a bool or an expression is not an exact rational.
        raise InvalidInput("continued_fraction requires an exact rational: a Rational or an int")
    terms = []
    with computation() as budget:
        while True:
            budget.tick()
            term, remainder = divmod(p, q)  # q > 0, so term = floor(p / q) and 0 <= remainder < q.
            terms.append(term)
            if not remainder:
                return ContinuedFraction(tuple(terms)), ()
            p, q = q, remainder


def _render(fraction, renderer, child):
    """[4; 2, 6, 7], or [5] for a single term; the same text is LaTeX math source."""
    head, *tail = (child(term) for term in fraction.terms)
    return "[" + head + ("; " + renderer.join(tail, ", ") if tail else "") + "]"


RENDER = {ContinuedFraction: _render}
