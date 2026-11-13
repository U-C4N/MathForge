"""The symmetric difference of two finite unions of real intervals (mathforge.sets), with rational or real algebraic
endpoints.

    set_symmetric_difference(a, b)  a ^ b, the reals in exactly one of a and b

Arguments and value. Both arguments are canonical mathforge.intervals.IntervalSets, as the IntervalSet constructor
builds them: sorted, disjoint, not mergeable intervals, each nonempty, with open infinite ends and ends that are
canonical Rationals or RealAlgebraicRoots (a primitive square-free polynomial with a positive leading coefficient and
the index of a real root). The value is the canonical IntervalSet of the points in exactly one of a and b:
IntervalSet(()) when it is empty (a == b) and IntervalSet((Interval(None, None),)) for the whole line, never an
EmptySet or a UniversalSet. Each finite end of the value is a record of an input, the first that names its number (a
before b, in the order of the intervals, lower before upper), so the value brings no polynomial of its own, as the
checker requires: an end written under a polynomial that no input end uses is refused.

Evidence: (samples,), exactly as for the four operations of mathforge.areas.sets.interval_algebra: a strictly
increasing tuple of canonical Rationals rho_0 < ... < rho_r that certifies the cut of the inputs' ends. The r points of
that cut, in increasing order and each value once, are the Rational ends of a and b, the roots of their linear
RealAlgebraicRoot ends and every real root of the polynomial of each other RealAlgebraicRoot end, the roots that are no
end included (an end sqrt 2 makes -sqrt 2 a point too). rho_0 lies below the first point, rho_k strictly between the
points k and k + 1, rho_r above the last; without a point the evidence is ((0,),). The checker
(mathforge.certify.sets.symmetric) accepts any such samples; the rational of least denominator in each gap costs it
least, as the interval_algebra family's docstring explains.

The value has at most 2 * (len(a) + len(b)) intervals, a bound that the checker tests before it reads the claim (the
true count is at most len(a) + len(b), as the checker's docstring proves).

Algorithm. The line of points of mathforge.areas.sets.interval_algebra, reused as it is: the cut
of a ^ b is the cut of a | b, so this family takes that family's steps with the truth (x in a) != (x in b).

1. Validation: both arguments are IntervalSets by an exact type test (an EmptySet, a UniversalSet, a list, a
   subclass: InvalidInput), and every record is read once by interval_algebra._records (InvalidInput for a record
   that no constructor made, ResourceLimitError for an integer longer than max_integer_bits).
2. The line (interval_algebra._Line): every point of the cut in
   increasing order, each value once, with rational brackets that leave a rational between neighbours. The Sturm
   chain of each polynomial of degree two or more proves it square-free and its isolation names its real roots (an
   index beyond them: InvalidInput). Each input is written as the truth of each member of the line (points and open
   cells), and InvalidInput is raised unless its intervals are exactly the maximal runs of its true members, which
   proves it canonical.
3. Truth: (x in a) != (x in b), member by member; the value is the list of maximal runs of true members, each finite
   end written as the first input record at its point (_Line.assemble), passed once through the IntervalSet
   constructor, which leaves it unchanged.
4. Samples: the rational of least denominator in each gap (_Line.samples).
5. Bounds: interval_algebra._bound refuses with ResourceLimitError a claim that the default limits might not verify or
   reload, whatever the caller's limits. Its estimate of the check holds here unchanged: the checker of this family
   builds the cut of set_union(a, b) and runs the same cell engine on it, and the value enters the estimate only
   through its intervals and ends.

The card names IntervalSet.symmetric_difference as the producer. That method is unchecked sugar: it gives no samples
and goes through a difference and a union of its own. The family takes the line above instead, the same points as
set_union, as the checker's engineer advised. The helpers are private to interval_algebra; a change of their
signatures there has to be made here too.

Work. As in interval_algebra: produce() runs in the active budget (operations_v6.run opens the request's) or in one of
the default limits opened for a direct call, charges one unit for the call, and the reused helpers, trusted.upoly and
the kernels charge the rest.
"""

from __future__ import annotations

from fractions import Fraction

from ...errors import InvalidInput, UnsupportedOperation
from ...intervals import IntervalSet
from ...limits import computation
from ...registry import Argument, OperationSpec

# The producer rules:
# - Imports at the top come only from modules that `import mathforge` already loads; kernels are imported inside
#   produce(). Never import mathforge.certify, mathforge.lean, mathforge.mcp, mathforge._generated or a generated
#   area module (mathforge.ntheory, ...).
# - produce() validates its inputs first, with tests like `type(x) is int` (a bool, a float or a str is never an
#   exact number), and raises InvalidInput. It builds records only from validated data. For a Matrix argument it
#   calls current_budget().matrix(rows, cols) before it validates the shape.
# - The value is never None. The evidence is a tuple of ints, Rationals, tuples, str and records: no list, dict or
#   float. Under the verifier, 23 + 2 * request + 2 * value + evidence nodes stay within max_evidence_items.
# - Every operation takes at least one argument. Only an Expression or a Matrix that Result.expression should hold
#   is named `expression` or `A`.
# - Operation names are new, and nothing is superseded (no SUPERSEDES): tools/new_family.py checked these names.
# - No speed-suite fixture and no hand-written lean module: the generic lean runner serves v6 operations.

# The scope text of the sets area's v0.9 family (mathforge.areas.sets.interval_algebra): the symmetric difference
# works on the same objects, real interval sets with rational or real algebraic ends.
_SCOPE = ("exact rational polynomials, real algebraic numbers and real interval sets; certified roots, solution sets "
          "and decimal digits")
_SET = "IntervalSet"  # The annotation of the interval_set kind, and the return type of the operation.

SPECS = (
    OperationSpec("set_symmetric_difference", "sets",
                  (Argument("a", _SET, None, "interval_set"), Argument("b", _SET, None, "interval_set")),
                  _SET, "Take the symmetric difference of two finite unions of real intervals.", _SCOPE, version=6),
)

# No record classes: the value is a v0.7 IntervalSet, which the registry does not let a v6 family claim as its own
# (every TYPES class must be defined in the family module); it serializes and renders without one, so the family
# has no RENDER either, as mathforge.areas.sets.interval_algebra.
TYPES: dict[str, type] = {}
CHECKER = "mathforge.certify.sets.symmetric"

_OPERATION = "set_symmetric_difference"
_ARGUMENTS = ("a", "b")  # In the order of the spec.
_NOT_A_SET = _OPERATION + " requires an IntervalSet for {}"
_NOT_CANONICAL = _OPERATION + " requires a canonical IntervalSet for {}, as the IntervalSet constructor builds it"


def _symmetric(in_a: bool, in_b: bool) -> bool:
    """Whether a real lies in a ^ b, from whether it lies in a and in b."""
    return in_a != in_b


def produce(operation, args):
    """(value, evidence) of set_symmetric_difference, for arguments that operations_v6.run has checked by name and
    kind; the module states the algorithm, the value and the evidence. InvalidInput for an argument that is no
    canonical IntervalSet, ResourceLimitError past the budget or past the bound of interval_algebra."""
    if type(operation) is not str or operation != _OPERATION:
        raise UnsupportedOperation(f"{operation!r} is not an operation of the symmetric family")
    if type(args) is not dict or set(args) != set(_ARGUMENTS):
        raise InvalidInput(f"{_OPERATION} takes exactly the arguments {', '.join(_ARGUMENTS)}")
    sets = tuple(args[name] for name in _ARGUMENTS)
    for name, value in zip(_ARGUMENTS, sets):
        if type(value) is not IntervalSet:  # An EmptySet, a UniversalSet, a list, a subclass: no IntervalSet.
            raise InvalidInput(_NOT_A_SET.format(name))
    from . import interval_algebra as algebra  # The producer helpers of the line, loaded for a call only.

    messages = [_NOT_CANONICAL.format(name) for name in _ARGUMENTS]
    with computation() as budget:
        budget.tick()  # The call itself.
        records = tuple(algebra._records(budget, message, value) for message, value in zip(messages, sets))
        line = algebra._Line(budget, records, messages)
        truth = tuple(map(_symmetric, *line.members))
        value = line.assemble(truth)
        samples = line.samples()
        algebra._bound(budget, sets, value, line, tuple(Fraction(x.numerator, x.denominator) for x in samples))
        return value, (samples,)
