"""A throwaway v6 family for the JSON v6 core tests; production never registers it.

``registry._testing_families(demo_family)`` registers it for the duration of a with block. Most operations
take one exact integer: demo_double doubles it, demo_defer always leaves the request unevaluated, and the
others each break the producer contract in one way. demo_echo and demo_rows take the arguments from which
run() copies a result's source fields.
"""

from dataclasses import dataclass

from mathforge.contracts import OperationRequest
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import ResourceLimitError
from mathforge.model import Symbol
from mathforge.registry import Argument, OperationSpec

from . import demo_checker

N = Argument("n", "int", None, "int")


def _spec(name, summary, arguments=(N,), returns="int"):
    return OperationSpec(name, "discrete", arguments, returns, summary, "tests only", version=6, mcp=False)


SPECS = (_spec("demo_double", "Double an exact integer."),
         _spec("demo_defer", "Leave every request unevaluated."),
         _spec("demo_crash", "Fail inside the producer."),
         _spec("demo_float", "Return a float, which no result may hold."),
         _spec("demo_misdefer", "Leave another request unevaluated."),
         _spec("demo_unsupported", "Declare every request out of scope."),
         _spec("demo_exhausted", "Exceed the computation budget."),
         _spec("demo_echo", "Return an expression unchanged.",
               (Argument("expression", "Expression", None, "scalar"), Argument("variable", "Symbol", None, "symbol")),
               "Expression"),
         _spec("demo_rows", "Count the rows of a matrix.", (Argument("A", "Matrix", None, "matrix"),)))


@dataclass(frozen=True)
class DemoPair:
    a: int
    b: int

    def __post_init__(self):
        if type(self.a) is not int or type(self.b) is not int:
            raise InvalidInput("A demo pair holds two exact integers")


TYPES = {"demo_pair": DemoPair}
CHECKER = demo_checker  # Production families name a module path below mathforge.certify instead.


def produce(operation, args):
    if operation == "demo_echo":
        if type(args["variable"]) is not Symbol or args["variable"].assumptions:
            raise InvalidInput("demo_echo takes a variable without assumptions")
        return args["expression"], ()
    if operation == "demo_rows":
        return len(args["A"].rows), ()
    n = args["n"]
    if type(n) is not int:  # Neither bool nor float is an exact integer.
        raise InvalidInput("An exact integer is required")
    if operation == "demo_double":
        return 2 * n, ()
    if operation == "demo_defer":
        return Unevaluated("sum", OperationRequest(operation, tuple(args.items())), Reason("no_applicable_rule")), ()
    if operation == "demo_misdefer":  # The record names another request than the one run() made.
        return Unevaluated("sum", OperationRequest(operation, (("n", n + 1),)), Reason("no_applicable_rule")), ()
    if operation == "demo_float":
        return n / 2, ()
    if operation == "demo_unsupported":
        raise UnsupportedOperation("demo_unsupported applies to no request")
    if operation == "demo_exhausted":
        raise ResourceLimitError("demo_exhausted always exceeds the budget")
    raise RuntimeError("demo_crash always fails")
