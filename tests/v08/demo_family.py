"""A throwaway v6 family for the JSON v6 core tests; production never registers it.

``registry._testing_families(demo_family)`` registers it for the duration of a with block. Each operation
takes one exact integer: demo_double doubles it, demo_defer always leaves the request unevaluated and
demo_crash fails inside the producer.
"""

from dataclasses import dataclass

from mathforge.contracts import OperationRequest
from mathforge.deferred import Reason, Unevaluated
from mathforge.errors import InvalidInput
from mathforge.registry import Argument, OperationSpec

from . import demo_checker


def _spec(name, summary):
    return OperationSpec(name, "discrete", (Argument("n", "int", None, "int"),), "int", summary, "tests only",
                         version=6, mcp=False)


SPECS = (_spec("demo_double", "Double an exact integer."),
         _spec("demo_defer", "Leave every request unevaluated."),
         _spec("demo_crash", "Fail inside the producer."))


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
    n = args["n"]
    if type(n) is not int:  # Neither bool nor float is an exact integer.
        raise InvalidInput("An exact integer is required")
    if operation == "demo_double":
        return 2 * n, ()
    if operation == "demo_defer":
        return Unevaluated("sum", OperationRequest(operation, tuple(args.items())), Reason("no_applicable_rule")), ()
    raise RuntimeError("demo_crash always fails")
