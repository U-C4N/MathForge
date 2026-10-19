"""A throwaway v6 family that supersedes integer_gcd; production never registers it.

``registry._testing_families(demo_dual_family)`` registers it for the duration of a with block. Its integer_gcd
keeps v0.7's arguments, defaults and public name. produce() declines a == 0, and every input that is not two exact
integers, so that the v0.7 implementation answers them with its own records and errors. demo_variable fails once
its arguments are validated (ruling 8).
"""

from math import gcd

from mathforge.errors import InvalidInput
from mathforge.operations_v6 import Declined
from mathforge.registry import Argument, OperationSpec
from mathforge.registry_v07 import V07_SPECS

from . import demo_dual_checker

V07_GCD = next(spec for spec in V07_SPECS if spec.name == "integer_gcd")

SPECS = (OperationSpec("integer_gcd", V07_GCD.area, V07_GCD.arguments, V07_GCD.returns, V07_GCD.summary,
                       "tests only", version=6),
         OperationSpec("demo_variable", "analysis",
                       (Argument("expression", "Expression", None, "scalar"), Argument("variable", "Symbol", None,
                                                                                        "symbol")),
                       "Expression", "Fail once the arguments are validated.", "tests only", version=6, mcp=False))
SUPERSEDES = frozenset({"integer_gcd"})
TYPES: dict[str, type] = {}
CHECKER = demo_dual_checker  # Production families name a module path below mathforge.certify instead.


def produce(operation, args):
    if operation == "demo_variable":
        raise InvalidInput("demo_variable rejects every request")
    a, b = args["a"], args["b"]
    if type(a) is not int or type(b) is not int or a == 0:
        raise Declined("The v0.7 implementation answers this input")
    return gcd(a, b), ()
