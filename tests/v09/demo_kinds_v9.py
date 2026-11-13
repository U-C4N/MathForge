"""A throwaway v6 family that echoes one argument of each kind added in v0.9; production never registers it.

``registry._testing_families(demo_kinds_v9)`` registers it for the duration of a with block. Unlike the v0.8 demo
families, its operations are exposed over MCP: demo_echo_polynomial and demo_echo_interval_set run through
mathforge.algebra, and demo_echo_relation through mathforge.equations, a tool that no production operation serves
yet. Every operation returns its one argument unchanged and records its name in CALLS, so that a test can tell
whether a request reached the producer.
"""

import sys

from mathforge.registry import Argument, OperationSpec

SPECS = (
    OperationSpec("demo_echo_polynomial", "algebra", (Argument("polynomial", "Polynomial", None, "polynomial"),),
                  "Polynomial", "Return a polynomial unchanged.", "tests only", version=6),
    OperationSpec("demo_echo_interval_set", "sets", (Argument("intervals", "IntervalSet", None, "interval_set"),),
                  "IntervalSet", "Return a union of real intervals unchanged.", "tests only", version=6),
    OperationSpec("demo_echo_relation", "equations", (Argument("relation", "str", "'='", "relation"),),
                  "str", "Return a relation unchanged.", "tests only", version=6),
)
TYPES: dict[str, type] = {}
CHECKER = sys.modules[__name__]  # Only a testing family may check itself.
CALLS: list[str] = []


def produce(operation, args):
    CALLS.append(operation)
    (value,) = args.values()  # run() has checked the argument names: each operation takes exactly one.
    return value, ()


def check_operation(operation, args, value, evidence):
    return evidence == () and (value,) == tuple(args.values())
