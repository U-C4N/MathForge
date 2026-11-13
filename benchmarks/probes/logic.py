"""Logic probes: propositional formulas and the truth values that MathForge reports."""

from . import Probe

GROUP = "logic"

PROBES = (
    Probe("logic_truth_table", GROUP, "Write the truth table of p and not q.",
          lambda mf: mf.logic.truth_table("p and not q")),
    Probe("logic_simplify", GROUP, "Simplify (p and q) or (p and not q).",
          lambda mf: mf.logic.simplify_logic("(p and q) or (p and not q)")),
    Probe("logic_truth_values", GROUP, "Which truth values does MathForge use for a claim?",
          lambda mf: tuple(mf.Truth.__members__), lambda v: v == ("TRUE", "FALSE", "UNKNOWN")),
)
