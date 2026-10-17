"""Fail when an independent checker can reach producer code through imports.

Modules of the packages in CHECKER_PACKAGES may import from mathforge only the packages and modules in
ALLOWED; imports from outside mathforge (standard library, third party) are not layers and are
ignored. The search is transitive: it starts from every checker module and follows imports through
every allowed module (the checker packages, mathforge.trusted, mathforge.errors, mathforge.limits and
mathforge.model), so a forbidden import hidden in any of them is reported against every checker that
reaches it. Names are matched on whole dotted components: mathforge.modelling is not mathforge.model.

Every import statement counts, including those inside functions, try blocks and TYPE_CHECKING
blocks. The top-level mathforge/__init__.py is never searched, but importing a name from it is itself
a violation. Calls to importlib.import_module and __import__ are not followed; instead,
tests/v08/test_layering.py fails when a checker module names importlib, __import__ or sys.modules at all.

Run `python tools/check_layering.py`; it exits with status 1 and prints each offending import path.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
import sys


TOP_PACKAGE = "mathforge"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / TOP_PACKAGE
CHECKER_PACKAGES = ("mathforge.certify",)
ALLOWED = ("mathforge.certify", "mathforge.trusted", "mathforge.errors", "mathforge.limits", "mathforge.model")


def within(name: str, packages: tuple[str, ...]) -> bool:
    """Whether a dotted name is one of the packages or lies below one (whole components only)."""
    return any(name == package or name.startswith(package + ".") for package in packages)


def module_files(package_root: Path) -> dict[str, Path]:
    """Dotted module name -> source file for every .py file in the tree (a package by its __init__)."""
    modules: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*.py")):
        parts = list(path.relative_to(package_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        # Sorting puts pkg/__init__.py before pkg.py; the import system also prefers the package.
        modules.setdefault(".".join([TOP_PACKAGE, *parts]), path)
    return modules


def imported_names(path: Path, module: str) -> list[str]:
    """Absolute dotted names that one source file imports; a name may be a module or an attribute."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                drop = node.level - 1  # the first dot is the package itself, every further dot is a parent
                if drop >= len(anchor):
                    raise ValueError(f"{path}:{node.lineno}: relative import beyond the top-level package")
                base = ".".join([*anchor[:len(anchor) - drop], *([node.module] if node.module else [])])
            names.extend(base if alias.name == "*" else f"{base}.{alias.name}" for alias in node.names)
    return names


def executed_modules(name: str, modules: dict[str, Path]) -> list[str]:
    """Allowed modules that importing `name` runs: the __init__ of each parent package, then the module.

    The top-level mathforge/__init__.py, the public facade that imports everything, is not allowed,
    so it is never searched.
    """
    parts = name.split(".")
    steps = (".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return [step for step in steps if within(step, ALLOWED) and step in modules]


def path_to(reached: dict[str, str | None], module: str) -> list[str]:
    path = []
    current: str | None = module
    while current is not None:
        path.append(current)
        current = reached[current]
    return path[::-1]


def find_violations(package_root: Path) -> dict[tuple[str, str], list[str]]:
    """Map (checker module, forbidden module) to the shortest import path from the first to the second."""
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise FileNotFoundError(f"No package directory: {package_root}")
    modules = module_files(package_root)
    imports: dict[str, list[str]] = {}
    found: dict[tuple[str, str], list[str]] = {}
    for checker in sorted(name for name in modules if within(name, CHECKER_PACKAGES)):
        reached: dict[str, str | None] = {checker: None}  # module -> the module whose import led here
        queue = deque([checker])
        while queue:
            module = queue.popleft()
            if module not in imports:
                imports[module] = imported_names(modules[module], module)
            for name in imports[module]:
                if not within(name, (TOP_PACKAGE,)):
                    continue  # standard library and third-party imports are not layers
                if not within(name, ALLOWED):
                    found.setdefault((checker, name), [*path_to(reached, module), name])
                    continue
                for step in executed_modules(name, modules):
                    if step not in reached:
                        reached[step] = module
                        queue.append(step)
    return found


def violations(package_root: Path) -> list[tuple[str, str]]:
    """Sorted (checker module, forbidden target module) pairs; an empty list means the layering holds."""
    return sorted(find_violations(package_root))


def main(package_root: Path = PACKAGE_ROOT) -> None:
    found = find_violations(package_root)
    if not found:
        print("No layering violations")
        return
    lines = ["Layering violations: independent checkers (" + ", ".join(CHECKER_PACKAGES)
             + ") may import only " + ", ".join(ALLOWED) + ".",
             "Each line is an import path from a checker module to a forbidden module:"]
    lines.extend("  " + " -> ".join(found[pair]) for pair in sorted(found))
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
