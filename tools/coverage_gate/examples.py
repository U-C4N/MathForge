"""Reference examples for the coverage gate: SymPy API resolution, a sandboxed example runner and the harvest.

Depth, the eighth criterion of the gate, is measured on 10-20 representative SymPy 1.14 examples per inventory row, each
frozen with the output SymPy recorded for it. This module finds and records them:

- resolve_api turns a row's API path ("sympy.factorint", "Matrix.rref", "S.Half") into the SymPy object, and raises
  LookupError with the reason instead of guessing when a path is unqualified, missing or ambiguous;
- ExampleRunner runs examples in a persistent worker process with a fixed PYTHONHASHSEED: a fresh namespace and a cleared
  SymPy cache per example, a timeout after which the worker is replaced, and a reason instead of a value for what cannot
  serve as a reference;
- harvest collects a row's candidates from, in priority order, the docstrings of its APIs, the SymPy guide sources
  (doc/src, optional) and, only for a set that stays below ten, the equality asserts of SymPy's installed test modules;
  it excludes the plan's exclusions with their reasons and drops the selected examples whose record changes with the
  hash seed;
- select keeps one example per code and picks at most twenty in turn across the row's APIs; curate applies the
  curators' recorded replacements and numbers the result; load_examples and write_examples read and write the examples
  file (docs/v08/reference_examples.json, local and never committed).

SymPy is a development dependency only: this module imports it lazily, never at import time, and nothing under src/
imports this module. What it records depends neither on the hash seed nor on timing nor on how many workers run.
"""

import ast
import contextlib
import dataclasses
import doctest
import functools
import importlib
import inspect
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import types
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .inventory import Row

__all__ = ["EXCLUSION_REASONS", "DROP_REASONS", "SOURCE_PREFIXES", "KINDS", "LOW", "HIGH", "Example", "Excluded",
           "CurationError", "resolve_api", "unresolved_apis", "resolvable_api_paths", "ExampleRunner", "harvest",
           "examine", "select", "curate", "number", "load_curation", "load_examples", "write_examples", "kind_of"]

# Why a candidate cannot serve as a reference. The harvest report counts them in this order.
EXCLUSION_REASONS = ("printing", "plotting", "random", "optional-dependency", "raises", "timeout", "unstable",
                     "duplicate")
# The only reasons a curator may give for dropping a selected example. Difficulty is never one of them.
DROP_REASONS = ("not-the-subject",) + EXCLUSION_REASONS
SOURCE_PREFIXES = ("docstring:", "guide:", "tests:")  # In priority order.
KINDS = ("exact", "numeric")
LOW, HIGH = 10, 20  # A set below LOW is short; no set exceeds HIGH.
GUIDE_SUFFIXES = (".rst", ".md", ".txt")

_ROW_ID = re.compile(r"\d\d\.\d\d", re.ASCII)
_RUN_REASONS = frozenset({"printing", "optional-dependency", "raises", "timeout", "unstable"})
_STARTUP_TIMEOUT = 300.0  # Importing SymPy takes a second or two; a loaded machine gets much more.


@dataclass(frozen=True)
class Example:
    """A reference example and the value SymPy 1.14 recorded for it. Every field is a string, in written key order."""
    id: str      # "<row>/<nn>", e.g. "27.14/03"; empty until number() gives the final position
    row: str
    source: str  # "docstring:<API path>" | "guide:<relative path>:<line>" | "tests:<module>::<function>"
    setup: str   # Statements run after `from sympy import *`
    code: str    # The single expression whose value is recorded
    output: str  # str(value) in SymPy 1.14
    srepr: str   # srepr(value) in SymPy 1.14, Dummy indices renumbered from 0 in order of appearance
    kind: str    # "exact" | "numeric" (the value holds floating-point numbers)


@dataclass(frozen=True)
class Excluded:
    """A candidate that cannot serve as a reference, and why (one of EXCLUSION_REASONS)."""
    row: str
    source: str
    code: str
    reason: str


class CurationError(ValueError):
    """A curation file or entry that cannot be applied as written."""


# ----------------------------------------------------------------------------------------------------------------------
# API resolution

_MISSING = object()


def resolve_api(path: str) -> object:
    """The SymPy 1.14 object that an inventory API path names.

    A path is "sympy.<dotted name>" (an attribute of the sympy package, or of the module that the import system finds
    for the longest importable prefix) or "<Class>.<attribute>" for an attribute of a top-level sympy class or object
    ("Matrix.rref", "Poly.primitive", "S.Half"). LookupError, with a message that starts with the path and gives the
    reason, is raised for a bare name, a missing name, a module in the class position and an ambiguous path: one whose
    attribute chain and import system name different objects, as when a package re-exports a function under the name of
    its module. The resolver never guesses."""
    parts = path.split(".") if isinstance(path, str) else []
    if not parts or not all(part.isidentifier() for part in parts):
        raise LookupError(f"{path}: not a dotted name")
    import sympy
    if len(parts) == 1:
        raise LookupError(f"{path}: not qualified; write sympy.{path} for a top-level name or <Class>.{path} for a "
                          f"method or attribute{_spellings(sympy, path)}")
    if parts[0] == "sympy":
        return _resolve_in_package(sympy, path, parts)
    head = getattr(sympy, parts[0], _MISSING)
    if head is _MISSING:
        raise LookupError(f"{path}: sympy has no top-level name {parts[0]!r}")
    if isinstance(head, types.ModuleType):
        raise LookupError(f"{path}: {parts[0]} is a module; write sympy.{path}")
    return _walk(head, parts[1:], path, parts[0])


def _walk(value: object, names: Sequence[str], path: str, prefix: str) -> object:
    for name in names:
        found = getattr(value, name, _MISSING)
        if found is _MISSING:
            raise LookupError(f"{path}: {prefix} has no attribute {name!r}")
        value, prefix = found, f"{prefix}.{name}"
    return value


def _resolve_in_package(sympy: types.ModuleType, path: str, parts: Sequence[str]) -> object:
    try:
        attribute, attribute_error = _walk(sympy, parts[1:], path, "sympy"), None
    except LookupError as error:
        attribute, attribute_error = None, error
    imported = _MISSING
    for end in range(len(parts), 0, -1):  # The longest importable prefix, then attributes.
        try:
            module = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        with contextlib.suppress(LookupError):
            imported = _walk(module, parts[end:], path, ".".join(parts[:end]))
        break
    if attribute_error is not None:
        if imported is _MISSING:
            raise attribute_error
        return imported
    if imported is not _MISSING and imported is not attribute:
        raise LookupError(f"{path}: ambiguous: as an attribute it is {_describe(attribute)}, but importing it gives "
                          f"{_describe(imported)}; name the object through the module that defines it")
    return attribute


def _describe(value: object) -> str:
    if isinstance(value, types.ModuleType):
        return f"the module {value.__name__}"
    name = getattr(value, "__qualname__", None)
    module = getattr(value, "__module__", None)
    if isinstance(name, str) and isinstance(module, str):
        kind = "class" if isinstance(value, type) else "function" if callable(value) else "object"
        return f"the {kind} {module}.{name}"
    return f"the {type(value).__name__} {value!r}"


def _spellings(sympy: types.ModuleType, name: str) -> str:
    """The qualified paths under which a bare name exists, as a hint for the curator."""
    found = []
    value = getattr(sympy, name, _MISSING)
    if value is not _MISSING and not isinstance(value, types.ModuleType):
        found.append(f"sympy.{name}")
    for owner in sorted(entry for entry in dir(sympy) if not entry.startswith("_")):
        value = getattr(sympy, owner, None)
        if isinstance(value, type) and hasattr(value, name):
            found.append(f"{owner}.{name}")
    return f" (it exists as {', '.join(found)})" if found else ""


def unresolved_apis(row: Row) -> list[tuple[str, str]]:
    """The row's API paths that do not resolve, with the reason, in row order; a row without paths gives one entry with
    an empty path."""
    if not row.sympy_api_paths:
        return [("", "the row has no sympy_api_paths")]
    problems = []
    for path in row.sympy_api_paths:
        try:
            resolve_api(path)
        except LookupError as error:
            message = str(error)
            problems.append((path, message[len(path) + 2:] if message.startswith(f"{path}: ") else message))
    return problems


def resolvable_api_paths(row: Row) -> tuple[str, ...]:
    """The row's API paths that resolve, in row order: the APIs that harvest and select take turns across."""
    return tuple(api.path for api in _resolved(row))


@dataclass(frozen=True)
class _Api:
    index: int    # Position among the row's resolvable APIs
    path: str
    name: str     # The last component: what guide and test code must call or read as an attribute
    value: object


def _resolved(row: Row) -> list[_Api]:
    apis = []
    for path in row.sympy_api_paths:
        try:
            value = resolve_api(path)
        except LookupError:
            continue
        apis.append(_Api(len(apis), path, _api_name(path), value))
    return apis


def _api_name(path: str) -> str:
    return path.rpartition(".")[2]


# ----------------------------------------------------------------------------------------------------------------------
# The sandboxed runner

# The worker keeps the request and reply pipes on private descriptors and points descriptors 0 and 1 at the null device,
# so an example that prints, reads stdin or writes to descriptor 1 cannot corrupt the protocol. Before each example it
# clears SymPy's cache and resets the global state that examples are known to change, so that a record depends on its
# setup and code alone, never on the examples that ran before it in the same worker.
_WORKER = r'''
import builtins
import io
import json
import os
import random
import re
import signal
import sys
import warnings


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C belongs to the harvester, which stops its workers.
    requests = os.fdopen(os.dup(0), "rb")
    replies = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)

    def send(message):
        replies.write(json.dumps(message).encode("ascii") + b"\n")
        replies.flush()

    try:
        import mpmath
        import sympy
        from sympy.core import random as sympy_random
        from sympy.core.cache import clear_cache
        from sympy.core.parameters import global_parameters
        from sympy.printing.printer import Printer
    except Exception as error:
        send({"ready": False, "error": "%s: %s" % (type(error).__name__, error)})
        return
    home = os.getcwd()
    recursion = sys.getrecursionlimit()
    digits = sys.get_int_max_str_digits()
    address = re.compile(r" at 0x[0-9A-Fa-f]+")
    dummy = re.compile(r"dummy_index=(\d+)")
    optional = ("numpy", "scipy", "matplotlib")

    def reason(error):
        if isinstance(error, ImportError):
            missing = (getattr(error, "name", None) or "").partition(".")[0]
            if (missing and missing not in ("sympy", "mpmath")) or any(name in str(error) for name in optional):
                return "optional-dependency"
        return "raises"

    def reset():
        clear_cache()
        global_parameters.evaluate = True
        global_parameters.distribute = True
        global_parameters.exp_is_pow = False
        Printer._global_settings.clear()
        mpmath.mp.dps = 15
        mpmath.mp.pretty = False
        random.seed(0)
        sympy_random.seed(0)
        sys.setrecursionlimit(recursion)
        sys.set_int_max_str_digits(digits)
        os.chdir(home)
        sys.stdin = io.StringIO()

    def run(setup, code):
        try:
            reset()
        except BaseException:  # An earlier example broke the global state: this worker cannot run this one.
            return None
        printed = io.StringIO()
        sys.stdout = printed
        try:
            with warnings.catch_warnings():
                namespace = {"__name__": "__example__", "__builtins__": builtins}
                exec("from sympy import *", namespace)
                exec(compile(setup, "<setup>", "exec"), namespace)
                mark = printed.tell()
                value = eval(compile(code, "<code>", "eval"), namespace)
                if value is None or printed.tell() != mark:
                    return {"error": "printing"}
                output = str(value)
                representation = sympy.srepr(value)
                output.encode("utf-8")
                representation.encode("utf-8")
        except BaseException as error:
            return {"error": reason(error)}
        finally:
            sys.stdout = sys.__stdout__
        if address.search(output) or address.search(representation):
            return {"error": "unstable"}
        numbers = {}
        representation = dummy.sub(
            lambda match: "dummy_index=%d" % numbers.setdefault(match.group(1), len(numbers)), representation)
        return {"output": output, "srepr": representation}

    send({"ready": True, "sympy": sympy.__version__})
    for line in requests:
        try:
            request = json.loads(line)
            setup, code = request["setup"], request["code"]
        except Exception:
            send({"error": "raises"})
            continue
        reply = run(setup, code)
        if reply is None:
            send({"error": "restart"})
            return
        send(reply)


main()
'''


class ExampleRunner:
    """Runs examples in a persistent worker process, one at a time; not for use by several threads at once.

    The worker is `python -s -P -c <worker>`: no user site-packages, no script or working directory on sys.path, a
    fresh temporary working directory, and an environment without any PYTHON* variable except PYTHONHASHSEED=hash_seed.
    That is what -I would give, except that -I (through -E) ignores PYTHONHASHSEED itself. Each example runs after
    clear_cache() in a fresh namespace holding `from sympy import *`; the worker answers with (str(value),
    srepr(value)) or with a reason: "printing" (the value is None, or the code printed), "optional-dependency" (a
    module that is not installed, or numpy, scipy or matplotlib), "raises" (an exception, or a crash of the worker),
    "timeout" (no answer within timeout seconds) or "unstable" (the text holds a memory address). After a crash or a
    timeout the worker is killed and a new one starts with the next example. Use it as a context manager, or close()."""

    def __init__(self, *, hash_seed: int, timeout: float = 60.0) -> None:
        if isinstance(hash_seed, bool) or not isinstance(hash_seed, int) or not 0 <= hash_seed <= 4294967295:
            raise ValueError("hash_seed must be an integer from 0 to 4294967295")
        if not timeout > 0:
            raise ValueError("timeout must be a positive number of seconds")
        self.hash_seed = hash_seed
        self.timeout = float(timeout)
        self.sympy_version: str | None = None  # As the worker reports it
        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue | None = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._closed = False
        self._start()

    def __enter__(self) -> "ExampleRunner":
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def run(self, setup: str, code: str) -> tuple[str, str] | str:
        """(output, srepr) of the value of code after setup, or the reason why there is none."""
        if not isinstance(setup, str) or not isinstance(code, str):
            raise TypeError("setup and code must be strings")
        if self._closed:
            raise RuntimeError("the example runner is closed")
        request = json.dumps({"setup": setup, "code": code}).encode("ascii") + b"\n"
        result = self._exchange(request)
        if result == "restart":  # The worker was left broken by an earlier example: a fresh one runs this one.
            result = self._exchange(request)
        return "raises" if result == "restart" else result

    def _exchange(self, request: bytes) -> tuple[str, str] | str:
        try:
            self._send(request)
        except OSError:  # The worker died between examples, which is not this example's doing: start another one.
            self._stop()
            self._send(request)
        try:
            line = self._replies.get(timeout=self.timeout)
        except queue.Empty:
            self._stop()
            return "timeout"
        try:
            reply = json.loads(line) if line is not None else None
        except ValueError:
            reply = None
        if isinstance(reply, dict):
            if set(reply) == {"error"} and reply["error"] in _RUN_REASONS:
                return reply["error"]
            if set(reply) == {"output", "srepr"} and all(isinstance(value, str) for value in reply.values()):
                return reply["output"], reply["srepr"]
            if reply == {"error": "restart"}:
                self._stop()
                return "restart"
        self._stop()  # The worker died with the example (None) or broke the protocol: never trust it again.
        return "raises"

    def close(self) -> None:
        """Stop the worker; the runner cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError):
                process.stdin.close()  # End of input: the worker leaves its loop.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        self._stop()

    def _send(self, request: bytes) -> None:
        if self._process is None:
            self._start()
        self._process.stdin.write(request)
        self._process.stdin.flush()

    def _start(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="mathforge-examples-", ignore_cleanup_errors=True)
        environment = {name: value for name, value in os.environ.items() if not name.upper().startswith("PYTHON")}
        environment["PYTHONHASHSEED"] = str(self.hash_seed)
        try:
            process = subprocess.Popen([sys.executable, "-s", "-P", "-c", _WORKER], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=directory.name,
                                       env=environment)
        except BaseException:
            directory.cleanup()
            raise
        replies: queue.Queue = queue.Queue()
        threading.Thread(target=_pump, args=(process.stdout, replies), daemon=True,
                         name=f"example-worker-{process.pid}").start()
        self._process, self._replies, self._directory = process, replies, directory
        try:
            line = replies.get(timeout=_STARTUP_TIMEOUT)
        except queue.Empty:
            line = None
        try:
            ready = json.loads(line) if line is not None else None
        except ValueError:
            ready = None
        if not (isinstance(ready, dict) and ready.get("ready") is True):
            detail = ready.get("error") if isinstance(ready, dict) else "no answer"
            self._stop()
            raise RuntimeError(f"the example worker did not start ({detail}); {sys.executable} needs SymPy 1.14")
        self.sympy_version = ready.get("sympy")

    def _stop(self) -> None:
        process, directory = self._process, self._directory
        self._process = self._replies = self._directory = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            process.wait()
        except OSError:
            pass
        finally:
            for stream in (process.stdin, process.stdout):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
            directory.cleanup()


def _pump(stream, replies: queue.Queue) -> None:
    """Move the worker's reply lines to a queue, so that waiting for one can time out; None marks the end."""
    try:
        for line in stream:
            replies.put(line)
    except (OSError, ValueError):
        pass
    finally:
        replies.put(None)


# Results are memoized for the life of the process: rows that share an API share its candidates, and a record depends
# on (hash seed, timeout, setup, code) alone. A candidate that timed out is not retried for another row.
_RESULTS: dict[tuple[int, float, str, str], tuple[str, str] | str] = {}
_RESULTS_LOCK = threading.Lock()


def _execute(runner: ExampleRunner, setup: str, code: str) -> tuple[str, str] | str:
    key = (runner.hash_seed, runner.timeout, setup, code)
    with _RESULTS_LOCK:
        if key in _RESULTS:
            return _RESULTS[key]
    result = runner.run(setup, code)
    with _RESULTS_LOCK:
        return _RESULTS.setdefault(key, result)


# ----------------------------------------------------------------------------------------------------------------------
# Reading doctests: statements, groups and setup slicing

@dataclass(frozen=True)
class _Statement:
    """A statement that shows no output, with what the setup slicer needs to know about it."""
    text: str
    defines: frozenset[str]   # Names it binds
    kills: frozenset[str]     # Names it rebinds whatever their earlier value (plain assignment, import, def)
    mutates: frozenset[str]   # Names whose object it may change in place
    uses: frozenset[str]      # Names it reads
    always: bool              # Its effect on names is not known (star import, var(), exec()): always kept
    aliases: tuple[tuple[str, str], ...]  # (local name, imported name) for `import ... as` / `from ... import ... as`


@dataclass(frozen=True)
class _Group:
    """One candidate of a document: the statement with output and the statements without output before it."""
    line: int                         # 1-based line of the statement within the document
    code: str                         # The expression, or for an excluded group the statement as written
    pool: tuple[_Statement, ...]
    reason: str | None                # "raises" or "printing" when the document itself rules the group out
    names: frozenset[str]             # What the code calls or reads as an attribute, aliases resolved


class _Usage(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: set[str] = set()
        self.attributes: set[str] = set()
        self.loads: set[str] = set()
        self.stores: set[str] = set()
        self.bound: set[str] = set()   # Comprehension targets and lambda arguments, local to their expression
        self.modules: set[str] = set()
        self.strings: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.attributes.add(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        (self.loads if isinstance(node.ctx, ast.Load) else self.stores).add(node.id)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.bound.update(name.id for name in ast.walk(node.target) if isinstance(name, ast.Name))
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        arguments = node.args
        self.bound.update(argument.arg for argument in (*arguments.posonlyargs, *arguments.args,
                                                        *arguments.kwonlyargs, arguments.vararg, arguments.kwarg)
                          if argument is not None)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.modules.add(node.module or "")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.add(node.value)

    @property
    def free(self) -> set[str]:
        return self.loads - self.bound


def _usage(node: ast.AST) -> _Usage:
    usage = _Usage()
    usage.visit(node)
    return usage


def _root(node: ast.AST) -> str | None:
    """The name at the bottom of an attribute, subscript or call chain: M in M[0, 1], a in a.b.c(1)."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


_OPAQUE_CALLS = frozenset({"var", "exec", "eval", "globals", "locals", "vars", "setattr", "init_session"})
_INERT = (ast.Pass, ast.Assert, ast.Raise, ast.Return, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)


def _statement(node: ast.stmt, text: str | None) -> _Statement:
    """Describe a statement; text is the source it was parsed from (None: write it with ast.unparse)."""
    written = (ast.get_source_segment(text, node) if text is not None else None) or ast.unparse(node)
    usage = _usage(node)
    defines: set[str] = set()
    kills: set[str] = set()
    mutates: set[str] = set()
    uses = set(usage.free)
    aliases: list[tuple[str, str]] = []
    always = False

    def bind(target: ast.AST, rebinds: bool) -> None:
        if isinstance(target, ast.Name):
            defines.add(target.id)
            if rebinds:
                kills.add(target.id)
            else:
                uses.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element, rebinds)
        elif isinstance(target, ast.Starred):
            bind(target.value, rebinds)
        else:  # An attribute or a subscript changes the object of the name it starts from.
            root = _root(target)
            if root is not None:
                mutates.add(root)
                uses.add(root)

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.name == "*":
                always = True
                continue
            local = alias.asname or alias.name.partition(".")[0]
            defines.add(local)
            kills.add(local)
        aliases.extend(_import_aliases(node))
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            bind(target, True)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        bind(node.target, isinstance(node, ast.AnnAssign))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defines.add(node.name)
        kills.add(node.name)
        local = set(usage.stores)
        if not isinstance(node, ast.ClassDef):
            arguments = node.args
            local.update(argument.arg for argument in (*arguments.posonlyargs, *arguments.args,
                                                       *arguments.kwonlyargs, arguments.vararg, arguments.kwarg)
                         if argument is not None)
        uses -= local
    elif isinstance(node, ast.Expr):
        call = node.value
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name) and call.func.id in _OPAQUE_CALLS:
                always = True
            elif isinstance(call.func, ast.Attribute):  # A method call may change its receiver: M.row_del(0).
                root = _root(call.func.value)
                if root is not None:
                    mutates.add(root)
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            root = _root(target)
            if root is not None:
                mutates.add(root)
    elif not isinstance(node, _INERT):  # for, while, if, with, try, match: what binds or changes inside counts.
        bound, always = _bindings(node)
        defines.update(bound)
        mutates.update(_mutations(node))
    return _Statement(written, frozenset(defines), frozenset(kills), frozenset(mutates), frozenset(uses), always,
                      tuple(aliases))


def _mutations(node: ast.AST) -> set[str]:
    """The names whose objects a compound statement may change in place anywhere inside it: the roots of attribute
    and subscript targets (assigned, augmented or deleted) and the receivers of method calls. Merely reading a name,
    or passing it to a function, does not count."""
    roots: set[str] = set()

    def target(node: ast.AST) -> None:
        if isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                target(element)
        elif isinstance(node, ast.Starred):
            target(node.value)
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            root = _root(node)
            if root is not None:
                roots.add(root)

    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.Delete)):
            for item in child.targets:
                target(item)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            target(child.target)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            root = _root(child.func.value)
            if root is not None:
                roots.add(root)
    return roots


def _import_aliases(node: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
    """(local name, imported name) for each `as` of an import: cf_p for continued_fraction_periodic."""
    return [(alias.asname, alias.name.rpartition(".")[2]) for alias in node.names if alias.asname]


def _bindings(node: ast.AST) -> tuple[set[str], bool]:
    """Every name a compound statement may bind anywhere inside it (targets, def and class names, imports, except-as
    names), and whether it binds names that cannot be known (a star import, var(), exec(), ...)."""
    names: set[str] = set()
    unknown = False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                if alias.name == "*":
                    unknown = True
                else:
                    names.add(alias.asname or alias.name.partition(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in _OPAQUE_CALLS:
            unknown = True
    return names, unknown


def _slice(pool: Sequence[_Statement], needed: Iterable[str]) -> list[_Statement]:
    """The statements of pool that the names in needed depend on, in order: a backward pass that keeps what binds or
    changes a needed name, plus what cannot be analysed, and then needs what the kept statement reads."""
    needed = set(needed)
    kept = []
    for statement in reversed(pool):
        if statement.always or statement.defines & needed or statement.mutates & needed:
            kept.append(statement)
            needed -= statement.kills
            needed |= statement.uses
    kept.reverse()
    return kept


def _setup(pool: Sequence[_Statement], *codes: str) -> str:
    needed: set[str] = set()
    for code in codes:
        try:
            needed |= _usage(ast.parse(code, mode="eval")).free
        except SyntaxError:
            pass
    return "\n".join(statement.text for statement in _slice(pool, needed))


def _alias_map(statements: Iterable[tuple[tuple[str, str], ...] | _Statement]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in statements:
        for local, name in (statement.aliases if isinstance(statement, _Statement) else statement):
            aliases[local] = name
    return aliases


def _called(node: ast.AST, aliases: Mapping[str, str]) -> frozenset[str]:
    """The names that code calls or reads as an attribute, with an imported alias also standing for its original."""
    usage = _usage(node)
    used = usage.calls | usage.attributes
    return frozenset(used | {aliases[name] for name in used if name in aliases})


@functools.lru_cache(maxsize=65536)
def _attribution_names(code: str, setup: str) -> frozenset[str]:
    try:
        node = ast.parse(code, mode="eval")
    except SyntaxError:
        return frozenset()
    aliases: dict[str, str] = {}
    try:
        statements = ast.parse(setup).body
    except SyntaxError:
        statements = []
    for statement in statements:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                local = alias.asname or alias.name.partition(".")[0]
                if alias.asname:
                    aliases[local] = alias.name.rpartition(".")[2]
                else:
                    aliases.pop(local, None)
    return _called(node, aliases)


_FENCE = re.compile(r"[ \t]*(?:```|~~~)")


def _doctests(text: str) -> Iterator[tuple[int, str, str, bool]]:
    """(0-based line, source, wanted output, whether a traceback is wanted) for each doctest example of text. A
    markdown code fence that doctest reads as output is not output. A malformed example spoils doctest's parse of the
    whole text, so the text is then parsed paragraph by paragraph and only the bad paragraphs are lost."""
    parser = doctest.DocTestParser()
    try:
        found = [(0, parser.get_examples(text))]
    except ValueError:
        found = []
        lines = text.split("\n")
        start = None
        for index, line in enumerate([*lines, ""]):
            if line.strip():
                start = index if start is None else start
            elif start is not None:
                with contextlib.suppress(ValueError):
                    found.append((start, parser.get_examples("\n".join(lines[start:index]))))
                start = None
    for offset, examples in found:
        for item in examples:
            want = "\n".join(line for line in item.want.split("\n") if not _FENCE.match(line)).strip()
            yield offset + item.lineno, item.source, want, item.exc_msg is not None


def _groups(text: str) -> tuple[_Group, ...]:
    """Doctest grouping: the statements without output up to the next statement with output form one example. The
    statement with output must be an expression: it becomes the code, and the statements before it in the document form
    the pool from which its setup is sliced. A wanted traceback rules a group out as "raises", output of a statement
    that is not an expression (a loop that prints) as "printing"; such a statement still binds its names for later
    groups. Statements that show output are never part of a later setup."""
    pool: list[_Statement] = []
    groups: list[_Group] = []
    for line, source, want, raised in _doctests(text):
        try:
            body = ast.parse(source).body
        except (SyntaxError, ValueError):
            if want:
                groups.append(_Group(line + 1, source.strip(), tuple(pool), "raises", frozenset()))
            continue
        if not body:
            continue
        statements = [_statement(node, source) for node in body]
        if not want:
            pool.extend(statements)
            continue
        last = body[-1]
        pool.extend(statements[:-1])
        snapshot = tuple(pool)
        expression = isinstance(last, ast.Expr)
        if raised:
            reason = "raises"
        elif not expression:
            reason = "printing"
            pool.append(statements[-1])
        else:
            reason = None
        node = last.value if expression else last
        code = (ast.get_source_segment(source, node) or ast.unparse(node)).strip()
        groups.append(_Group(line + last.lineno, code, snapshot, reason, _called(node, _alias_map(snapshot))))
    return tuple(groups)


@functools.lru_cache(maxsize=4096)
def _docstring_groups(text: str) -> tuple[_Group, ...]:
    return _groups(text)


def _docstring(value: object) -> str | None:
    try:
        text = inspect.getdoc(value)
    except Exception:
        return None
    return text if isinstance(text, str) and ">>>" in text else None


_GUIDES: dict[Path, tuple[tuple[str, tuple[_Group, ...]], ...]] = {}
_GUIDES_LOCK = threading.Lock()


def _guide_documents(doc_src: Path) -> tuple[tuple[str, tuple[_Group, ...]], ...]:
    """(relative POSIX path, groups) for every guide page with doctests, in path order; parsed once per process. The
    pages are only read as text: nothing under doc_src is imported or executed here."""
    root = Path(doc_src).resolve()
    with _GUIDES_LOCK:
        if root not in _GUIDES:
            if not root.is_dir():
                raise FileNotFoundError(f"{doc_src}: the guide sources are not a directory")
            pages = sorted((path.relative_to(root).as_posix(), path) for path in root.rglob("*")
                           if path.suffix.lower() in GUIDE_SUFFIXES and path.is_file())
            documents = []
            for relative, path in pages:
                text = path.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
                if ">>>" in text:
                    documents.append((relative, _groups(text)))
            _GUIDES[root] = tuple(documents)
        return _GUIDES[root]


# ----------------------------------------------------------------------------------------------------------------------
# SymPy's installed test modules

@dataclass(frozen=True)
class _Assert:
    """`assert <code> == <right>` at the top level of a test function."""
    path: Path
    module: str
    function: str
    line: int
    code: str
    right: str
    names: frozenset[str]


def _sympy_package_directory() -> Path:
    import sympy
    return Path(sympy.__file__).resolve().parent


_TESTS: dict[Path, dict[str, tuple[_Assert, ...]]] = {}
_TESTS_LOCK = threading.Lock()


def _test_index(package: Path) -> Mapping[str, tuple[_Assert, ...]]:
    """Equality asserts of every <package>/**/tests/test_*.py, by each name their left side calls or reads as an
    attribute; built once per process."""
    with _TESTS_LOCK:
        if package not in _TESTS:
            index: dict[str, list[_Assert]] = {}
            modules = sorted((path.relative_to(package).as_posix(), path) for path in package.rglob("test_*.py")
                             if path.parent.name == "tests")
            for relative, path in modules:
                module = ".".join((package.name, *relative[:-len(".py")].split("/")))
                for item in _module_asserts(path, module):
                    for name in item.names:
                        index.setdefault(name, []).append(item)
            _TESTS[package] = {name: tuple(items) for name, items in index.items()}
        return _TESTS[package]


def _is_test(node: ast.stmt) -> bool:
    return isinstance(node, ast.FunctionDef) and node.name.startswith("test")


def _equality(node: ast.stmt) -> tuple[ast.expr, ast.expr] | None:
    if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)):
        return node.test.left, node.test.comparators[0]
    return None


def _expression_text(lines: Sequence[str], node: ast.expr) -> str:
    """The expression as written when it fits on one line, else ast.unparse. (ast.get_source_segment would split the
    whole module again for every call, which makes indexing SymPy's tests quadratic.)"""
    if node.end_lineno == node.lineno and node.end_col_offset is not None:
        written = lines[node.lineno - 1].encode("utf-8")[node.col_offset:node.end_col_offset].decode("utf-8")
        if written.strip() == written and written:
            return written
    return ast.unparse(node)


def _module_asserts(path: Path, module: str) -> list[_Assert]:
    try:
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.split("\n")
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            aliases.update(_import_aliases(node))
    found = []
    for function in tree.body:
        if not _is_test(function):
            continue
        local = dict(aliases)
        for node in function.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                local.update(_import_aliases(node))
            sides = _equality(node)
            if sides is None:
                continue
            names = _called(sides[0], local)
            if names:
                found.append(_Assert(path, module, function.name, node.lineno, _expression_text(lines, sides[0]),
                                     _expression_text(lines, sides[1]), names))
    return found


@functools.lru_cache(maxsize=64)
def _test_module_pools(path: Path) -> tuple[tuple[_Statement, ...], dict[tuple[str, int], tuple[_Statement, ...]]]:
    """The module-level statements other than tests, and for each (test function, assert line) the statements of the
    function before that assert."""
    tree = ast.parse(path.read_bytes().decode("utf-8"))
    module_pool = tuple(_statement(node, None) for node in tree.body if not _is_test(node))
    before: dict[tuple[str, int], tuple[_Statement, ...]] = {}
    for function in tree.body:
        if not _is_test(function):
            continue
        statements: list[_Statement] = []
        for node in function.body:
            if _equality(node) is not None:
                before[(function.name, node.lineno)] = tuple(statements)
            elif not isinstance(node, ast.Assert):
                statements.append(_statement(node, None))
    return module_pool, before


def _assert_setup(item: _Assert) -> str:
    module_pool, before = _test_module_pools(item.path)
    return _setup(module_pool + before.get((item.function, item.line), ()), item.code, item.right)


def _home_module(value: object) -> str:
    if isinstance(value, property) and value.fget is not None:
        value = value.fget
    module = getattr(value, "__module__", None)
    return module if isinstance(module, str) else type(value).__module__


# ----------------------------------------------------------------------------------------------------------------------
# Static exclusion rules

_PRINTING = frozenset({
    "print", "pprint", "pretty", "pretty_print", "pprint_use_unicode", "pprint_try_use_unicode", "upretty",
    "latex", "print_latex", "multiline_latex", "mathml", "print_mathml", "str", "repr", "ascii", "format", "sstr",
    "sstrrepr", "srepr", "python", "print_python", "dotprint", "print_tree", "preview", "init_printing", "doprint",
    "ccode", "print_ccode", "cxxcode", "fcode", "print_fcode", "jscode", "print_jscode", "julia_code", "octave_code",
    "rcode", "print_rcode", "rust_code", "mathematica_code", "maple_code", "glsl_code", "print_glsl", "pycode",
    "smtlib_code", "__str__", "__repr__",
})
_RANDOM = frozenset({"choice", "choices", "shuffle", "sample", "sample_iter", "sample_stochastic_process", "seed",
                     "uniform", "gauss", "getrandbits", "urandom", "token_hex", "token_bytes"})
_DETERMINISTIC = frozenset({"random_symbols"})  # Lists the random variables of an expression: no randomness.
_OPTIONAL_MODULES = frozenset({"numpy", "scipy", "matplotlib"})
_OPTIONAL_NAMES = _OPTIONAL_MODULES | {"np", "plt", "pyplot"}


def _is_random(name: str) -> bool:
    return name not in _DETERMINISTIC and (name in _RANDOM or name.startswith("rand"))


def _is_plotting(name: str) -> bool:
    return name.lower().startswith("plot") or name in ("textplot", "PlotGrid")


def _random_subject(row: Row) -> bool:
    """Randomness is the subject of a row whose own APIs generate random values (sample, randprime, ...)."""
    return any(_is_random(_api_name(path)) for path in row.sympy_api_paths)


def _static_reason(setup: str, code: str, random_allowed: bool) -> str | None:
    """The exclusion that setup and code show without running: printing (the code calls a printer or formatter),
    plotting, random (unless randomness is the row's subject) or optional-dependency (numpy, scipy, matplotlib). Code
    that is not one expression, or a setup that does not parse, would raise."""
    try:
        code_usage = _usage(ast.parse(code, mode="eval"))
        setup_usage = _usage(ast.parse(setup))
    except (SyntaxError, ValueError):
        return "raises"
    names = set()
    modules = set()
    strings = set()
    loads = set()
    for usage in (code_usage, setup_usage):
        names |= usage.calls | usage.attributes
        loads |= usage.loads
        modules |= {part for module in usage.modules for part in module.split(".")}
        strings |= usage.strings
    if code_usage.calls & _PRINTING:
        return "printing"
    if any(_is_plotting(name) for name in names | loads | modules):
        return "plotting"
    if not random_allowed and (any(_is_random(name) for name in names) or "random" in loads | modules):
        return "random"
    if (loads | names) & _OPTIONAL_NAMES or modules & _OPTIONAL_MODULES or strings & _OPTIONAL_MODULES:
        return "optional-dependency"
    return None


def _code_key(code: str) -> str:
    """Codes that differ only in formatting are the same code."""
    try:
        return ast.dump(ast.parse(code, mode="eval"))
    except (SyntaxError, ValueError):
        return code.strip()


def kind_of(representation: str) -> str:
    """"numeric" when an srepr holds a floating-point number (Float, mpf or mpc, or a Python float or complex), else
    "exact"."""
    try:
        tree = ast.parse(representation, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):  # Not Python, or too deep for the parser
        return "numeric" if re.search(r"\b(?:Float|mpf|mpc)\(", representation) else "exact"
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
            return "numeric"
        if isinstance(node, ast.Call) and _callee(node.func) in ("Float", "mpf", "mpc"):
            return "numeric"
        if isinstance(node, ast.Name) and node.id in ("inf", "nan"):
            return "numeric"
    return "exact"


def _callee(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


# ----------------------------------------------------------------------------------------------------------------------
# The harvest

@dataclass(frozen=True)
class _Candidate:
    order: tuple            # (source tier, API index, position in the document)
    source: str
    setup: str
    code: str
    reason: str | None      # Ruled out by the document itself


def harvest(row: Row, runner: ExampleRunner, *, doc_src: Path | None,
            check_runner: ExampleRunner | None = None) -> tuple[list[Example], list[Excluded]]:
    """The row's candidates and its exclusions.

    Candidates come from the docstrings of the row's resolvable APIs (in row order), then from the guide pages under
    doc_src (attributed to the first API, in row order, whose name the code calls or reads as an attribute), each in
    document order; a candidate's setup is sliced from the statements before it to those its code needs. Every
    candidate is ruled out by the document, by the static rules or by running it at runner's hash seed, or becomes
    usable; one of each code is kept. The examples that select() would pick are then run again by check_runner (another
    hash seed; a temporary one with runner's seed + 1 when None) and excluded as "unstable" when their record changes,
    until the selection is stable. A selection below LOW is topped up to LOW from the equality asserts of SymPy's
    installed test modules: the left side is the code, and an assert counts only when its right side's output is the
    same, the asserts taken in turn across the APIs and only as far as needed.

    The returned candidates are in source order and carry no id yet; select(candidates,
    apis=resolvable_api_paths(row)) picks exactly the stability-checked examples."""
    if check_runner is None:
        with ExampleRunner(hash_seed=(runner.hash_seed + 1) % 4294967296, timeout=runner.timeout) as check:
            return harvest(row, runner, doc_src=doc_src, check_runner=check)
    if check_runner.hash_seed == runner.hash_seed:
        raise ValueError("the check runner must use another hash seed")
    apis = _resolved(row)
    random_allowed = _random_subject(row)
    usable: list[Example] = []
    excluded: list[Excluded] = []
    codes: set[str] = set()
    for candidate in sorted(_document_candidates(apis, doc_src), key=lambda candidate: candidate.order):
        reason = candidate.reason or _static_reason(candidate.setup, candidate.code, random_allowed)
        key = _code_key(candidate.code)
        if reason is None and key in codes:
            reason = "duplicate"
        result = None
        if reason is None:
            result = _execute(runner, candidate.setup, candidate.code)
            if isinstance(result, str):
                reason = result
        if reason is not None:
            excluded.append(Excluded(row.id, candidate.source, candidate.code, reason))
            continue
        codes.add(key)
        usable.append(Example("", row.id, candidate.source, candidate.setup, candidate.code, result[0], result[1],
                              kind_of(result[1])))
    paths = tuple(api.path for api in apis)
    checked: set[int] = set()
    while True:
        chosen = select(usable, apis=paths)
        pending = [item for item in chosen if id(item) not in checked]
        if not pending:
            break
        for item in pending:
            if _execute(check_runner, item.setup, item.code) == (item.output, item.srepr):
                checked.add(id(item))
            else:
                excluded.append(Excluded(row.id, item.source, item.code, "unstable"))
                usable.remove(item)
    if apis and len(chosen) < LOW:
        usable.extend(_from_test_modules(row, apis, runner, check_runner, codes, LOW - len(chosen), excluded,
                                         random_allowed))
    return usable, excluded


def _document_candidates(apis: Sequence[_Api], doc_src: Path | None) -> Iterator[_Candidate]:
    seen: set[str] = set()
    for api in apis:
        text = _docstring(api.value)
        if text is None or text in seen:  # Two paths of one object share its docstring: harvest it once.
            continue
        seen.add(text)
        for group in _docstring_groups(text):
            yield _candidate(group, (0, api.index, group.line), f"docstring:{api.path}")
    if doc_src is None:
        return
    names = {api.name for api in apis}
    for relative, groups in _guide_documents(doc_src):
        for group in groups:
            if not group.names & names:
                continue
            candidate = _candidate(group, (), f"guide:{relative}:{group.line}")
            index = (_first_api(group.names, apis) if candidate.reason is not None
                     else _first_api(_attribution_names(candidate.code, candidate.setup), apis))
            if index is not None:
                yield dataclasses.replace(candidate, order=(1, index, relative, group.line))


def _candidate(group: _Group, order: tuple, source: str) -> _Candidate:
    setup = "" if group.reason is not None else _setup(group.pool, group.code)
    return _Candidate(order, source, setup, group.code, group.reason)


def _first_api(names: frozenset[str], apis: Sequence[_Api]) -> int | None:
    return next((api.index for api in apis if api.name in names), None)


def _from_test_modules(row: Row, apis: Sequence[_Api], runner: ExampleRunner, check_runner: ExampleRunner,
                       codes: set[str], need: int, excluded: list[Excluded], random_allowed: bool) -> list[Example]:
    index = _test_index(_sympy_package_directory())
    queues = []
    for api in apis:
        # The test module of the API's own module comes first, then the rest of its package's tests, then all others.
        package, _, leaf = _home_module(api.value).rpartition(".")
        ordered = []
        for item in index.get(api.name, ()):
            if _first_api(item.names, apis) == api.index:
                rank = (0 if item.module == f"{package}.tests.test_{leaf}"
                        else 1 if item.module.startswith(f"{package}.tests.") else 2)
                ordered.append(((api.index, rank, item.module, item.line), item))
        ordered.sort(key=lambda pair: pair[0])
        queues.append((api, iter(ordered)))
    taken: list[tuple[tuple, Example]] = []
    active = list(range(len(queues)))
    while active and len(taken) < need:
        for position in list(active):
            if len(taken) >= need:
                break
            api, items = queues[position]
            found = _next_test_example(row, api, items, apis, runner, check_runner, codes, excluded, random_allowed)
            if found is None:
                active.remove(position)
            else:
                taken.append(found)
    return [item for _, item in sorted(taken, key=lambda pair: pair[0])]


def _next_test_example(row: Row, api: _Api, items: Iterator[tuple[tuple, _Assert]], apis: Sequence[_Api],
                       runner: ExampleRunner, check_runner: ExampleRunner, codes: set[str], excluded: list[Excluded],
                       random_allowed: bool) -> tuple[tuple, Example] | None:
    """The next assert of the API's queue that yields a usable, stable example, with its order; None when the queue is
    exhausted. Asserts it rules out are recorded in excluded, asserts SymPy does not confirm are skipped."""
    for order, item in items:
        setup = _assert_setup(item)
        if _first_api(_attribution_names(item.code, setup), apis) != api.index:
            continue
        source = f"tests:{item.module}::{item.function}"
        reason = _static_reason(setup, item.code, random_allowed)
        key = _code_key(item.code)
        if reason is None and key in codes:
            reason = "duplicate"
        left = None
        if reason is None:
            left = _execute(runner, setup, item.code)
            if isinstance(left, str):
                reason = left
        if reason is not None:
            excluded.append(Excluded(row.id, source, item.code, reason))
            continue
        right = _execute(runner, setup, item.right)
        if isinstance(right, str) or right[0] != left[0]:
            continue  # SymPy does not confirm the equality as written: not a candidate.
        if _execute(check_runner, setup, item.code) != left:
            excluded.append(Excluded(row.id, source, item.code, "unstable"))
            continue
        codes.add(key)
        return order, Example("", row.id, source, setup, item.code, left[0], left[1], kind_of(left[1]))
    return None


def examine(row: Row, source: str, setup: str, code: str, runner: ExampleRunner,
            check_runner: ExampleRunner) -> Example | Excluded:
    """One candidate through the same rules as a harvested one: the static rules, a run at runner's hash seed and a
    second run at check_runner's, which must give the same record. The example has no id yet."""
    reason = _static_reason(setup, code, _random_subject(row))
    result = None
    if reason is None:
        result = _execute(runner, setup, code)
        if isinstance(result, str):
            reason = result
        elif _execute(check_runner, setup, code) != result:
            reason = "unstable"
    if reason is not None:
        return Excluded(row.id, source, code, reason)
    return Example("", row.id, source, setup, code, result[0], result[1], kind_of(result[1]))


# ----------------------------------------------------------------------------------------------------------------------
# Selection

def select(candidates: Sequence[Example], *, low: int = LOW, high: int = HIGH,
           apis: Sequence[str] = ()) -> list[Example]:
    """At most high candidates, in candidate order.

    One candidate is kept per code (formatting aside), the first. Docstring and guide candidates are taken in turn
    across the APIs: the first of each API, then the second of each, and so on. A docstring candidate belongs to the
    API its source names; a guide or test candidate to the first of apis (the row's API paths, in row order) whose last
    component its code calls or reads as an attribute (an imported alias stands for its original), else to its own
    document. The APIs take their turns in the order of apis, then in order of first appearance. Test-module
    candidates only top up a selection below low, again in turn, up to low. A result below low marks a short row."""
    if not 0 < low <= high:
        raise ValueError("select needs 0 < low <= high")
    seen: set[str] = set()
    unique: list[Example] = []
    for item in candidates:
        key = _code_key(item.code)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    main = [item for item in unique if not item.source.startswith("tests:")]
    picked = _rotate(_by_api(main, apis), high)
    if len(picked) < low:
        picked += _rotate(_by_api([item for item in unique if item.source.startswith("tests:")], apis),
                          low - len(picked))
    chosen = {id(item) for item in picked}
    return [item for item in unique if id(item) in chosen]


def _by_api(items: Sequence[Example], apis: Sequence[str]) -> list[list[Example]]:
    groups: dict[str, list[Example]] = {path: [] for path in apis}
    for item in items:
        groups.setdefault(_api_key(item, apis), []).append(item)
    return [group for group in groups.values() if group]


def _api_key(item: Example, apis: Sequence[str]) -> str:
    if item.source.startswith("docstring:"):
        return item.source[len("docstring:"):]
    names = _attribution_names(item.code, item.setup)
    for path in apis:
        if _api_name(path) in names:
            return path
    return item.source.rpartition(":")[0].rstrip(":")  # Its document: guide:<path> or tests:<module>.


def _rotate(groups: Sequence[Sequence[Example]], limit: int) -> list[Example]:
    picked: list[Example] = []
    depth = 0
    while len(picked) < limit and any(depth < len(group) for group in groups):
        for group in groups:
            if depth < len(group) and len(picked) < limit:
                picked.append(group[depth])
        depth += 1
    return picked


# ----------------------------------------------------------------------------------------------------------------------
# Curation

def load_curation(path: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Read a curation file: an object keyed by row id whose values hold "drop" ([{"source", "code", "reason"}], the
    reason one of DROP_REASONS) and "add" ([{"source", "setup", "code"}]) lists. CurationError names the file and the
    problem."""
    path = Path(path)
    try:
        document = json.loads(path.read_bytes().decode("utf-8-sig"))
        return _checked_curation(document)
    except ValueError as error:
        raise CurationError(f"{path}: {error}") from error


def _checked_curation(document: object) -> dict[str, dict[str, list[dict[str, str]]]]:
    if not isinstance(document, dict):
        raise CurationError("a curation file is an object keyed by row id")
    entries = {}
    for row_id, entry in document.items():
        if not _ROW_ID.fullmatch(row_id):
            raise CurationError(f"{row_id!r} is not a row id")
        if not isinstance(entry, dict) or not set(entry) <= {"drop", "add"}:
            raise CurationError(f"{row_id}: an entry holds only 'drop' and 'add' lists")
        drops = _curation_items(row_id, "drop", entry.get("drop", []), ("source", "code", "reason"))
        adds = _curation_items(row_id, "add", entry.get("add", []), ("source", "setup", "code"))
        for drop in drops:
            if drop["reason"] not in DROP_REASONS:
                raise CurationError(f"{row_id}: drop reason {drop['reason']!r} is not allowed; the reasons are "
                                    f"{', '.join(DROP_REASONS)}")
        for item in drops + adds:
            if not item["source"].startswith(SOURCE_PREFIXES):
                raise CurationError(f"{row_id}: source {item['source']!r} does not start with "
                                    f"{', '.join(SOURCE_PREFIXES)}")
        for add in adds:
            try:
                ast.parse(add["code"], mode="eval")
            except SyntaxError:
                raise CurationError(f"{row_id}: added code {add['code']!r} is not one expression") from None
            try:
                ast.parse(add["setup"])
            except SyntaxError:
                raise CurationError(f"{row_id}: added setup {add['setup']!r} does not parse") from None
        entries[row_id] = {"drop": drops, "add": adds}
    return entries


def _curation_items(row_id: str, name: str, items: object, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not isinstance(items, list):
        raise CurationError(f"{row_id}: {name} must be a list")
    for item in items:
        if not isinstance(item, dict) or set(item) != set(fields) or not all(isinstance(item[key], str)
                                                                             for key in fields):
            raise CurationError(f"{row_id}: each {name} entry has exactly the string fields {', '.join(fields)}")
    return [dict(item) for item in items]


def curate(row: Row, selected: Sequence[Example], entry: Mapping[str, Sequence[Mapping[str, str]]] | None,
           runner: ExampleRunner, check_runner: ExampleRunner, *,
           high: int = HIGH) -> tuple[list[Example], list[tuple[str, str, str, str]]]:
    """Apply a curation entry to the row's selection and number the result.

    Each drop removes the selected example with its source and code (formatting aside); one that matches none is an
    error. Each addition goes through examine() and is appended; an addition that would be excluded, or that repeats
    a kept code, is an error, and so is a result above high. Returns the numbered examples and the notes (action,
    source, code, reason or output) for the report."""
    kept = list(selected)
    notes: list[tuple[str, str, str, str]] = []
    entry = entry or {}
    for drop in entry.get("drop", ()):
        target = next((item for item in kept
                       if item.source == drop["source"] and _code_key(item.code) == _code_key(drop["code"])), None)
        if target is None:
            raise CurationError(f"{row.id}: drop {drop['source']} {drop['code']!r} matches no selected example")
        kept.remove(target)
        notes.append(("drop", drop["source"], drop["code"], drop["reason"]))
    for add in entry.get("add", ()):
        if _code_key(add["code"]) in {_code_key(item.code) for item in kept}:
            raise CurationError(f"{row.id}: added example {add['code']!r} repeats the code of a kept example")
        result = examine(row, add["source"], add["setup"], add["code"], runner, check_runner)
        if isinstance(result, Excluded):
            raise CurationError(f"{row.id}: added example {add['source']} {add['code']!r} is excluded: "
                                f"{result.reason}")
        kept.append(result)
        notes.append(("add", add["source"], add["code"], result.output))
    if len(kept) > high:
        raise CurationError(f"{row.id}: {len(kept)} examples after curation; at most {high}")
    return number(row.id, kept), notes


def number(row_id: str, examples: Sequence[Example]) -> list[Example]:
    """The examples with their final ids, <row>/01, <row>/02, ... in order."""
    return [dataclasses.replace(item, id=f"{row_id}/{position:02d}") for position, item in enumerate(examples, 1)]


# ----------------------------------------------------------------------------------------------------------------------
# The examples file

_FIELDS = tuple(field.name for field in dataclasses.fields(Example))


def load_examples(path: Path) -> dict[str, list[Example]]:
    """Read an examples file: {"rows": {row id: [example, ...]}}. Every example has exactly the Example fields, all
    strings; its row is the row it is listed under and its id is <row>/<nn>, unique in the file; its kind is exact or
    numeric and its source starts with docstring:, guide: or tests:. ValueError names the file and the problem."""
    path = Path(path)
    try:
        document = json.loads(path.read_bytes().decode("utf-8-sig"))
        if not isinstance(document, dict) or set(document) != {"rows"} or not isinstance(document["rows"], dict):
            raise ValueError("an examples file is an object with one 'rows' object")
        loaded: dict[str, list[Example]] = {}
        seen: set[str] = set()
        for row_id, items in document["rows"].items():
            if not isinstance(items, list):
                raise ValueError(f"row {row_id}: expected a list of examples")
            examples = []
            for position, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != set(_FIELDS):
                    raise ValueError(f"row {row_id}: example #{position} must have exactly the fields "
                                     f"{', '.join(_FIELDS)}")
                if not all(isinstance(item[name], str) for name in _FIELDS):
                    raise ValueError(f"row {row_id}: every field of example #{position} must be a string")
                examples.append(Example(**{name: item[name] for name in _FIELDS}))
            _check_examples(row_id, examples, seen)
            loaded[row_id] = examples
        return loaded
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _check_examples(row_id: object, examples: Sequence[Example], seen: set[str]) -> None:
    if not isinstance(row_id, str) or not _ROW_ID.fullmatch(row_id):
        raise ValueError(f"{row_id!r} is not a row id")
    for item in examples:
        if not isinstance(item, Example):
            raise ValueError(f"row {row_id}: {item!r} is not an Example")
        if not all(isinstance(getattr(item, name), str) for name in _FIELDS):
            raise ValueError(f"row {row_id}: every field of {item.id!r} must be a string")
        if item.row != row_id:
            raise ValueError(f"row {row_id}: example {item.id!r} belongs to row {item.row!r}")
        if not re.fullmatch(rf"{re.escape(row_id)}/\d\d", item.id):
            raise ValueError(f"row {row_id}: example id {item.id!r} is not {row_id}/<nn>")
        if item.id in seen:
            raise ValueError(f"row {row_id}: duplicate example id {item.id}")
        seen.add(item.id)
        if item.kind not in KINDS:
            raise ValueError(f"row {row_id}: example {item.id} has kind {item.kind!r}, not exact or numeric")
        if not item.source.startswith(SOURCE_PREFIXES):
            raise ValueError(f"row {row_id}: example {item.id} has source {item.source!r}")


def write_examples(path: Path, examples: Mapping[str, Sequence[Example]]) -> None:
    """Write the canonical file: rows in id order, each example's fields in Example field order, indent 1, UTF-8 without
    a BOM, "\\n" line ends and a closing newline. The examples are checked as load_examples checks them, and the text
    goes to a temporary file that then replaces the target, so a failure never leaves half a file."""
    path = Path(path)
    seen: set[str] = set()
    rows = {}
    for row_id in sorted(examples):
        items = list(examples[row_id])
        _check_examples(row_id, items, seen)
        rows[row_id] = [dataclasses.asdict(item) for item in items]
    text = json.dumps({"rows": rows}, ensure_ascii=False, indent=1) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:  # Bytes, so Windows does not turn "\n" into "\r\n".
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
