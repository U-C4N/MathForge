"""REL-06/07/08/09/10/12: Test artifacts from outside the source checkout.

Run ``python -m build`` first, then ``python tools/check_distribution.py``.
Both wheel and sdist are installed into fresh dependency-free runtime venvs.
Each artifact also receives a separate optional-MCP installation and real stdio
integration check unless --core-only is explicitly requested.
The sdist is rebuilt by pip's isolated PEP 517 build before installation.
"""

from __future__ import annotations

import argparse
import configparser
import email.parser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
PROBE = r'''
from importlib import metadata, resources
from pathlib import Path
import importlib.util
import sys
import mathforge as mf
checkout, expected_version = Path(sys.argv[1]).resolve(), sys.argv[2]
installed = Path(mf.__file__).resolve()
assert not installed.is_relative_to(checkout / "src"), installed
assert mf.__version__ == expected_version == metadata.version("mathforge")
assert resources.files("mathforge").joinpath("py.typed").is_file()
for dependency in ("sympy", "numpy", "scipy", "sage", "mcp", "pydantic"):
    assert importlib.util.find_spec(dependency) is None, dependency
x = mf.symbol("x")
result = mf.solve(mf.Eq(x**3 - 2, 0), for_=x)
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert mf.verify(result).verified
assert mf.loads(mf.dumps(result)) == result
assert mf.det(mf.Matrix(((1, 2), (3, 4)))) == mf.Rational(-2)
f = mf.parse_expression("((x-1)*(x^2+2))/(x*(x-1))", symbols={"x": x})
assert mf.limit(f, x, 1) == mf.LimitValue("finite", mf.Rational(3))
for new_result in (mf.operations.limit(f, x, 0), mf.operations.analyze_function(f, x)):
    assert new_result.execution_status is mf.ExecutionStatus.COMPLETED
    assert mf.verify(new_result).verified
    assert mf.to_data(new_result)["version"] == 3
    assert mf.loads(mf.dumps(new_result)) == new_result
assert resources.files("mathforge").joinpath("schemas/mathforge-v3.schema.json").is_file()
assert resources.files("mathforge").joinpath("schemas/mathforge-v4.schema.json").is_file()
assert resources.files("mathforge").joinpath("schemas/mathforge-v5.schema.json").is_file()
assert resources.files("mathforge").joinpath("schemas/mathforge-v6.schema.json").is_file()
factor = mf.operations.factor(mf.Polynomial((4, 0, 0, 0, 1), x))
assert mf.verify(factor).verified and mf.to_data(factor)["version"] == 4
assert mf.loads(mf.dumps(factor)) == factor
alpha = mf.RealAlgebraicRoot((-2, 0, 0, 1), 0)
assert mf.algebraic(alpha**3) == mf.Rational(2)
assert mf.integrate(x, x, 0, mf.sqrt(2)) == mf.Rational(1)
assert mf.charpoly(mf.Matrix(((0, 2), (1, 0))), x) == mf.Polynomial((-2, 0, 1), x)
from mathforge import functions as fn
for module in ("number_theory", "combinatorics", "complex_numbers", "probability", "statistics",
               "linear_extensions", "functions", "function_calculus", "function_limits", "function_integration", "rational_integration",
               "dispatch_v5", "operations_v5", "verification_v5", "api_v7", "values_v7"):
    assert importlib.util.find_spec("mathforge." + module) is not None, module
new_results = (mf.operations.is_prime(65537), mf.operations.crt(((2, 6), (8, 9))),
    mf.operations.multinomial((2, 3, 1)), mf.operations.complex_sqrt(mf.ComplexAlgebraic(3, 4)),
    mf.operations.binomial_distribution(4, mf.Rational(1, 3)), mf.operations.variance((1, 2, 4)),
    mf.operations.lu(mf.Matrix(((0, 2), (1, 3)))),
    mf.operations.function_differentiate(mf.real_function(fn.sin(x), (x,)), x),
    mf.operations.function_integrate(mf.real_function(fn.sin(x), (x,)), x, 0, 1),
    mf.operations.function_limit(mf.real_function(fn.sin(x)/fn.as_real_expr(x), (x,)), x, 0),
    mf.operations.rational_antiderivative(mf.rational_function(1, x**5-x+1, variable=x), x))
for result in new_results:
    assert result.execution_status is mf.ExecutionStatus.COMPLETED, result.error
    assert mf.verify(result).verified
    assert mf.to_data(result)["version"] == 5
    restored = mf.loads(mf.dumps(result))
    assert restored == result and mf.verify(restored).verified
    assert mf.render_text(restored) and mf.to_latex(restored)
deferred = mf.operations.function_antiderivative(mf.real_function(fn.sin(fn.as_real_expr(x)**2), (x,)), x)
assert deferred.outcome is mf.Outcome.NO_CONCLUSION and deferred.value is None
assert deferred.deferred is not None and not mf.verify(deferred).verified
assert mf.loads(mf.dumps(deferred)) == deferred
print("Isolated installed package:", installed)
'''

MCP_PROBE = r'''
from importlib import metadata
from pathlib import Path
import mathforge as mf
import mcp
import sys
assert not Path(mf.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve() / "src")
assert mf.__version__ == metadata.version("mathforge") == sys.argv[2]
assert metadata.version("mcp").split(".", 1)[0] == "2"
x = mf.symbol("x")
assert mf.verify(mf.solve(mf.Eq(x**2, 2), for_=x)).verified
# The 30 v0.7 tools lead, unchanged; the area tools with operations and the catalog follow.
from mathforge import registry
from mathforge.mcp.boundary import _FIELDS
areas = sorted(tool for tool in registry.MCP_TOOLS if registry.tool_operations(tool))
tools = [tool["name"] for tool in mf.capabilities()["tools"]]
assert len(_FIELDS) == 30 and tools == ["mathforge." + name for name in _FIELDS] + areas + ["mathforge.catalog"]
assert mf.verify(mf.operations.analyze_function(x**3, x)).verified
'''


def active_dependencies(requirements, *, extra="") -> list[str]:
    """Evaluate PEP 508 markers in the development harness, not the base venv."""
    from packaging.requirements import Requirement

    return [raw for raw in requirements
            if (requirement := Requirement(raw)).marker is None
            or requirement.marker.evaluate({"extra": extra})]


def command(arguments, *, cwd, environment):
    subprocess.run([str(argument) for argument in arguments], cwd=cwd,
                   env=environment, check=True)


def check_archives(wheel: Path, sdist: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert not any("docs" in Path(name).parts for name in names), "Wheel contains local docs"
        assert "mathforge/py.typed" in names, "Wheel does not include py.typed"
        for module in (REPOSITORY / "src" / "mathforge").rglob("*.py"):
            relative = module.relative_to(REPOSITORY / "src").as_posix()
            assert relative in names, f"Missing wheel module: {relative}"
        for schema in (REPOSITORY / "src" / "mathforge" / "schemas").glob("*.json"):
            assert f"mathforge/schemas/{schema.name}" in names, f"Missing wheel schema: {schema.name}"
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        package_metadata = email.parser.Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        assert package_metadata["Version"] == expected_version
        assert package_metadata["Requires-Python"] == ">=3.11"
        entrypoint_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(archive.read(entrypoint_name).decode("utf-8"))
        assert entrypoints["console_scripts"]["mathforge-mcp"] == "mathforge.mcp:main"
        requirements = package_metadata.get_all("Requires-Dist") or []
        assert not active_dependencies(requirements), "Default runtime dependencies were added"
        assert "mcp" in (package_metadata.get_all("Provides-Extra") or []), "Missing MCP extra"
        from packaging.requirements import Requirement
        optional = [Requirement(raw) for raw in active_dependencies(requirements, extra="mcp")]
        assert len(optional) == 1 and optional[0].name == "mcp", "Unexpected MCP dependencies"
        assert str(optional[0].specifier) == "<3,>=2.3", "Unexpected supported MCP SDK range"
    with tarfile.open(sdist, "r:gz") as archive:
        # Inspect member names without extracting a potentially unsafe archive.
        names = {name.partition("/")[2] for name in archive.getnames()}
        assert not any(name == "docs" or name.startswith("docs/") for name in names), \
            "Sdist contains local docs"
        for directory, patterns in (("assets", ("*.svg",)),
                                    ("tests", ("*.py", "*.json")),
                                    ("examples", ("*.py",)),
                                    ("tools", ("*.py",)),
                                    ("benchmarks", ("*.py", "*.json")),
                                    ("documentation", ("*.md", "*.json")),
                                    ("schemas", ("*.json",))):
            for pattern in patterns:
                for source in (REPOSITORY / directory).rglob(pattern):
                    relative = source.relative_to(REPOSITORY).as_posix()
                    assert relative in names, f"Missing sdist input: {relative}"
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "CHANGELOG.md", "LICENSE", "assets/mark.svg",
                     "requirements-dev.txt", "requirements-mcp.txt"):
            assert name in names, f"Missing sdist input: {name}"


def isolated_check(artifact: Path, expected_version: str, *, skip_tests: bool, mcp_extra=False) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="mathforge-artifact-") as temporary:
        directory = Path(temporary)
        environment_path = directory / "environment"
        venv.EnvBuilder(with_pip=True).create(environment_path)
        python = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        harness = directory / "harness"
        harness.mkdir()
        for name in ("assets", "tests", "examples", "benchmarks", "schemas", "tools", "documentation"):
            source = REPOSITORY / name
            if source.is_dir():
                shutil.copytree(source, harness / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "CHANGELOG.md", "requirements-mcp.txt"):
            shutil.copy2(REPOSITORY / name, harness / name)
        if mcp_extra:
            command((python, "-I", "-m", "pip", "install", "-c", harness / "requirements-mcp.txt",
                     f"{artifact}[mcp]"), cwd=harness, environment=environment)
            command((python, "-I", "-c", MCP_PROBE, REPOSITORY, expected_version),
                    cwd=harness, environment=environment)
            command((python, "-I", harness / "tools/check_mcp.py"), cwd=harness, environment=environment)
            command((python, "-I", harness / "tools/check_docs.py", "--mcp"),
                    cwd=harness, environment=environment)
            return
        command((python, "-I", "-m", "pip", "install", "--no-deps", artifact), cwd=harness, environment=environment)
        command((python, "-I", "-c", PROBE, REPOSITORY, expected_version),
                cwd=harness, environment=environment)
        missing_extra = subprocess.run([str(python), "-I", "-m", "mathforge.mcp"],
                                       cwd=harness, env=environment, input="", capture_output=True, text=True)
        assert missing_extra.returncode != 0 and not missing_extra.stdout, missing_extra
        assert "mathforge[mcp]" in missing_extra.stderr, missing_extra.stderr
        if not skip_tests:
            command((python, "-I", "-m", "unittest", "discover", "-s", harness / "tests", "-v"),
                    cwd=harness, environment=environment)
        for example in sorted((harness / "examples").glob("*.py")):
            command((python, "-I", example), cwd=harness, environment=environment)
        command((python, "-I", harness / "tools" / "check_docs.py"),
                cwd=harness, environment=environment)
        command((python, "-I", harness / "tools/check_benchmarks.py"), cwd=harness, environment=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=REPOSITORY / "dist")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Only smoke-test artifacts and examples (full tests are the release default)")
    parser.add_argument("--core-only", action="store_true",
                        help="Explicitly omit optional MCP installation (not the full release gate)")
    arguments = parser.parse_args()
    expected = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    distribution = arguments.dist.resolve()
    wheel_candidates = sorted(distribution.glob(f"mathforge-{expected}-*.whl"))
    sdist = distribution / f"mathforge-{expected}.tar.gz"
    if len(wheel_candidates) != 1 or not sdist.is_file():
        parser.error("Expected one wheel and one sdist for the current version; run python -m build")
    wheel = wheel_candidates[0]
    check_archives(wheel, sdist, expected)
    for artifact in (wheel, sdist):
        print(f"Checking {artifact.name}", flush=True)
        isolated_check(artifact, expected, skip_tests=arguments.skip_tests)
        if not arguments.core_only:
            isolated_check(artifact, expected, skip_tests=False, mcp_extra=True)
    print("Wheel and sdist passed isolated " + ("core-only" if arguments.core_only else "core and MCP") + " checks.")


if __name__ == "__main__":
    main()
