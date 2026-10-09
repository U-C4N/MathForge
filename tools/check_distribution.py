"""REL-06/07/08/09/10/12: Test artifacts from outside the source checkout.

Run ``python -m build`` first, then ``python tools/check_distribution.py``.
Both wheel and sdist are installed into fresh dependency-free runtime venvs.
The sdist is rebuilt by pip's isolated PEP 517 build before installation.
"""

from __future__ import annotations

import argparse
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
assert (metadata.requires("mathforge") or []) == []
assert resources.files("mathforge").joinpath("py.typed").is_file()
for dependency in ("sympy", "numpy", "scipy", "sage"):
    assert importlib.util.find_spec(dependency) is None, dependency
x = mf.symbol("x")
result = mf.solve(mf.Eq(x**3 - 2, 0), for_=x)
assert result.execution_status is mf.ExecutionStatus.COMPLETED
assert mf.verify(result).verified
assert mf.loads(mf.dumps(result)) == result
assert mf.det(mf.Matrix(((1, 2), (3, 4)))) == mf.Rational(-2)
print("Isolated installed package:", installed)
'''


def command(arguments, *, cwd, environment):
    subprocess.run([str(argument) for argument in arguments], cwd=cwd,
                   env=environment, check=True)


def check_archives(wheel: Path, sdist: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert not any("docs" in Path(name).parts for name in names), "Wheel contains local docs"
        assert "mathforge/py.typed" in names, "Wheel does not include py.typed"
        for module in (REPOSITORY / "src" / "mathforge").glob("*.py"):
            assert f"mathforge/{module.name}" in names, f"Missing wheel module: {module.name}"
        for schema in (REPOSITORY / "src" / "mathforge" / "schemas").glob("*.json"):
            assert f"mathforge/schemas/{schema.name}" in names, f"Missing wheel schema: {schema.name}"
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        package_metadata = email.parser.Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        assert package_metadata["Version"] == expected_version
        assert not package_metadata.get_all("Requires-Dist"), "Runtime dependencies were added"
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
                                    ("schemas", ("*.json",))):
            for pattern in patterns:
                for source in (REPOSITORY / directory).rglob(pattern):
                    relative = source.relative_to(REPOSITORY).as_posix()
                    assert relative in names, f"Missing sdist input: {relative}"
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "CHANGELOG.md", "LICENSE", "assets/mark.svg"):
            assert name in names, f"Missing sdist input: {name}"


def isolated_check(artifact: Path, expected_version: str, *, skip_tests: bool) -> None:
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
        for name in ("assets", "tests", "examples", "benchmarks", "schemas", "tools"):
            source = REPOSITORY / name
            if source.is_dir():
                shutil.copytree(source, harness / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "CHANGELOG.md"):
            shutil.copy2(REPOSITORY / name, harness / name)
        command((python, "-I", "-m", "pip", "install", "--no-deps", artifact),
                cwd=harness, environment=environment)
        command((python, "-I", "-c", PROBE, REPOSITORY, expected_version),
                cwd=harness, environment=environment)
        if not skip_tests:
            command((python, "-I", "-m", "unittest", "discover", "-s", harness / "tests", "-v"),
                    cwd=harness, environment=environment)
        for example in sorted((harness / "examples").glob("*.py")):
            command((python, "-I", example), cwd=harness, environment=environment)
        command((python, "-I", harness / "tools" / "check_docs.py"),
                cwd=harness, environment=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=REPOSITORY / "dist")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Only smoke-test artifacts and examples (full tests are the release default)")
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
    print("Wheel and sdist passed isolated checks.")


if __name__ == "__main__":
    main()
