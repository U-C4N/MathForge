"""REL-15: Validate a local release candidate and compare the final wheel's core.

The candidate is built only in a temporary staging checkout. Only version
assignments change there. No tag, commit, upload, push, or package publication is
performed. The persisted report contains hashes and check status, not artifacts.

Run this script before the final build. Afterward run:
    python tools/check_release_candidate.py --compare-final dist/mathforge-0.9.0-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "assets", "tests", "tools", "examples", "benchmarks", "documentation", ".github")
ROOT_FILES = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "CHANGELOG.md",
              "requirements-dev.txt", "requirements-mcp.txt")
IGNORE = shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__",
                                "*.pyc", ".pytest_cache", ".mypy_cache", ".venv", ".venv-*")


def normalize_core_file(name: str, payload: bytes) -> bytes:
    text = payload.decode("utf-8").replace("\r\n", "\n")
    if name == "mathforge/__init__.py":
        text, count = re.subn(r'^__version__[ \t]*=[ \t]*[\'\"][^\'\"]+[\'\"][ \t]*$',
                              '__version__ = "<VERSION>"', text, flags=re.MULTILINE)
        if count != 1:
            raise AssertionError("Expected exactly one package version assignment")
    return text.encode("utf-8")


def core_member(name: str) -> bool:
    path = Path(name)
    return (name.startswith("mathforge/") and "__pycache__" not in path.parts
            and (path.suffix in (".py", ".json") or path.name == "py.typed"))


def fingerprint(entries) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(entries):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(normalize_core_file(name, payload)).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    source = root / "src"
    entries = []
    for path in (source / "mathforge").rglob("*"):
        if path.is_file():
            if not path.resolve().is_relative_to(source.resolve()):
                raise AssertionError(f"Source link leaves the source tree: {path}")
            name = path.relative_to(source).as_posix()
            if core_member(name):
                entries.append((name, path.read_bytes()))
    if not entries:
        raise AssertionError("No core source files found")
    return fingerprint(entries)


def wheel_fingerprint(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return fingerprint((name, archive.read(name)) for name in archive.namelist() if core_member(name))


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return email.parser.Parser().parsestr(archive.read(name).decode("utf-8"))["Version"]


def wheel_metadata_fingerprint(path: Path) -> str:
    """Dependencies and entry points must also survive candidate -> final unchanged."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for suffix in ("METADATA", "entry_points.txt"):
            names = [name for name in archive.namelist() if name.endswith(".dist-info/" + suffix)]
            if len(names) != 1:
                raise AssertionError(f"Expected exactly one {suffix} in the wheel")
            payload = archive.read(names[0]).decode("utf-8").replace("\r\n", "\n")
            if suffix == "METADATA":
                payload, count = re.subn(r"^Version: [^\n]+$", "Version: <VERSION>", payload,
                                         count=1, flags=re.MULTILINE)
                if count != 1:
                    raise AssertionError("Wheel metadata has no version")
            digest.update(suffix.encode("utf-8") + b"\0" + payload.encode("utf-8"))
    return digest.hexdigest()


def replace_version(path: Path, variable: str, original: str, candidate: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = rf'^{re.escape(variable)}[ \t]*=[ \t]*[\'\"]{re.escape(original)}[\'\"][ \t]*$'
    text, count = re.subn(pattern, f'{variable} = "{candidate}"', text, flags=re.MULTILINE)
    if count != 1:
        raise AssertionError(f"Expected one {original} version assignment in {path.name}")
    path.write_text(text, encoding="utf-8", newline="\n")


def artifact_record(path: Path) -> dict:
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_version(target: str, rc_number: int) -> str:
    if type(rc_number) is not int or rc_number < 1:
        raise ValueError("rc-number must be a positive integer")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", target):
        raise ValueError("The target must be a final major.minor.patch version")
    return f"{target}rc{rc_number}"


def validate_candidate(report_path: Path, *, rc_number=1) -> None:
    target = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    candidate = candidate_version(target, rc_number)
    init = (REPOSITORY / "src/mathforge/__init__.py").read_text(encoding="utf-8")
    versions = re.findall(r'^__version__[ \t]*=[ \t]*[\'\"]([^\'\"]+)[\'\"][ \t]*$', init, flags=re.MULTILINE)
    if versions != [target]:
        raise AssertionError("Package __version__ must match pyproject.toml")
    source_hash = source_fingerprint(REPOSITORY)
    report = {"target_version": target, "candidate_version": candidate,
              "status": "running", "normalized_core_sha256": source_hash,
              "checks": {}, "artifacts": []}
    save_report(report_path, report)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    try:
        with tempfile.TemporaryDirectory(prefix="mathforge-rc-") as temporary:
            stage = Path(temporary).resolve() / "source"
            stage.mkdir()
            for name in DIRECTORIES:
                source = REPOSITORY / name
                if source.is_dir():
                    shutil.copytree(source, stage / name, ignore=IGNORE)
            for name in ROOT_FILES:
                source = REPOSITORY / name
                if not source.is_file():
                    raise AssertionError(f"Missing release input: {name}")
                shutil.copy2(source, stage / name)
            replace_version(stage / "pyproject.toml", "version", target, candidate)
            replace_version(stage / "src/mathforge/__init__.py", "__version__", target, candidate)
            candidate_hash = source_fingerprint(stage)
            if candidate_hash != source_hash:
                raise AssertionError("Candidate core differs from final source beyond its version")
            report["candidate_core_sha256"] = candidate_hash
            subprocess.run([sys.executable, "-m", "build"], cwd=stage, env=environment, check=True)
            report["checks"]["build"] = "passed"
            wheel = next((stage / "dist").glob(f"mathforge-{candidate}-*.whl"))
            if wheel_version(wheel) != candidate or wheel_fingerprint(wheel) != source_hash:
                raise AssertionError("Candidate wheel core/metadata differs from validated source")
            report["checks"]["candidate_wheel_core"] = "passed"
            report["normalized_metadata_sha256"] = wheel_metadata_fingerprint(wheel)
            subprocess.run([sys.executable, str(stage / "tools/check_distribution.py"),
                            "--dist", str(stage / "dist")], cwd=stage, env=environment, check=True)
            report["checks"]["isolated_wheel_sdist_core_mcp_tests_and_public_docs"] = "passed"
            report["artifacts"] = [artifact_record(path) for path in sorted((stage / "dist").iterdir())
                                   if path.suffix == ".whl" or path.name.endswith(".tar.gz")]
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save_report(report_path, report)
    print(f"Local {candidate} candidate passed; report: {report_path}")


def compare_final(wheel: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise AssertionError("No successful candidate validation report is available")
    expected = report["normalized_core_sha256"]
    if source_fingerprint(REPOSITORY) != expected:
        raise AssertionError("Core source changed after candidate validation; rerun the candidate gate")
    if wheel_version(wheel) != report["target_version"]:
        raise AssertionError("Final wheel has the wrong version")
    if wheel_metadata_fingerprint(wheel) != report.get("normalized_metadata_sha256"):
        raise AssertionError("Final wheel metadata/entry points changed; rerun the candidate gate")
    actual = wheel_fingerprint(wheel)
    if actual != expected:
        raise AssertionError("Final wheel core differs from the validated candidate")
    report["final_comparison"] = {"status": "passed", "normalized_core_sha256": actual,
                                  **artifact_record(wheel)}
    save_report(report_path, report)
    print("Final wheel matches the validated candidate core after version normalization.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=REPOSITORY / "dist/release-candidate-validation.json")
    parser.add_argument("--compare-final", type=Path, help="Compare an already-built final wheel with the candidate")
    parser.add_argument("--rc-number", type=int, default=1, help="Positive candidate number (default: 1)")
    arguments = parser.parse_args()
    if arguments.rc_number < 1:
        parser.error("--rc-number must be a positive integer")
    report_path = arguments.report.resolve()
    if arguments.compare_final is None:
        validate_candidate(report_path, rc_number=arguments.rc_number)
    else:
        compare_final(arguments.compare_final.resolve(), report_path)


if __name__ == "__main__":
    main()
