#!/usr/bin/env python3
"""
run_library_tests.py — Stage 1a. Runs EVERY subject library's own test suite
and writes ONE JSON PER LIBRARY.

v3 fixes a real bug found by inspecting the repos directly, not by assumption:
several libraries' test files DON'T follow pytest's default collection
pattern (test_*.py / *_test.py), so pytest silently collects zero tests even
when pointed straight at the right directory:

  - ltn (LTNtorch)   tests/tests.py        <- single file, no "_" after "test"
  - scallopy         etc/scallopy/tests/test.py   <- same issue
  - spl's substrate  pypsdd/tests/testmpe.py etc. <- "test" + name, no "_"

This version scans .py file CONTENTS for "def test_" instead of trusting
filenames, and always invokes pytest with `-o python_files=*.py` so it
collects from any .py file in the target directory regardless of naming.

Also, some libraries genuinely have no Python test suite to run at all:
  - neurasp, slash    only scattered per-example test.py demo scripts, not
                       a real unit-test suite
  - neupsl (pslpython) the actual test suite is Java/JUnit under Maven —
                       pytest can never run this one; flagged as
                       runner="maven_not_pytest" rather than faked

Usage:
    python run_library_tests.py --list
    python run_library_tests.py                              # site-packages only
    python run_library_tests.py --auto-clone                  # clone whatever's missing/testless, then run
    python run_library_tests.py --auto-clone --only ltn,scallopy
    python run_library_tests.py --source-checkout ltn=/Users/me/repos/LTNtorch
    python run_library_tests.py --timeout 120
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHECKOUT_DIR = HERE / "_src_checkouts"

TEST_FUNC_RE = re.compile(r"^\s*def\s+test_\w+\s*\(", re.MULTILINE)


# ---------------------------------------------------------------------------
# Library registry
# ---------------------------------------------------------------------------

@dataclass
class LibrarySpec:
    name: str
    import_name: str
    test_subdir_candidates: list
    repo_url: Optional[str] = None
    repo_test_subdir: str = "tests"
    documented_command: Optional[str] = None
    pip_name: Optional[str] = None
    runner: str = "pytest"          # "pytest" | "maven_not_pytest" | "no_test_suite"
    notes: str = ""


LIBRARIES: list[LibrarySpec] = [
    LibrarySpec(
        name="deepproblog",
        import_name="deepproblog",
        test_subdir_candidates=["tests"],
        repo_url="https://github.com/ML-KULeuven/deepproblog.git",
        repo_test_subdir="src/deepproblog/tests",
        documented_command="python -m deepproblog test",
        notes="ships tests in the pip package itself; standard test_*.py naming",
    ),
    LibrarySpec(
        name="ltn",
        import_name="ltn",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/logictensornetworks/LTNtorch.git",
        repo_test_subdir="tests",
        pip_name="LTNtorch",
        notes="pip package ships no tests/ at all; repo has tests/tests.py (single file, "
              "NON-standard filename — needs python_files override to collect). Confirmed by direct inspection.",
    ),
    LibrarySpec(
        name="scallopy",
        import_name="scallopy",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/scallop-lang/scallop.git",
        repo_test_subdir="etc/scallopy/tests",
        notes="tests/test.py — NON-standard filename (no underscore); needs Rust toolchain to build scallopy itself",
    ),
    LibrarySpec(
        name="neurasp",
        import_name="neurasp",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/azreasoners/NeurASP.git",
        repo_test_subdir="tests",
        runner="no_test_suite",
        notes="confirmed: no tests/ directory in the repo at all. Only scattered examples/*/test.py demo "
              "scripts, which are runnable programs, not a pytest unit-test suite. A fuzzing harness for "
              "this library needs a different seed source than 'run the test suite'.",
    ),
    LibrarySpec(
        name="deepstochlog",
        import_name="deepstochlog",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/ML-KULeuven/deepstochlog.git",
        repo_test_subdir="tests",
        notes="confirmed: tests/test_*.py, standard naming, six files (test_citeseer, test_parser, "
              "test_anbncn, test_examples, test_tabled_tree_builder, test_bracket)",
    ),
    LibrarySpec(
        name="deepsoftlog",
        import_name="deepsoftlog",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/jjcmoon/DeepSoftLog.git",
        repo_test_subdir="tests",
        notes="confirmed: tests/test_*.py, standard naming, five files",
    ),
    LibrarySpec(
        name="pylon",
        import_name="pylon",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/pylon-lib/pylon.git",
        repo_test_subdir="tests",
        pip_name="pylon-lib",
        notes="confirmed: tests/test_*.py, standard naming, ten+ files",
    ),
    LibrarySpec(
        name="slash",
        import_name="SLASH",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/ml-research/SLASH.git",
        repo_test_subdir="tests",
        runner="no_test_suite",
        notes="confirmed: no tests/ directory anywhere in the repo. Only "
              "src/experiments/vqa/test.py and test.sh, which are experiment runner scripts, not unit tests.",
    ),
    LibrarySpec(
        name="spl",
        import_name="spl",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/KareemYousrii/SPL.git",
        repo_test_subdir="grids/pypsdd/pypsdd/tests",
        notes="confirmed: SPL itself has no top-level tests/. Only its BUNDLED pypsdd substrate library "
              "has tests (testmpe.py, testdata.py, testpsdd.py, testkl.py, testsdd.py — NON-standard "
              "naming, no underscore). Running these tests SDD/pypsdd, not SPL's own neurosymbolic layer — "
              "flag this distinction if citing results from here.",
    ),
    LibrarySpec(
        name="neupsl",
        import_name="pslpython",
        test_subdir_candidates=["../tests", "../../tests", "tests"],
        repo_url="https://github.com/linqs/psl.git",
        repo_test_subdir="psl-python",
        pip_name="pslpython",
        runner="maven_not_pytest",
        notes="confirmed: PSL's real test suite is Java/JUnit under Maven modules (psl-java, psl-parser), "
              "run via 'mvn test' from the repo root — not pytest, not Python at all. The pslpython pip "
              "package is a thin wrapper with no dedicated Python test suite of its own. Per the population "
              "doc's own note: consider substituting A-NeSI or DomiKnowS if this blocks the pipeline.",
    ),
    LibrarySpec(
        name="problog",
        import_name="problog",
        test_subdir_candidates=["test"],
        repo_url="https://github.com/ML-KULeuven/problog.git",
        repo_test_subdir="test",
        notes="substrate beneath deepproblog — ships tests in the pip package, standard naming",
    ),
    LibrarySpec(
        name="pysdd",
        import_name="pysdd",
        test_subdir_candidates=["../tests", "test", "tests"],
        repo_url="https://github.com/wannesm/PySDD.git",
        repo_test_subdir="tests",
        notes="substrate: SDD compiler beneath problog",
    ),
    LibrarySpec(
        name="clingo",
        import_name="clingo",
        test_subdir_candidates=["../tests", "test", "tests"],
        repo_url="https://github.com/potassco/clingo.git",
        repo_test_subdir="libclingo/tests",
        runner="no_test_suite",
        notes="substrate: C++ core with CMake/ctest tests, not pytest-runnable as-is; python bindings "
              "have no dedicated Python test suite of their own",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_environment() -> dict:
    packages = {}
    for spec in LIBRARIES:
        try:
            mod = importlib.import_module(spec.import_name)
            packages[spec.import_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[spec.import_name] = None
    for extra in ["torch", "torchvision", "pytest", "numpy", "pyswip"]:
        try:
            mod = importlib.import_module(extra)
            packages[extra] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[extra] = None

    cuda = {"available": False, "version": None, "device_count": 0}
    try:
        import torch
        cuda["available"] = torch.cuda.is_available()
        cuda["version"] = torch.version.cuda
        cuda["device_count"] = torch.cuda.device_count()
    except Exception:
        pass

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": cuda,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def has_tests(path: Path) -> bool:
    """Content-based detection: any .py file under path containing a
    'def test_*(' function, regardless of the FILE's own name. This is what
    catches ltn's tests.py, scallopy's test.py, spl's testmpe.py etc. — none
    of which match pytest's filename-based default collection pattern."""
    if not path.is_dir():
        return False
    for py_file in path.rglob("*.py"):
        try:
            text = py_file.read_text(errors="ignore")
        except Exception:
            continue
        if TEST_FUNC_RE.search(text):
            return True
    return False


def find_installed_test_root(spec: LibrarySpec) -> tuple[Optional[Path], dict]:
    diag = {"import_ok": False, "package_dir": None, "tried": []}
    try:
        module = importlib.import_module(spec.import_name)
    except Exception as e:
        diag["import_error"] = f"{type(e).__name__}: {e}"
        return None, diag

    diag["import_ok"] = True
    mod_file = getattr(module, "__file__", None)
    if mod_file is None:
        diag["error"] = "module has no __file__ (namespace package?)"
        return None, diag

    package_dir = Path(mod_file).resolve().parent
    diag["package_dir"] = str(package_dir)

    for candidate in spec.test_subdir_candidates:
        candidate_path = (package_dir / candidate).resolve()
        diag["tried"].append(str(candidate_path))
        if has_tests(candidate_path):
            return candidate_path, diag

    return None, diag


def clone_repo(spec: LibrarySpec) -> tuple[Optional[Path], dict]:
    diag = {"repo_url": spec.repo_url}
    if spec.repo_url is None:
        diag["error"] = "no repo_url configured for this library"
        return None, diag

    dest = CHECKOUT_DIR / spec.name
    if dest.exists() and any(dest.iterdir()):
        diag["action"] = "reused_existing_checkout"
    else:
        diag["action"] = "cloned"
        CHECKOUT_DIR.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", spec.repo_url, str(dest)]
        diag["clone_command"] = " ".join(cmd)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            diag["clone_returncode"] = proc.returncode
            if proc.returncode != 0:
                diag["clone_stderr_tail"] = proc.stderr.splitlines()[-15:]
                return None, diag
        except subprocess.TimeoutExpired:
            diag["error"] = "git clone timed out after 300s"
            return None, diag
        except FileNotFoundError:
            diag["error"] = "git not found on PATH"
            return None, diag

    test_path = (dest / spec.repo_test_subdir).resolve()
    diag["test_path_tried"] = str(test_path)
    if has_tests(test_path):
        return test_path, diag

    diag["fallback_search"] = True
    candidates = sorted({
        p.parent for p in dest.rglob("*.py")
        if TEST_FUNC_RE.search(p.read_text(errors="ignore") or "")
    })
    if candidates:
        diag["fallback_candidates"] = [str(c) for c in candidates]
        return candidates[0], diag

    diag["error"] = f"no test_* functions found under {spec.repo_test_subdir} or anywhere in the checkout"
    return None, diag


def run_pytest(test_root: Path, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        junit_path = Path(tmp) / "report.xml"
        cmd = [
            sys.executable, "-m", "pytest", str(test_root),
            "-q", "--no-header", "-p", "no:cacheprovider",
            # collect from ANY .py file, not just test_*.py / *_test.py —
            # required for ltn (tests.py), scallopy (test.py), spl's pypsdd
            # substrate (testmpe.py etc.)
            "-o", "python_files=*.py",
            # don't let ONE broken test file (missing optional dep, missing
            # data file not included in a shallow clone, etc.) abort
            # collection of every other test in the directory — confirmed
            # needed for deepstochlog: 2 of 25 files fail to import (dgl
            # binary mismatch, missing data file) and without this flag
            # pytest refuses to run the other 23 at all.
            "--continue-on-collection-errors",
            f"--junitxml={junit_path}",
        ]
        # Several libraries' test files import repo-local helper packages
        # that sit NEXT TO the tests/ directory in the checkout (e.g.
        # deepstochlog's tests/test_anbncn.py does `import examples.anbncn...`
        # where examples/ is a sibling of tests/, not part of the installed
        # package). pytest only makes that importable if the checkout root is
        # on sys.path. Confirmed by direct reproduction: without this, only
        # 1 of 6 deepstochlog test files collects (9/~29 tests) because the
        # other 5 raise ImportError on `examples.*` and get silently counted
        # as collection errors rather than failures. Adding the checkout
        # root (test_root's parent) to PYTHONPATH when it contains such a
        # sibling directory fixes this without needing per-library special
        # casing in the caller.
        env = os.environ.copy()
        repo_root_candidate = test_root.parent
        sibling_pkg_dirs = [d for d in repo_root_candidate.glob("*")
                             if d.is_dir() and d.name not in ("tests", "test", "__pycache__")
                             and (d / "__init__.py").exists()]
        if sibling_pkg_dirs:
            env["PYTHONPATH"] = os.pathsep.join(
                [str(repo_root_candidate)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
            )
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            seconds = round(time.time() - start, 2)
        except subprocess.TimeoutExpired as e:
            return {
                "command": " ".join(cmd),
                "seconds": timeout,
                "returncode": None,
                "exit_meaning": "timeout",
                "status": "timeout",
                "stdout_tail": (e.stdout or "")[-2000:].splitlines()[-10:] if e.stdout else [],
                "stderr_tail": (e.stderr or "")[-2000:].splitlines()[-10:] if e.stderr else [],
            }

        result = {
            "command": " ".join(cmd),
            "seconds": seconds,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-10:],
            "stderr_tail": proc.stderr.splitlines()[-10:],
            "warnings": [l for l in proc.stdout.splitlines() if "arning" in l][:10],
        }

        if proc.returncode == 0:
            result["exit_meaning"] = "all_passed"
        elif proc.returncode == 1:
            result["exit_meaning"] = "tests_failed"
        elif proc.returncode == 5:
            result["exit_meaning"] = "no_tests_collected"
        else:
            result["exit_meaning"] = f"pytest_exit_{proc.returncode}"

        if junit_path.exists():
            counts, failing, passing = parse_junit(junit_path)
            result["junit"] = {
                "counts": counts,
                "failing_tests": failing,
                "n_failing_recorded": len(failing),
                "passing_tests": passing,
            }
            if result["exit_meaning"] == "no_tests_collected":
                result["status"] = "no_tests_collected"
            elif counts["failures"] or counts["errors"]:
                result["status"] = "tests_failed"
            elif result["returncode"] == 0:
                result["status"] = "ok"
            else:
                result["status"] = "runner_error"
        else:
            result["status"] = "runner_error"
            result["error"] = "no JUnit XML produced — pytest likely crashed before collection"

        return result


def parse_junit(junit_path: Path) -> tuple[dict, list]:
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]

    tests = failures = errors = skipped = 0
    failing_tests = []
    passing_tests = []  # [{"test": "...", "file": "..."}] — used to gate per-test fuzzing

    for suite in suites:
        tests += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))

        for case in suite.findall("testcase"):
            classname = case.attrib.get("classname", "")
            testname = case.attrib.get("name", "")
            file_ = case.attrib.get("file", "")
            full = f"{classname}::{testname}" if classname else testname
            failed_or_errored = False
            for tag, kind in (("failure", "failure"), ("error", "error")):
                node = case.find(tag)
                if node is not None:
                    failed_or_errored = True
                    msg = (node.attrib.get("message") or "").strip()
                    failing_tests.append({"test": full, "kind": kind, "message": msg[:300]})
            skipped_node = case.find("skipped")
            if not failed_or_errored and skipped_node is None:
                passing_tests.append({"test": full, "file": file_})

    passed = tests - failures - errors - skipped
    counts = {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped, "passed": passed}
    return counts, failing_tests, passing_tests


def run_one_library(
    spec: LibrarySpec,
    environment: dict,
    timeout: int,
    auto_clone: bool,
    source_checkout_override: Optional[Path],
) -> dict:
    record = {
        "library": spec.name,
        "import_name": spec.import_name,
        "notes": spec.notes,
        "documented_command": spec.documented_command,
        "repo_url": spec.repo_url,
        "runner": spec.runner,
    }

    if spec.runner == "no_test_suite":
        record["status"] = "no_test_suite"
        record["hint"] = spec.notes
        return record

    if spec.runner == "maven_not_pytest":
        record["status"] = "wrong_runner_for_pytest"
        record["hint"] = spec.notes + " Run manually: git clone " + str(spec.repo_url) + " && cd psl && mvn test"
        return record

    # 1. explicit --source-checkout override always wins
    if source_checkout_override is not None:
        test_path = (source_checkout_override / spec.repo_test_subdir).resolve()
        if not has_tests(test_path):
            if has_tests(source_checkout_override):
                test_path = source_checkout_override
            else:
                record["status"] = "tests_not_found"
                record["source_checkout_override"] = str(source_checkout_override)
                record["tried"] = [str(test_path), str(source_checkout_override)]
                return record
        record["test_root"] = str(test_path)
        record["test_source"] = "source_checkout_override"
        record.update(run_pytest(test_path, timeout=timeout))
        return record

    installed_version = environment["packages"].get(spec.import_name)

    # 2. try the installed package's own tests, if the package is installed
    if installed_version is not None:
        test_root, diag = find_installed_test_root(spec)
        if test_root is not None:
            record["test_root"] = str(test_root)
            record["test_source"] = "site_packages"
            record["locate_diagnostics"] = diag
            record.update(run_pytest(test_root, timeout=timeout))
            return record
        record["locate_diagnostics"] = diag

    # 3. try cloning, if allowed
    if auto_clone:
        clone_path, clone_diag = clone_repo(spec)
        record["clone_diagnostics"] = clone_diag
        if clone_path is not None:
            record["test_root"] = str(clone_path)
            record["test_source"] = "auto_clone"
            record.update(run_pytest(clone_path, timeout=timeout))
            return record
        record["status"] = "clone_failed" if installed_version is None else "installed_no_tests_clone_failed"
        return record

    # 4. no clone attempted
    if installed_version is None:
        record["status"] = "not_installed"
        record["how_to_install"] = f"pip install {spec.pip_name or spec.import_name}"
        record["hint"] = "rerun with --auto-clone to fetch the repo instead, or install the package first"
    else:
        record["status"] = "tests_not_found"
        record["hint"] = (
            "Installed, but no tests found in the package. Rerun with --auto-clone "
            f"to pull {spec.repo_url}, or pass --source-checkout {spec.name}=/path/to/repo."
        )
    return record


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def parse_source_checkouts(raw_values: list[str]) -> dict[str, Path]:
    result = {}
    for raw in raw_values:
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                print(f"WARNING: ignoring malformed --source-checkout value: {pair!r} (expected name=path)", file=sys.stderr)
                continue
            name, path = pair.split("=", 1)
            result[name.strip()] = Path(path.strip()).expanduser().resolve()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--auto-clone", action="store_true")
    parser.add_argument("--source-checkout", action="append", default=[])
    args = parser.parse_args()

    if args.list:
        print(f"{'library':13s} {'runner':20s} {'import_name':13s} repo_url")
        for spec in LIBRARIES:
            print(f"{spec.name:13s} {spec.runner:20s} {spec.import_name:13s} {spec.repo_url or '(none)'}")
            print(f"{'':13s} {'':20s} {'':13s} {spec.notes}")
        return

    selected = LIBRARIES
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        selected = [s for s in LIBRARIES if s.name in wanted]
        missing = wanted - {s.name for s in selected}
        if missing:
            print(f"WARNING: unknown library names ignored: {sorted(missing)}", file=sys.stderr)

    overrides = parse_source_checkouts(args.source_checkout)
    unknown_overrides = set(overrides) - {s.name for s in LIBRARIES}
    if unknown_overrides:
        print(f"WARNING: --source-checkout names not in registry: {sorted(unknown_overrides)}", file=sys.stderr)

    print("==> Collecting environment")
    environment = collect_environment()

    summary = {"environment": environment, "auto_clone": args.auto_clone, "libraries": []}

    for spec in selected:
        print(f"\n==> {spec.name}")
        record = run_one_library(
            spec, environment, timeout=args.timeout,
            auto_clone=args.auto_clone,
            source_checkout_override=overrides.get(spec.name),
        )
        record["environment"] = environment
        out_path = RESULTS_DIR / f"{spec.name}_tests.json"
        write_json(out_path, record)

        status = record.get("status", "unknown")
        if status in ("ok", "tests_failed") and "junit" in record:
            c = record["junit"]["counts"]
            src = record.get("test_source", "?")
            print(f"    status={status}  passed={c['passed']} failed={c['failures']} errors={c['errors']} "
                  f"skipped={c['skipped']}  source={src}  ({record.get('seconds', '?')}s)")
        elif status == "not_installed":
            print(f"    status=not_installed  ({record.get('how_to_install', '')})")
        elif status in ("no_test_suite", "wrong_runner_for_pytest"):
            print(f"    status={status}  {record.get('hint', '')[:120]}...")
        elif status == "tests_not_found":
            print(f"    status=tests_not_found  {record.get('hint', '')}")
        elif status in ("clone_failed", "installed_no_tests_clone_failed"):
            print(f"    status={status}  see clone_diagnostics in {out_path}")
        else:
            print(f"    status={status}")

        summary["libraries"].append({
            "library": spec.name,
            "status": status,
            "runner": spec.runner,
            "test_source": record.get("test_source"),
            "counts": record.get("junit", {}).get("counts"),
            "result_file": str(out_path),
        })

    write_json(RESULTS_DIR / "_summary.json", summary)

    print("\n==> Per-library JSON files:")
    for entry in summary["libraries"]:
        print(f"  {entry['result_file']}")
    print(f"  {RESULTS_DIR / '_summary.json'}")

    ok = sum(1 for e in summary["libraries"] if e["status"] == "ok")
    failed = sum(1 for e in summary["libraries"] if e["status"] == "tests_failed")
    not_installed = sum(1 for e in summary["libraries"] if e["status"] == "not_installed")
    not_pytest = sum(1 for e in summary["libraries"] if e["status"] in ("no_test_suite", "wrong_runner_for_pytest"))
    unresolved = sum(1 for e in summary["libraries"] if e["status"] in
                      ("tests_not_found", "clone_failed", "installed_no_tests_clone_failed", "no_tests_collected"))
    print(f"\n==> ok={ok}  tests_failed={failed}  not_installed={not_installed}  "
          f"not_pytest_runnable={not_pytest}  unresolved={unresolved}  total={len(summary['libraries'])}")
    if not args.auto_clone and (not_installed or unresolved):
        print("    Tip: rerun with --auto-clone to pull missing repos.")


if __name__ == "__main__":
    main()
