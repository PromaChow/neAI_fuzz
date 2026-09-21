#!/usr/bin/env python3
"""
fuzz_auto_deepstochlog.py — fuzzes the FOCAL METHOD each passing test is
actually testing, by calling that method directly with mutated arguments.

Two discovery modes, both grounded in the passing test's own real source
(nothing hand-authored):

  MODE A — DSL-string calls. A passing test builds a program string
  containing a DeepStochLog probability annotation (`<number> :: head -->
  body.`) and hands it to an imported function (e.g. `parse_rules`). The
  literal probability value is mutated in place inside the real program
  text, and the real function is called on the mutated program.

  MODE B — general argument calls. A passing test calls ANY function
  resolvable through that file's own imports with literal int/float/bool
  arguments (e.g. test_anbncn.py's `create_anbncn_language(min_length=3,
  max_length=3)`). Each numeric/boolean argument is mutated one at a time
  and the REAL function is called directly with the mutated argument —
  this is what covers tests like test_anbncn_creation that Mode A can't
  see, since they never build a `::` string at all.

Validity labeling: for a numeric argument whose original value is >= 0
(the common "length/count" shape), a negative mutant is treated as
presumed-invalid — DOC-INVALID — and everything else as DOC-VALID. This is
a heuristic, not a proven contract, and is reported as such. Boolean flips
and arguments with no clear sign convention get no presumed validity
(labeled N/A) — both are legitimate inputs, so the case is reported for
inspection rather than scored as a violation either way.

Rules (same discipline as the rest of this study):
  1. Refuses to run unless results/deepstochlog_tests.json shows the
     baseline suite actually passed something (passed > 0).
  2. Every case is tied to a real passing test and calls the real,
     dynamically-imported focal function — nothing here is simulated.
  3. A DOC-INVALID case that the function accepts without complaint is
     reported as "SILENTLY ACCEPTED (no validation observed)" — described
     factually, not automatically asserted as a bug, since some functions
     are legitimately designed to clamp/tolerate out-of-range input rather
     than reject it (this script reports which one happened; it does not
     assume malicious intent from the library author).

Usage:
    python fuzz_auto_deepstochlog.py --list    # show discovered focal calls + generated mutants, run nothing
    python fuzz_auto_deepstochlog.py           # gate-check, discover, mutate, run
    python fuzz_auto_deepstochlog.py --force   # skip the gate check (debug only)
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import pathlib
import re
import signal
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
GATE_FILE = RESULTS / "deepstochlog_tests.json"

PROB_LIT_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*::')
STDLIB_SKIP = {"self", "print", "len", "list", "set", "range", "sorted", "isinstance"}
CASE_TIMEOUT_SECONDS = 10  # backstop: a mutated argument can make some focal
                            # functions combinatorially expensive; a hung or
                            # runaway case is recorded as TIMEOUT, not left
                            # to kill the whole run.


class _CaseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _CaseTimeout(f"exceeded {CASE_TIMEOUT_SECONDS}s")


# ---------------------------------------------------------------------------
# Stage 1 — gate check
# ---------------------------------------------------------------------------
def check_gate(force: bool) -> dict:
    if force:
        return {"gate": "skipped", "reason": "--force"}
    if not GATE_FILE.exists():
        sys.exit(f"REFUSING TO RUN: {GATE_FILE} missing. "
                  f"Run `python run_library_tests.py --only deepstochlog --auto-clone` first.")
    data = json.loads(GATE_FILE.read_text())
    status = data.get("status")
    junit = data.get("junit", {})
    passed = junit.get("counts", {}).get("passed", 0)
    if status not in ("ok", "tests_failed") or passed == 0:
        sys.exit(f"REFUSING TO RUN: baseline shows status={status!r}, passed={passed}.")
    passing = junit.get("passing_tests", [])
    print(f"Gate OK: baseline passed={passed} ({len(passing)} passing test names recorded).")
    return {"gate": "passed", "baseline_passed": passed, "test_root": data.get("test_root"),
            "passing_tests": passing}


def resolve_test_file(classname: str, search_roots: list[pathlib.Path]) -> pathlib.Path | None:
    parts = classname.split(".")
    for root in search_roots:
        for cut in range(len(parts), 0, -1):
            candidate = root.joinpath(*parts[:cut]).with_suffix(".py")
            if candidate.is_file():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Shared: resolve literal-valued expressions (str/int/float/bool), including
# simple local-variable assignments and string concatenation, from a test's
# own AST — no execution, purely static.
# ---------------------------------------------------------------------------
def resolve_literal_expr(node, local_vars: dict):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = resolve_literal_expr(node.operand, local_vars)
        return -v if isinstance(v, (int, float)) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = resolve_literal_expr(node.left, local_vars)
        right = resolve_literal_expr(node.right, local_vars)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    if isinstance(node, ast.Name) and node.id in local_vars:
        return local_vars[node.id]
    return None


def build_import_map(tree: ast.Module) -> dict[str, tuple[str, str]]:
    import_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_map[alias.asname or alias.name] = (node.module, alias.name)
    return import_map


def build_local_vars(fn: ast.FunctionDef) -> dict:
    local_vars: dict = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            val = resolve_literal_expr(node.value, local_vars)
            if val is not None:
                local_vars[node.targets[0].id] = val
    return local_vars


# ---------------------------------------------------------------------------
# MODE A discovery — DSL program strings with a `::` literal
# ---------------------------------------------------------------------------
def discover_mode_a(file_path: pathlib.Path, func_names: set[str], tree, import_map) -> list[dict]:
    src = file_path.read_text(errors="ignore")
    discovered = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in func_names):
            continue
        local_vars = build_local_vars(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
                resolved = resolve_literal_expr(node.args[0], local_vars)
                if isinstance(resolved, str) and PROB_LIT_RE.search(resolved):
                    target = import_map.get(node.func.id)
                    if target is None:
                        continue
                    discovered.append({
                        "mode": "A", "test": fn.name, "file": str(file_path),
                        "focal_module": target[0], "focal_attr": target[1], "program": resolved,
                    })
    return discovered


# ---------------------------------------------------------------------------
# MODE B discovery — general calls with literal int/float/bool arguments
# ---------------------------------------------------------------------------
def discover_mode_b(file_path: pathlib.Path, func_names: set[str], tree, import_map) -> list[dict]:
    discovered = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in func_names):
            continue
        local_vars = build_local_vars(fn)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            fname = node.func.id
            if fname in STDLIB_SKIP or fname not in import_map:
                continue
            module_name, attr_name = import_map[fname]
            try:
                mod = importlib.import_module(module_name)
                callable_obj = getattr(mod, attr_name)
                sig = inspect.signature(callable_obj)
            except Exception:
                continue

            # bind positional + keyword args to parameter names
            params = list(sig.parameters.values())
            bound: dict = {}
            ok = True
            for i, arg in enumerate(node.args):
                if i >= len(params):
                    ok = False
                    break
                val = resolve_literal_expr(arg, local_vars)
                if val is None:
                    ok = False
                    break
                bound[params[i].name] = val
            for kw in node.keywords:
                if kw.arg is None:
                    ok = False
                    break
                val = resolve_literal_expr(kw.value, local_vars)
                if val is None:
                    ok = False
                    break
                bound[kw.arg] = val
            if not ok or not bound:
                continue
            # only worth fuzzing if at least one bound arg is numeric/bool
            numeric_args = {k: v for k, v in bound.items() if isinstance(v, (int, float, bool))}
            if not numeric_args:
                continue
            discovered.append({
                "mode": "B", "test": fn.name, "file": str(file_path),
                "focal_module": module_name, "focal_attr": attr_name,
                "all_args": bound, "numeric_args": numeric_args,
            })
    return discovered


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------
def mutate_float(v: float) -> list[tuple[str, float]]:
    return [
        ("negate", -v), ("scale_5x", v * 5), ("add_one", v + 1.0), ("zero", 0.0),
        ("above_one_eps", 1.0 + 1e-6), ("nan", float("nan")),
        ("pos_inf", float("inf")), ("neg_inf", float("-inf")),
    ]


# Mode-A only: a STRING-level format mutation, not a numeric one. Tests the
# parser's own tokenizer rather than out-of-range values. Confirmed live
# against the real parser before adding this in: a stray extra decimal point
# (e.g. "0..2", "1.0.0") is not rejected — it's silently re-tokenized as TWO
# separate clauses (a bare fact ending at the first ".", then a new rule
# starting after it), producing a structurally different program with no
# error raised. This is the same underlying ambiguity that made "1.1" (an
# add_one mutation of 0.1) get silently split into "1." + "1.0 :: bar-->a."
# in the base numeric corpus — this operator targets that bug class
# directly instead of finding it by accident.
def mutate_format(v0: float) -> list[tuple[str, str]]:
    whole, _, frac = fmt(v0).partition(".")
    frac = frac or "0"
    return [("double_dot", f"{whole}..{frac}")]


def mutate_int(v: int) -> list[tuple[str, int]]:
    # kept deliberately modest: some focal functions here are combinatorial
    # in their length arguments (cubic in one case), so "huge" is a bounded
    # multiplier, not an unbounded one — CASE_TIMEOUT_SECONDS is the real
    # backstop against a mutation that's still too expensive.
    return [
        ("negate", -v), ("zero", 0),
        ("scale_10x", v * 10 if v != 0 else 20),
        ("off_by_one_under", v - 1),
    ]


def mutate_bool(v: bool) -> list[tuple[str, bool]]:
    return [("flip", not v)]


def fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(float(v))


def presumed_validity(v0, mutant) -> str:
    """Heuristic only: a non-negative numeric argument's negative mutant is
    presumed invalid. Bools and negative-sign-convention numbers get 'N/A' —
    no oracle is asserted."""
    if isinstance(v0, bool) or isinstance(mutant, bool):
        return "N/A"
    if isinstance(v0, (int, float)) and v0 >= 0:
        if isinstance(mutant, (int, float)) and math.isfinite(mutant) and mutant >= 0:
            return "DOC-VALID"
        return "DOC-INVALID"
    return "N/A"


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------
def build_corpus(mode_a: list[dict], mode_b: list[dict]) -> list[dict]:
    corpus = []
    for call in mode_a:
        program = call["program"]
        for lit_idx, lit_m in enumerate(PROB_LIT_RE.finditer(program)):
            v0 = float(lit_m.group(1))
            for op, v in [("seed", v0)] + mutate_float(v0):
                mutated_program = program[:lit_m.start(1)] + fmt(v) + program[lit_m.end(1):]
                corpus.append({
                    "mode": "A", "op": op, "value": v, "literal_index": lit_idx,
                    "from_test": call["test"], "from_file": call["file"],
                    "focal_module": call["focal_module"], "focal_attr": call["focal_attr"],
                    "call_args": {"program": mutated_program},
                    "validity": "DOC-VALID" if (0.0 <= v <= 1.0 and math.isfinite(v)) else "DOC-INVALID",
                })
            for op, lit_str in mutate_format(v0):
                mutated_program = program[:lit_m.start(1)] + lit_str + program[lit_m.end(1):]
                corpus.append({
                    "mode": "A", "op": op, "value": lit_str, "literal_index": lit_idx,
                    "from_test": call["test"], "from_file": call["file"],
                    "focal_module": call["focal_module"], "focal_attr": call["focal_attr"],
                    "call_args": {"program": mutated_program},
                    "validity": "DOC-INVALID",  # malformed by construction — a well-formed
                    # program never contains a stray second '.'
                    "check_fidelity": True,
                })

    for call in mode_b:
        for arg_name, v0 in call["numeric_args"].items():
            mutations = (mutate_bool(v0) if isinstance(v0, bool)
                         else mutate_int(v0) if isinstance(v0, int)
                         else mutate_float(v0))
            for op, v in [("seed", v0)] + mutations:
                mutated_args = dict(call["all_args"])
                mutated_args[arg_name] = v
                corpus.append({
                    "mode": "B", "op": op, "value": v, "mutated_arg": arg_name,
                    "from_test": call["test"], "from_file": call["file"],
                    "focal_module": call["focal_module"], "focal_attr": call["focal_attr"],
                    "call_args": mutated_args,
                    "validity": presumed_validity(v0, v),
                })
    return corpus


# ---------------------------------------------------------------------------
# Execution — calls the real, dynamically-imported focal function
# ---------------------------------------------------------------------------
def run_case(case: dict) -> dict:
    out = {k: case[k] for k in
           ("mode", "op", "value", "from_test", "focal_module", "focal_attr", "call_args", "validity")}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    timed_out = False
    misparsed = False
    try:
        mod = importlib.import_module(case["focal_module"])
        focal_fn = getattr(mod, case["focal_attr"])
        if case["mode"] == "A":
            result = focal_fn(case["call_args"]["program"])
            if case.get("check_fidelity") and getattr(result, "rules", None):
                first_rule = result.rules[0]
                if not hasattr(first_rule, "probability"):
                    # the literal was silently re-tokenized into a bare fact
                    # plus a separate rule, rather than being rejected —
                    # accepted, but as a DIFFERENT program than was written
                    misparsed = True
        else:
            result = focal_fn(**case["call_args"])
        out["result_repr"] = repr(result)[:150]
        accepted = True
    except _CaseTimeout:
        timed_out = True
        out["error"] = f"TIMEOUT after {CASE_TIMEOUT_SECONDS}s (likely combinatorial blowup)"
        accepted = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if timed_out:
        out["verdict"] = "TIMEOUT"
    elif misparsed:
        out["verdict"] = "MISPARSED (silently split into a different program, not rejected)"
    elif case["validity"] == "DOC-VALID":
        out["verdict"] = "FALSE REJECT" if not accepted else "ok"
    elif case["validity"] == "DOC-INVALID":
        out["verdict"] = "SILENTLY ACCEPTED (no validation observed)" if accepted else "REJECTED (correct)"
    else:
        out["verdict"] = "accepted (no oracle)" if accepted else "raised (no oracle)"
    out["ok"] = out["verdict"] not in (
        "FALSE REJECT", "SILENTLY ACCEPTED (no validation observed)",
        "MISPARSED (silently split into a different program, not rejected)",
    )
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the gate check (debug only)")
    args = ap.parse_args()

    gate = check_gate(args.force)
    test_root = gate.get("test_root")
    if not test_root:
        sys.exit("REFUSING TO RUN: no test_root recorded in the gate file.")
    test_root_path = pathlib.Path(test_root).resolve()
    search_roots = [test_root_path, test_root_path.parent, test_root_path.parent.parent]

    # many example-based tests import from an `examples.*` package that only
    # resolves relative to the checkout root, not site-packages — without
    # this, discovery silently finds nothing for any test that uses it.
    for root in search_roots:
        if (root / "examples").is_dir():
            sys.path.insert(0, str(root))
            break

    by_file: dict[pathlib.Path, set[str]] = {}
    for entry in gate.get("passing_tests", []):
        classname_and_name = entry["test"]
        if "::" in classname_and_name:
            classname, funcname = classname_and_name.split("::", 1)
        else:
            *cls_parts, funcname = classname_and_name.split(".")
            classname = ".".join(cls_parts)
        f = resolve_test_file(classname, search_roots)
        if f is not None:
            by_file.setdefault(f, set()).add(funcname.split(".")[-1])

    mode_a, mode_b = [], []
    for file_path, func_names in by_file.items():
        try:
            tree = ast.parse(file_path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        import_map = build_import_map(tree)
        mode_a.extend(discover_mode_a(file_path, func_names, tree, import_map))
        mode_b.extend(discover_mode_b(file_path, func_names, tree, import_map))

    if not mode_a and not mode_b:
        sys.exit("No focal calls discovered in any passing test (neither a `::` DSL string nor a "
                  "resolvable call with literal numeric/bool arguments). Nothing to fuzz.")

    corpus = build_corpus(mode_a, mode_b)
    print(f"Discovered {len(mode_a)} Mode-A (DSL-string) call(s) and {len(mode_b)} Mode-B "
          f"(general-argument) call(s) in passing tests:")
    for c in mode_a:
        print(f"  [A] {c['test']:<35} -> {c['focal_module']}.{c['focal_attr']}(<program string>)")
    for c in mode_b:
        print(f"  [B] {c['test']:<35} -> {c['focal_module']}.{c['focal_attr']}"
              f"({', '.join(f'{k}={v!r}' for k, v in c['all_args'].items())})")
    print(f"\nGenerated {len(corpus)} mutation cases\n")

    if args.list:
        for c in corpus:
            print(f"[{c['mode']}] {c['op']:<15} {c['validity']:<10} value={c['value']!r:<10} "
                  f"<- {c['from_test']} via {c['focal_module']}.{c['focal_attr']}")
        return 0

    t0 = time.time()
    rows = [run_case(c) for c in corpus]

    payload = {
        "library": "deepstochlog",
        "mode": "focal_method_mutation_v2_general",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
        "gate": gate,
        "mode_a_calls": mode_a, "mode_b_calls": mode_b,
        "scope_note": "Mode A mutates the probability literal in a real DSL program string and "
                       "calls the real parser. Mode B mutates literal int/float/bool arguments of "
                       "ANY resolvable focal call and calls that real function directly. Validity "
                       "for Mode B is a presumed non-negativity heuristic (stated, not a proven "
                       "contract) — bool/sign-ambiguous args get no oracle and are reported, not scored.",
        "n_cases": len(rows),
        "n_violations": sum(1 for r in rows if not r["ok"]),
        "results": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fuzz_results_deepstochlog_auto.json").write_text(json.dumps(payload, indent=1, default=str))

    print(f"{len(rows)} cases — {payload['n_violations']} presumed violations "
          f"(Mode-B N/A cases are reported, not scored)\n")
    for r in rows:
        mark = "!" if not r["ok"] else " "
        shown = r.get("error") or r.get("result_repr")
        print(f"{mark}[{r['mode']}] {r['op']:<15} {r['verdict']:<38} {str(shown)[:45]}  <- {r['from_test']}")
    print(f"\nwrote {RESULTS / 'fuzz_results_deepstochlog_auto.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
