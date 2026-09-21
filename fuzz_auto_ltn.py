#!/usr/bin/env python3
"""
fuzz_auto_ltn.py — focal-method fuzzer for LTN (LTNtorch), same methodology as
fuzz_auto_deepstochlog.py: baseline-gated, never a synthetic reimplementation,
every call goes to the REAL production class.

Two independent fuzzing modes now live here:

MODE 1 — the `p` hyper-parameter contract (unchanged from before). AST-
discovers real call sites with literal numeric/bool arguments inside LTN's
OWN passing test file, mutates those literals, and calls the REAL class
directly. LTN has no string DSL, so there is no Mode-A (DSL-string) target,
and almost every call in tests/tests.py passes torch.nn.Module instances or
lambdas, not literals. Direct AST inspection found exactly one real
literal-argument attack surface: the `p` hyper-parameter of the two
power-mean aggregation operators, `ltn.fuzzy_ops.AggregPMean` /
`AggregPMeanError`, constructed in test_Quantifier as `AggregPMean(p=2)` /
`AggregPMeanError(p=2)`. `stable` (bool) kwargs on AndProd/OrProbSum/
ImpliesReichenbach/ImpliesGoguen are discovered too but deliberately
excluded from scoring: LTN's own docstrings document `stable=False` as a
supported, valid alternate mode, not a contract violation.

2026-09-21a discovery-completeness fix: discovery used to only look at
KEYWORD arguments on a fully-qualified dotted call target. It now also
binds positional arguments to the resolved class's real `__init__`
signature (via `inspect.signature`) and resolves a bare name pulled in via
this file's own `from ltn import LTNObject, Constant, ...` line, not only a
dotted `ltn.fuzzy_ops.X` path. Re-run against the real file, this surfaces
the SAME 23 AggregPMean/AggregPMeanError(p=2) sites and the same 4
stable=False sites as before — zero new call sites, because every other
bare-name-imported constructor call in this file pairs a literal with a
real tensor/model/lambda that can't be reconstructed from the call site
alone. Recorded so this isn't re-investigated later expecting a different
answer.

MODE 2 — the truth-value domain contract (new, 2026-09-21b). Mode 1 only
ever exercised ONE documented contract (`p >= 1`) on TWO classes. But
reading `ltn/core.py` directly turned up a much broader, independently
enforced contract that applies to nearly everything else in the test file:

  - `ltn.core.Connective.__call__` (core.py, ~line 1222): before applying
    the wrapped connective operator, it calls
    `ltn.fuzzy_ops.check_values(*[o.value for o in operands])`, which
    raises ValueError if ANY operand's value tensor has an entry outside
    [0., 1.]. Its own docstring: "an LTN connective can be applied only to
    LTN objects containing truth values, namely values in [0., 1.]."
  - `ltn.core.Quantifier.__call__` (core.py, ~line 1471) calls the exact
    same `check_values(formula.value)` on the formula before aggregating.
  - `ltn.core.Predicate.__call__` (core.py, ~line 618-620) inlines the
    identical range check on its OWN output: "Expected the output of a
    predicate to be in the range [0., 1.] ... Check your predicate
    implementation!" — a third, independent implementation of the same
    rule, worth testing separately since a bug in one wouldn't imply a bug
    in the others.

None of this needs AST discovery: `Connective`/`Quantifier`/`Predicate`
apply to a fixed vocabulary of real operator classes actually constructed
in tests/tests.py (test_Connective, test_Quantifier), so the corpus below
is hand-declared — same pattern fuzz_auto_lnn.py already uses when AST
discovery finds nothing to work with — but every target is a class the
test file itself imports and constructs, and every value mutation is
grounded directly in check_values' own boundary condition
(`v >= 0. and v <= 1.`, both inclusive). This covers 18 connective
constructions (14 distinct classes, 4 of them exercised in both
stable=True and stable=False mode since the test file constructs both),
4 aggregators, and the Predicate mechanism itself — the great majority of
the real functions the test file actually exercises, not just the two
AggregP* classes Mode 1 was limited to. 18 + 4 + 1 = 23 targets x 10
truth-value mutations = 230 Mode 2 cases.

Usage:
    python fuzz_auto_ltn.py --list            # print discovered/declared targets, run nothing
    python fuzz_auto_ltn.py                   # gate-check, then run both modes
    python fuzz_auto_ltn.py --force           # skip the gate check (debug only)
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import pathlib
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
GATE_FILE = RESULTS / "ltn_tests.json"

CASE_TIMEOUT_SECONDS = 10


class _CaseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _CaseTimeout(f"case exceeded {CASE_TIMEOUT_SECONDS}s")


# ---------------------------------------------------------------------------
# Gate: refuse to run unless LTN's own suite actually passed something
# ---------------------------------------------------------------------------

def check_gate(force: bool) -> dict:
    if force:
        return {"gate": "skipped", "reason": "--force"}
    if not GATE_FILE.exists():
        sys.exit(f"REFUSING TO RUN: {GATE_FILE} missing. "
                  f"Run `python run_library_tests.py --only ltn --auto-clone` first.")
    data = json.loads(GATE_FILE.read_text())
    status = data.get("status")
    passed = data.get("junit", {}).get("counts", {}).get("passed", 0)
    if status not in ("ok", "tests_failed") or passed == 0:
        sys.exit(f"REFUSING TO RUN: baseline shows status={status!r}, passed={passed}. "
                  f"Fuzzing only runs for libraries whose own tests passed.")
    print(f"Gate OK: baseline passed={passed} — proceeding.")
    return {"gate": "passed", "baseline_passed": passed, "baseline_status": status,
            "test_root": data.get("test_root")}


# ---------------------------------------------------------------------------
# MODE 1 — discovery: find real call sites in tests/tests.py with literal
# numeric/bool arguments — positional OR keyword — resolved either via a
# fully-qualified dotted chain (`ltn.fuzzy_ops.AggregPMean`) or via a bare
# name this file itself imported with `from ltn import ...`.
# ---------------------------------------------------------------------------

@dataclass
class CallSite:
    qualname: str            # e.g. "ltn.fuzzy_ops.AggregPMean" or "ltn.Constant"
    lineno: int
    kwargs: dict              # literal args found, bound to their real parameter
                               # names (positional args included, via the real
                               # class's own __init__ signature)
    has_call_method: bool = False


def resolve_literal(node: ast.AST):
    """Return a literal Python value for simple int/float/bool AST nodes, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = resolve_literal(node.operand)
        return -inner if inner is not None else None
    return None


def attr_chain_name(node: ast.AST) -> str | None:
    """Turn `ltn.fuzzy_ops.AggregPMean` (an Attribute/Name chain) into a dotted string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def build_import_map(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Map a bare name pulled in via `from <module> import <name> [as alias]`
    to (module, real_name) — e.g. {"Constant": ("ltn", "Constant")}."""
    import_map: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_map[alias.asname or alias.name] = (node.module, alias.name)
    return import_map


def resolve_callable(qualname: str):
    """Resolve a dotted name like 'ltn.fuzzy_ops.AggregPMean' to the real object,
    by importing progressively longer module prefixes."""
    parts = qualname.split(".")
    for split in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    return None


def resolve_call_target(func_node: ast.AST, import_map: dict[str, tuple[str, str]]):
    """Resolve whatever a Call's `.func` node refers to: a dotted attribute
    chain rooted at an importable module, or a bare name this test file
    itself pulled in with `from <module> import <name>`."""
    qualname = attr_chain_name(func_node)
    if qualname is not None and "." in qualname:
        obj = resolve_callable(qualname)
        if obj is not None:
            return qualname, obj
    if isinstance(func_node, ast.Name) and func_node.id in import_map:
        module_name, real_name = import_map[func_node.id]
        try:
            mod = importlib.import_module(module_name)
            obj = getattr(mod, real_name)
            return f"{module_name}.{real_name}", obj
        except (ImportError, AttributeError):
            return None, None
    if qualname is not None:
        obj = resolve_callable(qualname)
        if obj is not None:
            return qualname, obj
    return None, None


def discover_call_sites(tree: ast.Module, import_map: dict[str, tuple[str, str]]) -> list[CallSite]:
    """Find every Call that resolves to a real class and supplies EVERY
    argument it writes out (positional or keyword) as a literal
    int/float/bool, with at least one numeric/bool. See module docstring
    for the 2026-09-21a completeness fix and its (null) effect on results.
    """
    sites: list[CallSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualname, target = resolve_call_target(node.func, import_map)
        if target is None or not inspect.isclass(target):
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.values())
        bound: dict = {}
        ok = True
        for i, arg in enumerate(node.args):
            if i >= len(params):
                ok = False
                break
            val = resolve_literal(arg)
            if val is None:
                ok = False
                break
            bound[params[i].name] = val
        if ok:
            for kw in node.keywords:
                if kw.arg is None:
                    ok = False
                    break
                val = resolve_literal(kw.value)
                if val is None:
                    ok = False
                    break
                bound[kw.arg] = val
        if not ok or not bound:
            continue
        if not any(isinstance(v, (int, float, bool)) for v in bound.values()):
            continue
        sites.append(CallSite(qualname=qualname, lineno=node.lineno, kwargs=bound))
    return sites


# ---------------------------------------------------------------------------
# MODE 1 — mutation and oracle for the `p` parameter
# ---------------------------------------------------------------------------

NAN, INF = float("nan"), float("inf")


def mutate_numeric(name: str, v0):
    """Mutations for an int/float literal. Returns [(op, value), ...]."""
    if isinstance(v0, bool):
        return [("flip", not v0)]
    muts = [
        ("zero", 0),
        ("negate", -v0 if v0 != 0 else -1),
        ("half", v0 / 2 if v0 else 0.5),
        ("fractional_below_one", 0.5),
        ("nan", NAN),
        ("pos_inf", INF),
        ("neg_inf", -INF),
        ("huge", v0 * 1_000_000 if v0 else 1_000_000),
    ]
    seen = set()
    out = []
    for op, val in muts:
        if op in seen:
            continue
        seen.add(op)
        out.append((op, val))
    return out


P_DOCUMENTED_CLASSES = {"ltn.fuzzy_ops.AggregPMean", "ltn.fuzzy_ops.AggregPMeanError"}


def p_validity(p) -> str:
    """DOC-VALID iff p is a finite number >= 1 (AggregPMean*'s own docstring
    requirement, read directly from ltn/fuzzy_ops.py)."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return "DOC-INVALID"
    if not math.isfinite(f):
        return "DOC-INVALID"
    return "DOC-VALID" if f >= 1 else "DOC-INVALID"


TEST_XS = None  # set lazily to a torch tensor in [0,1]


def run_p_case(cls_qualname: str, param_name: str, mutant_p) -> dict:
    """Construct the REAL AggregPMean/AggregPMeanError with the mutated p and
    call it exactly as the library's own docstring example does."""
    global TEST_XS
    import torch
    if TEST_XS is None:
        TEST_XS = torch.tensor([0.2, 0.5, 0.9])

    out = {"class": cls_qualname, "param": param_name, "p": mutant_p,
           "validity": p_validity(mutant_p)}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        cls = resolve_callable(cls_qualname)
        op = cls(**{param_name: mutant_p})
        result = op(TEST_XS, dim=0)
        val = float(result)
        out["result"] = val
        out["in_range"] = math.isfinite(val) and -1e-6 <= val <= 1 + 1e-6
        accepted = True
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        accepted = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if accepted is None:
        out["verdict"] = "TIMEOUT"
    elif out["validity"] == "DOC-VALID":
        out["verdict"] = ("FALSE REJECT" if not accepted else
                           "OUT OF RANGE" if not out["in_range"] else "ok")
    else:
        out["verdict"] = "REJECTED (correct)" if not accepted else "SILENTLY ACCEPTED"
    out["ok"] = out["verdict"] in ("ok", "REJECTED (correct)")
    return out


def run_generic_case(site: CallSite, param_name: str, op_name: str, mutant_val) -> dict:
    """For discovered call sites outside the documented p-contract (the
    `stable` bool flags): construct the real object and record what
    happens, but never assign a verdict — no documented contract to check."""
    out = {"class": site.qualname, "param": param_name, "mutation": op_name,
           "value": mutant_val, "validity": "N/A"}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        cls = resolve_callable(site.qualname)
        kwargs = dict(site.kwargs)
        kwargs[param_name] = mutant_val
        obj = cls(**kwargs)
        out["constructed"] = repr(obj)
        out["verdict"] = "constructed ok (no documented contract to check)"
        out["ok"] = True
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        out["verdict"] = "TIMEOUT"
        out["ok"] = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        out["verdict"] = "rejected at construction"
        out["ok"] = True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return out


# ---------------------------------------------------------------------------
# MODE 2 — truth-value domain contract. Grounded directly in ltn/core.py:
#   Connective.__call__  -> ltn.fuzzy_ops.check_values(*operand values)
#   Quantifier.__call__  -> ltn.fuzzy_ops.check_values(formula.value)
#   Predicate.__call__   -> inline range check on its own output
# check_values' own condition (read from ltn/fuzzy_ops.py) is
# `v >= 0. and v <= 1.`, both bounds INCLUSIVE — so 0.0 and 1.0 are the
# DOC-VALID boundary controls, everything outside is DOC-INVALID.
# ---------------------------------------------------------------------------

TRUTH_VALUE_MUTATIONS = [
    ("valid_zero", 0.0, "DOC-VALID"),
    ("valid_one", 1.0, "DOC-VALID"),
    ("valid_mid", 0.5, "DOC-VALID"),
    ("invalid_just_below_zero", -1e-6, "DOC-INVALID"),
    ("invalid_below", -0.5, "DOC-INVALID"),
    ("invalid_just_above_one", 1.0 + 1e-6, "DOC-INVALID"),
    ("invalid_above", 1.5, "DOC-INVALID"),
    ("invalid_nan", NAN, "DOC-INVALID"),
    ("invalid_pos_inf", INF, "DOC-INVALID"),
    ("invalid_neg_inf", -INF, "DOC-INVALID"),
]


def get_connective_targets():
    """Every connective actually constructed in tests/tests.py's
    test_Connective, hand-declared (there's no literal argument to
    AST-discover here — the target IS the class itself), each paired with
    its real arity (1 = UnaryConnectiveOperator, 2 = Binary)."""
    import ltn
    fo = ltn.fuzzy_ops
    return [
        ("NotStandard", 1, lambda: fo.NotStandard()),
        ("NotGodel", 1, lambda: fo.NotGodel()),
        ("AndMin", 2, lambda: fo.AndMin()),
        ("AndProd(stable=True)", 2, lambda: fo.AndProd(stable=True)),
        ("AndProd(stable=False)", 2, lambda: fo.AndProd(stable=False)),
        ("AndLuk", 2, lambda: fo.AndLuk()),
        ("OrMax", 2, lambda: fo.OrMax()),
        ("OrProbSum(stable=True)", 2, lambda: fo.OrProbSum(stable=True)),
        ("OrProbSum(stable=False)", 2, lambda: fo.OrProbSum(stable=False)),
        ("OrLuk", 2, lambda: fo.OrLuk()),
        ("ImpliesKleeneDienes", 2, lambda: fo.ImpliesKleeneDienes()),
        ("ImpliesGodel", 2, lambda: fo.ImpliesGodel()),
        ("ImpliesReichenbach(stable=True)", 2, lambda: fo.ImpliesReichenbach(stable=True)),
        ("ImpliesReichenbach(stable=False)", 2, lambda: fo.ImpliesReichenbach(stable=False)),
        ("ImpliesGoguen(stable=True)", 2, lambda: fo.ImpliesGoguen(stable=True)),
        ("ImpliesGoguen(stable=False)", 2, lambda: fo.ImpliesGoguen(stable=False)),
        ("ImpliesLuk", 2, lambda: fo.ImpliesLuk()),
        ("Equiv", 2, lambda: fo.Equiv(fo.AndProd(), fo.ImpliesReichenbach())),
    ]


def get_aggregator_targets():
    """Every aggregator actually constructed in tests/tests.py's
    test_Quantifier, wrapped in a real Quantifier to exercise the formula
    domain check (a DIFFERENT contract from Mode 1's `p >= 1` check on
    these same two AggregP* classes)."""
    import ltn
    fo = ltn.fuzzy_ops
    return [
        ("AggregMin", lambda: fo.AggregMin()),
        ("AggregMean", lambda: fo.AggregMean()),
        ("AggregPMean", lambda: fo.AggregPMean()),
        ("AggregPMeanError", lambda: fo.AggregPMeanError()),
    ]


def run_connective_domain_case(label, arity, make_op, op_name, mutant_value, expected_validity) -> dict:
    """Build the REAL ltn.Connective wrapping the real operator, feed it a
    real LTNObject whose value tensor is the mutated truth value (broadcast
    over 5 individuals, matching test_Connective's own `op1 = LTNObject(...,
    ["x"])` idiom), and check whether Connective.__call__'s own
    check_values() call enforces its documented [0., 1.] contract."""
    import torch
    import ltn
    out = {"target": label, "kind": "connective", "op": op_name,
           "value": mutant_value, "validity": expected_validity}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        conn = ltn.Connective(make_op())
        v = float(mutant_value)
        operand1 = ltn.LTNObject(torch.full((5,), v, dtype=torch.float32), ["x"])
        if arity == 1:
            result = conn(operand1)
        else:
            operand2 = ltn.LTNObject(torch.full((5,), 0.5, dtype=torch.float32), ["x"])
            result = conn(operand1, operand2)
        out["result_sample"] = float(result.value.flatten()[0])
        accepted = True
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        accepted = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if accepted is None:
        out["verdict"] = "TIMEOUT"
        out["ok"] = False
    elif expected_validity == "DOC-VALID":
        out["verdict"] = "ok" if accepted else "FALSE REJECT"
        out["ok"] = accepted
    else:
        out["verdict"] = "SILENTLY ACCEPTED" if accepted else "REJECTED (correct)"
        out["ok"] = not accepted
    return out


def run_quantifier_domain_case(label, make_agg, op_name, mutant_value, expected_validity) -> dict:
    """Build the REAL ltn.Quantifier wrapping the real aggregator, feed it a
    real formula LTNObject whose value tensor is the mutated truth value,
    and check whether Quantifier.__call__'s own check_values(formula.value)
    call enforces the same documented [0., 1.] contract — independent of
    Mode 1's `p >= 1` check on the same AggregPMean/AggregPMeanError
    classes."""
    import torch
    import ltn
    out = {"target": label, "kind": "quantifier_formula", "op": op_name,
           "value": mutant_value, "validity": expected_validity}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        quant = ltn.Quantifier(make_agg(), "e")
        v = float(mutant_value)
        n = 5
        formula = ltn.LTNObject(torch.full((n,), v, dtype=torch.float32), ["x"])
        xvar = ltn.Variable("x", torch.zeros((n, 1)))
        result = quant(xvar, formula)
        out["result_sample"] = float(result.value)
        accepted = True
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        accepted = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if accepted is None:
        out["verdict"] = "TIMEOUT"
        out["ok"] = False
    elif expected_validity == "DOC-VALID":
        out["verdict"] = "ok" if accepted else "FALSE REJECT"
        out["ok"] = accepted
    else:
        out["verdict"] = "SILENTLY ACCEPTED" if accepted else "REJECTED (correct)"
        out["ok"] = not accepted
    return out


def run_predicate_domain_case(op_name, mutant_value, expected_validity) -> dict:
    """Build a REAL ltn.Predicate whose func passes its input straight
    through (the same 'unbounded output' idiom test_Predicate itself uses
    for `wrong_predicate`), feed it a Variable whose value IS the mutated
    truth value, and check whether Predicate.__call__'s own inline output
    range check enforces the documented [0., 1.] contract — a THIRD,
    independent implementation of the same rule (not a call through
    check_values at all), worth testing separately."""
    import torch
    import ltn
    out = {"target": "Predicate(identity-like func)", "kind": "predicate_output",
           "op": op_name, "value": mutant_value, "validity": expected_validity}
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        pred = ltn.Predicate(func=lambda t: t.squeeze(-1))
        v = float(mutant_value)
        n = 5
        xvar = ltn.Variable("x", torch.full((n,), v, dtype=torch.float32))
        result = pred(xvar)
        out["result_sample"] = float(result.value.flatten()[0])
        accepted = True
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        accepted = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if accepted is None:
        out["verdict"] = "TIMEOUT"
        out["ok"] = False
    elif expected_validity == "DOC-VALID":
        out["verdict"] = "ok" if accepted else "FALSE REJECT"
        out["ok"] = accepted
    else:
        out["verdict"] = "SILENTLY ACCEPTED" if accepted else "REJECTED (correct)"
        out["ok"] = not accepted
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print discovered/declared targets, run nothing")
    ap.add_argument("--force", action="store_true", help="skip the gate check (debug only)")
    args = ap.parse_args()

    gate = check_gate(args.force)
    test_root = pathlib.Path(gate.get("test_root") or "")
    test_file = test_root / "tests.py" if test_root and (test_root / "tests.py").exists() else None
    if test_file is None:
        candidates = [test_root / "tests.py", pathlib.Path("/tmp/ltn/tests/tests.py")]
        test_file = next((c for c in candidates if c.exists()), None)
    if test_file is None:
        sys.exit("Could not locate tests/tests.py from the gate file's test_root; "
                 "pass its real path or rerun run_library_tests.py --only ltn --auto-clone.")

    tree = ast.parse(test_file.read_text())
    import_map = build_import_map(tree)
    sites = discover_call_sites(tree, import_map)

    n_connective_targets = len(get_connective_targets())
    n_aggregator_targets = len(get_aggregator_targets())
    n_domain_cases = (n_connective_targets + n_aggregator_targets + 1) * len(TRUTH_VALUE_MUTATIONS)

    if args.list:
        print(f"MODE 1 (p-contract, AST-discovered): {len(sites)} call site(s) that resolve to a "
              f"real class and supply EVERY written argument as a literal int/float/bool, in {test_file}:\n")
        for s in sites:
            tag = "P-DOCUMENTED (p>=1)" if s.qualname in P_DOCUMENTED_CLASSES else "undocumented contract"
            print(f"  line {s.lineno:<5} {s.qualname:<35} kwargs={s.kwargs}  [{tag}]")
        print(f"\nMODE 2 (truth-value domain contract, hand-declared — see module docstring): "
              f"{n_connective_targets} connective(s) + {n_aggregator_targets} aggregator(s) + 1 Predicate "
              f"mechanism, each x {len(TRUTH_VALUE_MUTATIONS)} value mutations = {n_domain_cases} cases\n")
        for label, arity, _ in get_connective_targets():
            print(f"  [connective, arity={arity}] {label}")
        for label, _ in get_aggregator_targets():
            print(f"  [quantifier/formula]        {label}")
        print(f"  [predicate output]           Predicate(identity-like func)")
        return 0

    warnings.filterwarnings("ignore")
    t0 = time.time()

    # --- Mode 1 ---
    p_rows: list[dict] = []
    generic_rows: list[dict] = []
    for site in sites:
        if site.qualname in P_DOCUMENTED_CLASSES and "p" in site.kwargs:
            v0 = site.kwargs["p"]
            for op_name, mutant in mutate_numeric("p", v0):
                p_rows.append({**run_p_case(site.qualname, "p", mutant),
                                "op": op_name, "seed_value": v0, "site_line": site.lineno})
        else:
            for pname, v0 in site.kwargs.items():
                for op_name, mutant in mutate_numeric(pname, v0):
                    generic_rows.append({**run_generic_case(site, pname, op_name, mutant),
                                          "seed_value": v0, "site_line": site.lineno})

    # --- Mode 2 ---
    domain_rows: list[dict] = []
    for label, arity, make_op in get_connective_targets():
        for op_name, mutant, validity in TRUTH_VALUE_MUTATIONS:
            domain_rows.append(run_connective_domain_case(label, arity, make_op, op_name, mutant, validity))
    for label, make_agg in get_aggregator_targets():
        for op_name, mutant, validity in TRUTH_VALUE_MUTATIONS:
            domain_rows.append(run_quantifier_domain_case(label, make_agg, op_name, mutant, validity))
    for op_name, mutant, validity in TRUTH_VALUE_MUTATIONS:
        domain_rows.append(run_predicate_domain_case(op_name, mutant, validity))

    n_violations = sum(1 for r in p_rows if not r["ok"])
    n_domain_violations = sum(1 for r in domain_rows if not r["ok"])

    payload = {
        "library": "ltn",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
        "gate": gate,
        "test_file": str(test_file),
        "mode_1_method": (
            "AST-discovered every Call in ltn's own passing test file that resolves to a real class "
            "(fully-qualified path or a bare name this file imported with `from ltn import ...`) and "
            "supplies EVERY written argument as a literal int/float/bool. Mutated the literal(s) and "
            "invoked the REAL class exactly as the library's own docstring/tests do."
        ),
        "mode_1_scope_note": (
            "Only ltn.fuzzy_ops.AggregPMean / AggregPMeanError's `p` parameter has a documented contract "
            "(p >= 1) found this way. The `stable` bool flags on AndProd/OrProbSum/ImpliesReichenbach/"
            "ImpliesGoguen are recorded under generic_results with validity=N/A and never counted as "
            "violations, because LTN's own docstrings document stable=False as a supported alternate mode."
        ),
        "discovery_completeness_note": (
            "2026-09-21a: discovery now also binds positional literal arguments and resolves bare names "
            "imported via `from ltn import ...`, not only fully-qualified dotted paths. Re-run against the "
            "real file, this is a null result -- same 23 + 4 sites as before, because every other bare-name "
            "constructor call in this file pairs a literal with a real tensor/model/lambda that can't be "
            "reconstructed from the call site alone."
        ),
        "mode_2_method": (
            "2026-09-21b: read ltn/core.py directly and found the truth-value domain contract enforced "
            "independently in three places: Connective.__call__ (check_values on all operands), "
            "Quantifier.__call__ (check_values on the formula), and Predicate.__call__ (an inline range "
            "check on its own output). Hand-declared every connective/aggregator actually constructed in "
            "tests/tests.py's test_Connective/test_Quantifier (there's no literal argument to discover here "
            "-- the class itself is the target), wrapped each in the real ltn.Connective/ltn.Quantifier, and "
            "fed it a real LTNObject/Variable whose value tensor is set to boundary or out-of-range values "
            "(0.0 and 1.0 inclusive boundaries per check_values' own `v >= 0. and v <= 1.` condition; "
            "-1e-6, -0.5, 1+1e-6, 1.5, NaN, +inf, -inf as the invalid probes). Predicate is tested the same "
            "way with an identity-like func (the same 'unbounded output' idiom test_Predicate's own "
            "`wrong_predicate` uses)."
        ),
        "n_call_sites_discovered": len(sites),
        "n_p_cases": len(p_rows),
        "n_generic_cases": len(generic_rows),
        "n_violations": n_violations,
        "n_domain_cases": len(domain_rows),
        "n_domain_violations": n_domain_violations,
        "p_results": p_rows,
        "generic_results": generic_rows,
        "domain_results": domain_rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "fuzz_results_ltn_auto.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    print(f"\nMode 1: {len(p_rows)} p-contract cases — {n_violations} oracle violations "
          f"({len(generic_rows)} generic cases recorded, no contract to violate)")
    print(f"Mode 2: {len(domain_rows)} truth-value domain cases — {n_domain_violations} oracle violations\n")
    for r in p_rows:
        mark = "!" if not r["ok"] else " "
        print(f"{mark}[p]      {r['op']:<20} {r['verdict']:<20} "
              f"{str(r.get('error') or r.get('result'))[:60]}")
    for r in domain_rows:
        mark = "!" if not r["ok"] else " "
        print(f"{mark}[domain] {r['target']:<32} {r['op']:<24} {r['verdict']:<20} "
              f"{str(r.get('error') or r.get('result_sample'))[:60]}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
