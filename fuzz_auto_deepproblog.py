#!/usr/bin/env python3
"""
fuzz_auto_deepproblog.py — restores DeepProbLog coverage to the "auto" fuzzer
generation (fuzz_auto_ltn.py, fuzz_auto_lnn.py, fuzz_auto_deepstochlog.py),
after the original ad hoc harness (fuzz_dpl.py / fuzz_dpl_b.py, 18 Sep 2026,
see claude/deepproblog-fuzz-run-2026-09-18.md) was folded into the generic
fuzz_inputs.py and no DeepProbLog-specific "auto" script existed any more.

Same discipline as the rest of this study:
  1. Refuses to run unless results/deepproblog_tests.json shows the
     library's OWN pytest suite actually passed something (passed > 0).
  2. Every case is grounded in something DeepProbLog's own passing tests
     (site-packages: test_engine.py, test_model.py, test_semiring.py,
     test_neural_predicate.py) actually construct — never a hand-invented
     program.
  3. Execution calls the REAL, documented production pipeline directly —
     `deepproblog.model.Model` -> `.set_engine(ExactEngine(...))` ->
     `.solve([Query(...)])` — exactly what every one of those test files'
     own `_create_model`/`_solve` helpers do. This is the same style as
     fuzz_auto_lnn.py: DeepProbLog's test suite has no single reusable
     "parse this program string" function the way DeepStochLog's
     `parse_rules` is (Mode A in fuzz_auto_deepstochlog.py) — the real
     entry point is always this multi-step pipeline, so that is what gets
     called, directly, with mutated arguments.

Two discovery surfaces, both found by walking the AST of DeepProbLog's own
passing test files (no hand-authored program text):

  SURFACE "program" — every `<number> :: pred(args).` / `t(<number>) ::
  pred(args).` probabilistic-fact clause found in a module- or
  function-scope string constant that is itself referenced (directly, or
  through the file's own `_create_model`/`_solve` helpers) inside a
  `Model(...)` call in the same file (test_model.py's `_simple_program`,
  test_semiring.py's per-test `program` strings). test_engine.py's
  `test_ad`/`test_ad2`/`test_fact`/... programs are NOT included here: that
  whole file does `pytest.skip(..., allow_module_level=True)` when
  ApproximateEngine/PySDD isn't installed, so none of it is in this
  project's "passing tests" — this script only ever scans files a passing
  test actually came from (see resolve_test_file / gate.passing_tests).
  Clauses are grouped by
  `;`-chaining into their annotated-disjunction branches (a lone clause is
  a group of one). Each branch is DIRECTLY queryable by its own predicate
  (`Query(parse("<pred(args)>."))`), so mutating one branch's literal and
  querying that branch's own name gives an exact oracle with no dependency
  on whatever the original test derived from it: DOC-VALID (finite,
  in [0,1]) must come back and equal the literal exactly (this is the
  library's own suite's oracle, taken verbatim — `assert 0.0 <= p <= 1.0`
  at deepproblog/tests/test_semiring.py:191 — generalised from "checked
  once in one test" to "checked on every literal the suite has").
  Branches whose predicate argument is the anonymous variable `_` are
  skipped: querying `_` binds a fresh variable each time and the result
  key is not something this script can match reliably against a fixed
  expected term, so it is left out rather than guessed at.

  SURFACE "belief" — the neural-predicate side of the interface. In
  test_neural_predicate.py, `dummy_values1/2/3` are literal
  `{Term("i1"): [floats...], ...}` dicts fed through the REAL `DummyNet`
  (deepproblog.utils.standard_networks.DummyNet) into a REAL `Network`,
  wired to a real `nn(name,[X],...) :: pred(X,...)` declaration in that
  file's own `program` string. The declaration's own arity (read directly
  from that string, not assumed) fixes the role, exactly as this project's
  prior run classified it by reading deepproblog/network.py /
  graph_semiring.py directly:
    nn(name,[X],Y,[d0,d1,...]) :: pred(X,Y).   -> nn/4, exhaustive AD:
        must be a probability vector over the domain, summing to 1
        (dummy1/net1 in the real file).
    nn(name,[X]) :: pred(X).                   -> nn/2, probabilistic
        fact: must be a single value in [0,1] (dummy2/net2).
    nn(name,[X],Y) :: pred(X,Y).                -> nn/3, deterministic
        tensor output, NOT a belief — correctly excluded, same as the
        original run's role classification (dummy3/net3).
  Only the `i1` key is mutated (the only one either real query needs); the
  seed mutant set is the literal vectors this project's own 18 Sep run
  reported as violations (F1 raw logits, F3 sigmoid/all-zero non-AD sums,
  F7 nan/inf) plus a `nan`/`inf` case, so this script is a direct, executed
  re-derivation of those findings through the "auto" discipline rather
  than a new claim.

Usage:
    python fuzz_auto_deepproblog.py --list      # print discovered corpus, run nothing
    python fuzz_auto_deepproblog.py             # gate-check, then run
    python fuzz_auto_deepproblog.py --force     # skip the gate check (debug only)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
import signal
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
GATE_FILE = RESULTS / "deepproblog_tests.json"

CASE_TIMEOUT_SECONDS = 10
NAN, INF = float("nan"), float("inf")


class _CaseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _CaseTimeout(f"case exceeded {CASE_TIMEOUT_SECONDS}s")


# ---------------------------------------------------------------------------
# Gate: refuse to run unless DeepProbLog's own suite actually passed something
# ---------------------------------------------------------------------------
def check_gate(force: bool) -> dict:
    if force:
        return {"gate": "skipped", "reason": "--force"}
    if not GATE_FILE.exists():
        sys.exit(f"REFUSING TO RUN: {GATE_FILE} missing. "
                  f"Run `python run_library_tests.py --only deepproblog` first.")
    data = json.loads(GATE_FILE.read_text())
    status = data.get("status")
    junit = data.get("junit", {})
    passed = junit.get("counts", {}).get("passed", 0)
    if status not in ("ok", "tests_failed") or passed == 0:
        sys.exit(f"REFUSING TO RUN: baseline shows status={status!r}, passed={passed}. "
                  f"Rerun `python run_library_tests.py --only deepproblog` against an "
                  f"environment that actually has deepproblog installed.")
    test_root = data.get("test_root")
    passing_tests = junit.get("passing_tests", [])
    print(f"Gate OK: baseline passed={passed} (test_root={test_root}) — proceeding.")
    return {"gate": "passed", "baseline_passed": passed, "baseline_status": status,
            "test_root": test_root, "passing_tests": passing_tests}


def resolve_test_file(classname: str, test_root: pathlib.Path) -> pathlib.Path | None:
    # deepproblog's own test suite is a flat directory (test_engine.py,
    # test_model.py, ...), classnames in the junit report have no dots.
    flat = test_root / f"{classname.split('::')[0].split('.')[-1]}.py"
    if flat.is_file():
        return flat
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = test_root.joinpath(*parts[:cut]).with_suffix(".py")
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# SURFACE "program" — discovery
# ---------------------------------------------------------------------------

# One AD branch: <number>::pred(args). / <number>::pred(args); / t(<lit>)::pred(args)[.;]
# `nn(...)::...` never matches (its annotation is neither a bare number nor t(...)).
CLAUSE_RE = re.compile(
    r"(?P<annot>-?\d+(?:\.\d+)?|t\(\s*(?P<tparam>-?[\w.+-]+)\s*\))"
    r"\s*::\s*"
    r"(?P<pred>[A-Za-z_]\w*(?:\([^()]*\))?)"
    r"\s*(?P<term>[.;])"
)


def _string_value(node: ast.AST) -> str | None:
    """A plain string constant, OR a `"<template>".format(...)` call on one
    -- e.g. test_semiring.py's `program = "{p} :: a.\\n0.5 :: b.\\n...".format(p=p)`.
    The pre-format template text is what gets scanned; a `{p}`-style
    placeholder never matches CLAUSE_RE, so only literals already concrete
    in the template (like the `0.5 :: b.` alongside it) are ever discovered
    -- nothing about the parametrize value itself is invented."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)):
        return node.func.value.value
    return None


def module_level_string_vars(tree: ast.Module) -> dict[str, str]:
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            sval = _string_value(node.value)
            if sval is not None:
                out[node.targets[0].id] = sval
    return out


def function_local_string_vars(fn: ast.FunctionDef) -> dict[str, str]:
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            sval = _string_value(node.value)
            if sval is not None:
                out[node.targets[0].id] = sval
    return out


def used_as_model_program(varname: str, source: str) -> bool:
    """Lightweight but real check: the variable actually appears as an
    argument inside a `Model(` call somewhere in the same file (covers the
    common case `Model(program, ...)` / `Model(_simple_program, ...)`), or
    inside a `_create_model(...)` / `_solve(...)` helper call -- every real
    test file that defines these (`_create_model(program) -> Model: model =
    Model(program, [], load=False); ...`, and `_solve(program, term=...):
    model = _create_model(program); ...`) is a thin wrapper that bottoms out
    in exactly that Model(...) call, confirmed by reading each definition
    directly rather than assumed."""
    return re.search(rf"(?:Model|_create_model|_solve)\(\s*{re.escape(varname)}\b", source) is not None


def parse_clause_groups(program_text: str) -> list[list[dict]]:
    """Split every `<annot>::pred(args)[.;]` match into AD-branch groups: a
    run of matches chained by ';' terminators, ended by the first '.'."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for m in CLAUSE_RE.finditer(program_text):
        annot_text = m.group("annot")
        tparam = m.group("tparam")
        if tparam is not None:
            kind = "tparam"
            try:
                lit = float(tparam)
            except ValueError:
                continue  # t(_) / t(X) -- not a literal, nothing to mutate
        else:
            kind = "lit"
            lit = float(annot_text)
        pred = m.group("pred")
        if re.search(r"\(\s*_\s*\)|,\s*_\s*[,)]|\(\s*_\s*,", pred):
            # anonymous-variable argument -- not reliably re-queryable, skip
            branch_skipped = True
        else:
            branch_skipped = False
        current.append({
            "start": m.start("annot"), "end": m.end("annot"),
            "annot_text": annot_text, "kind": kind, "literal": lit,
            "pred": pred, "skipped": branch_skipped,
        })
        if m.group("term") == ".":
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def discover_program_surface(test_file: pathlib.Path) -> list[dict]:
    source = test_file.read_text()
    tree = ast.parse(source)
    module_vars = module_level_string_vars(tree)

    # Per-function candidate dicts, kept SEPARATE: several real test files
    # (test_semiring.py in particular) have multiple `def test_*` functions
    # that each assign their own literal program text to a local variable
    # named `program` -- flattening those into one shared dict across the
    # whole file silently clobbers all but the last one found.
    scopes: list[dict[str, str]] = [module_vars]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            local = dict(module_vars)
            local.update(function_local_string_vars(node))
            scopes.append(local)

    discovered = []
    seen_programs = set()
    for candidates in scopes:
        for varname, text in candidates.items():
            if "::" not in text or text in seen_programs:
                continue
            if not used_as_model_program(varname, source):
                continue
            if re.search(r"\{[^{}]*\}", text):
                # a `.format(...)` template with placeholders OTHER than the
                # ones already resolved into concrete literals above (e.g.
                # test_semiring.py's `"{p} :: a.\n0.5 :: b.\n...".format(p=p)`)
                # can't be fed to Model() as-is -- every mutant built from it,
                # including the unmutated seed, would fail to parse on the
                # leftover placeholder, not on anything this script mutated.
                # Skip the whole program rather than report that as a finding.
                continue
            groups = parse_clause_groups(text)
            if not groups:
                continue
            seen_programs.add(text)
            discovered.append({
                "file": str(test_file), "var": varname, "program": text, "groups": groups,
            })
    return discovered


# ---------------------------------------------------------------------------
# SURFACE "belief" — discovery (test_neural_predicate.py's DummyNet dicts)
# ---------------------------------------------------------------------------

NN4_RE = re.compile(r"nn\(\s*(\w+)\s*,\s*\[[^\]]*\]\s*,\s*\w+\s*,\s*\[([^\]]*)\]\s*\)\s*::\s*(\w+)\(")
NN3_RE = re.compile(r"nn\(\s*(\w+)\s*,\s*\[[^\]]*\]\s*,\s*\w+\s*\)\s*::\s*(\w+)\(")
NN2_RE = re.compile(r"nn\(\s*(\w+)\s*,\s*\[[^\]]*\]\s*\)\s*::\s*(\w+)\(")


def dict_literal_i1_vector(node: ast.Dict) -> list | None:
    for k, v in zip(node.keys, node.values):
        is_i1 = (isinstance(k, ast.Call) and isinstance(k.func, ast.Name) and k.func.id == "Term"
                 and len(k.args) == 1 and isinstance(k.args[0], ast.Constant) and k.args[0].value == "i1")
        if not is_i1:
            continue
        if not isinstance(v, ast.List):
            return None
        vals = []
        for elt in v.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                vals.append(float(elt.value))
            else:
                return None
        return vals
    return None


def discover_belief_surface(test_file: pathlib.Path) -> list[dict]:
    source = test_file.read_text()
    tree = ast.parse(source)

    dict_vectors: dict[str, list] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict)):
            vec = dict_literal_i1_vector(node.value)
            if vec is not None:
                dict_vectors[node.targets[0].id] = vec

    # Network(DummyNet(<dictvar>), "<netname>") -- textual match is enough
    # here (both args are simple literals/names in every real call site) and
    # avoids re-deriving a full expression evaluator for one pattern.
    net_map: dict[str, str] = {}
    for m in re.finditer(r'Network\(\s*DummyNet\(\s*(\w+)\s*\)\s*,\s*"?([A-Za-z_]\w*)"?', source):
        dictvar, netname = m.group(1), m.group(2)
        if dictvar in dict_vectors:
            net_map[dictvar] = netname

    module_vars = module_level_string_vars(tree)
    program_text = next((v for v in module_vars.values() if "nn(" in v), None)

    discovered = []
    if program_text is not None:
        for dictvar, netname in net_map.items():
            role, domain, pred = None, None, None
            for m in NN4_RE.finditer(program_text):
                if m.group(1) == netname:
                    role = "nn4_ad"
                    domain = [d.strip() for d in m.group(2).split(",") if d.strip()]
                    pred = m.group(3)
                    break
            if role is None:
                for m in NN3_RE.finditer(program_text):
                    if m.group(1) == netname:
                        role = "nn3_deterministic"
                        pred = m.group(2)
                        break
            if role is None:
                for m in NN2_RE.finditer(program_text):
                    if m.group(1) == netname:
                        role = "nn2_fact"
                        pred = m.group(2)
                        break
            if role is None or role == "nn3_deterministic":
                continue  # not a belief -- correctly excluded, same as the original run
            discovered.append({
                "file": str(test_file), "dictvar": dictvar, "netname": netname,
                "pred": pred, "role": role, "domain": domain,
                "seed_vector": dict_vectors[dictvar],
            })
    return discovered


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate_literal(v0: float) -> list[tuple[str, float]]:
    return [
        ("negate", -v0 if v0 != 0 else -0.5),
        ("above_one", v0 + 1.0 if v0 <= 0.5 else v0 * 2.5),
        ("zero", 0.0),
        ("nan", NAN),
        ("pos_inf", INF),
        ("neg_inf", -INF),
    ]


def belief_vector_mutants(seed: list[float]) -> list[tuple[str, list[float]]]:
    n = len(seed)
    if n == 1:
        return [
            ("raw_negative", [-0.5]),
            ("raw_above_one", [1.7]),
            ("nan", [NAN]),
            ("pos_inf", [INF]),
        ]
    # Reuses the exact mutant vectors this project's 18 Sep run reported as
    # F1 (raw logits)/F3 (non-normalised sigmoid-style outputs)/F7 (nan/inf),
    # so this is a re-derivation of those findings, not a new invention.
    base = [
        ("raw_logits", [3.2, -1.1, 0.4]),
        ("raw_logits_2", [-0.5, 0.75, 0.75]),
        ("sigmoid_not_softmax", [0.9, 0.9, 0.9]),
        ("low_sigmoid", [0.1, 0.1, 0.1]),
        ("all_zero", [0.0, 0.0, 0.0]),
        ("nan", [NAN, 0.5, 0.5]),
        ("pos_inf", [INF, 0.0, 0.0]),
        ("neg_inf", [-INF, 1.0, 0.0]),
    ]
    out = []
    for op, vec in base:
        if len(vec) == n:
            out.append((op, vec))
        else:
            v = (vec + [0.0] * n)[:n]
            out.append((op, v))
    return out


def fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(float(v))


def literal_validity(v: float) -> str:
    return "DOC-VALID" if (isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0) else "DOC-INVALID"


# ---------------------------------------------------------------------------
# Execution — SURFACE "program": real Model -> ExactEngine -> solve pipeline
# ---------------------------------------------------------------------------

def run_program_case(program_text: str, branch: dict, mutant_op: str, mutant_val: float,
                      from_var: str, from_file: str, group_size: int) -> dict:
    out = {
        "surface": "program", "from_file": from_file, "from_var": from_var,
        "pred": branch["pred"], "kind": branch["kind"], "op": mutant_op,
        "value": mutant_val, "seed_value": branch["literal"], "group_size": group_size,
        "validity": literal_validity(mutant_val),
    }
    mutated = program_text[:branch["start"]] + fmt(mutant_val) + program_text[branch["end"]:]

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        from deepproblog.model import Model
        from deepproblog.engines import ExactEngine
        from deepproblog.query import Query
        from deepproblog.utils import parse

        model = Model(mutated, [], load=False)
        model.set_engine(ExactEngine(model))
        query = Query(parse(f"{branch['pred']}."))
        result = model.solve([query])[0].result
        val = result.get(query.query)
        if val is None:
            out["result"] = None
            out["error"] = "query had no solution (empty result dict)"
            accepted = False
        else:
            fval = float(val)
            out["result"] = fval
            out["in_range"] = math.isfinite(fval) and -1e-6 <= fval <= 1 + 1e-6
            out["exact_match"] = math.isfinite(fval) and abs(fval - mutant_val) < 1e-6
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
        if not accepted:
            out["verdict"] = "FALSE REJECT"
        elif not out["in_range"]:
            out["verdict"] = "OUT OF RANGE"
        elif not out["exact_match"]:
            out["verdict"] = "SILENT MISCOMPUTE (accepted, in range, but != literal)"
        else:
            out["verdict"] = "ok"
    else:  # DOC-INVALID
        out["verdict"] = "REJECTED (correct)" if not accepted else (
            "SILENTLY ACCEPTED (no validation observed)"
            if not out.get("exact_match") else
            "SILENTLY ACCEPTED, pass-through exact"
        )
    out["ok"] = out["verdict"] in ("ok", "REJECTED (correct)")
    return out


# ---------------------------------------------------------------------------
# Execution — SURFACE "belief": real DummyNet -> Network -> Model pipeline
# ---------------------------------------------------------------------------

def run_belief_case(entry: dict, op: str, mutant_vector: list[float]) -> dict:
    out = {
        "surface": "belief", "from_file": entry["file"], "netname": entry["netname"],
        "pred": entry["pred"], "role": entry["role"], "domain": entry["domain"],
        "op": op, "value": mutant_vector, "seed_value": entry["seed_vector"],
    }
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT_SECONDS)
    try:
        from deepproblog.model import Model
        from deepproblog.network import Network
        from deepproblog.engines import ExactEngine
        from deepproblog.query import Query
        from deepproblog.utils.standard_networks import DummyNet
        from problog.logic import Term

        values = {Term("i1"): mutant_vector}
        net = Network(DummyNet(values), entry["netname"])
        if entry["role"] == "nn4_ad":
            program = (f"nn({entry['netname']},[X],Y,[{','.join(entry['domain'])}]) "
                       f":: {entry['pred']}(X,Y).")
        else:  # nn2_fact
            program = f"nn({entry['netname']},[X]) :: {entry['pred']}(X)."
        model = Model(program, [net], load=False)
        model.set_engine(ExactEngine(model))

        if entry["role"] == "nn4_ad":
            branch_vals = {}
            for branch in entry["domain"]:
                q = Query(Term(entry["pred"], Term("i1"), Term(branch)))
                r = model.solve([q])[0].result
                v = r.get(q.query)
                branch_vals[branch] = float(v) if v is not None else None
            out["branch_values"] = branch_vals
            finite_vals = [v for v in branch_vals.values() if v is not None and math.isfinite(v)]
            all_finite = len(finite_vals) == len(branch_vals)
            all_in_range = all_finite and all(0.0 <= v <= 1.0 for v in finite_vals)
            total = sum(finite_vals) if all_finite else None
            out["sum"] = total
            out["all_in_range"] = all_in_range
            out["sums_to_one"] = total is not None and abs(total - 1.0) < 1e-4
            seed_is_valid = all(0.0 <= v <= 1.0 for v in mutant_vector) and math.isclose(sum(mutant_vector), 1.0, abs_tol=1e-4)
            out["validity"] = "DOC-VALID" if seed_is_valid else "DOC-INVALID"
            if op == "seed":
                out["verdict"] = "ok" if (all_finite and all_in_range and out["sums_to_one"]) else "HARNESS BROKEN"
            elif out["validity"] == "DOC-VALID":
                out["verdict"] = "ok" if (all_finite and all_in_range and out["sums_to_one"]) else "OUT OF RANGE / NOT NORMALISED"
            else:
                out["verdict"] = ("SILENTLY ACCEPTED (no validation observed)"
                                   if all_finite else "SILENTLY ACCEPTED (non-finite propagated)")
            out["ok"] = out["verdict"] == "ok"
        else:  # nn2_fact
            q = Query(Term(entry["pred"], Term("i1")))
            r = model.solve([q])[0].result
            v = r.get(q.query)
            fv = float(v) if v is not None else None
            out["result"] = fv
            finite = fv is not None and math.isfinite(fv)
            in_range = finite and 0.0 <= fv <= 1.0
            seed_is_valid = 0.0 <= mutant_vector[0] <= 1.0
            out["validity"] = "DOC-VALID" if seed_is_valid else "DOC-INVALID"
            if op == "seed":
                out["verdict"] = "ok" if in_range else "HARNESS BROKEN"
            elif out["validity"] == "DOC-VALID":
                out["verdict"] = "ok" if in_range else "OUT OF RANGE"
            else:
                out["verdict"] = "SILENTLY ACCEPTED (no validation observed)" if finite else "SILENTLY ACCEPTED (non-finite propagated)"
            out["ok"] = out["verdict"] == "ok"
    except _CaseTimeout as e:
        out["error"] = f"TIMEOUT: {e}"
        out["verdict"] = "TIMEOUT"
        out["ok"] = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        out["verdict"] = "REJECTED (correct)" if op != "seed" else "HARNESS BROKEN (seed rejected)"
        out["ok"] = out["verdict"] == "REJECTED (correct)"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gate = check_gate(args.force)
    test_root_str = gate.get("test_root")
    if not test_root_str:
        sys.exit("REFUSING TO RUN: no test_root recorded in the gate file.")
    test_root = pathlib.Path(test_root_str)

    # Only files that DeepProbLog's own suite actually passed at least one
    # test in -- e.g. test_engine.py skips its entire module
    # (`pytest.skip(..., allow_module_level=True)`) when PySDD/ApproximateEngine
    # isn't installed, and a skipped file is not "the passing tests".
    passing_files: set[pathlib.Path] = set()
    for entry in gate.get("passing_tests", []):
        classname = entry["test"].split("::")[0]
        f = resolve_test_file(classname, test_root)
        if f is not None:
            passing_files.add(f)
    test_files = sorted(passing_files)
    if not test_files:
        sys.exit(f"No passing test files resolved under {test_root} from the gate's "
                  f"passing_tests list.")

    program_surface: list[dict] = []
    belief_surface: list[dict] = []
    for f in test_files:
        try:
            program_surface.extend(discover_program_surface(f))
        except SyntaxError:
            pass
        try:
            belief_surface.extend(discover_belief_surface(f))
        except SyntaxError:
            pass

    n_branches = sum(1 for d in program_surface for g in d["groups"] for b in g if not b["skipped"])
    print(f"Discovered {len(program_surface)} program string(s) across {len(test_files)} test file(s), "
          f"{n_branches} mutable AD-branch literal(s); {len(belief_surface)} belief-vector surface(s).")

    if args.list:
        for d in program_surface:
            print(f"\n[program] {pathlib.Path(d['file']).name} :: {d['var']}")
            for g in d["groups"]:
                shown = " ; ".join(
                    f"{b['annot_text']}::{b['pred']}" + ("  [SKIPPED: anon var]" if b["skipped"] else "")
                    for b in g
                )
                print(f"    group(size={len(g)}): {shown}")
        for e in belief_surface:
            print(f"\n[belief]  {pathlib.Path(e['file']).name} :: {e['netname']} "
                  f"({e['role']}, pred={e['pred']}, domain={e['domain']}) seed={e['seed_vector']}")
        return 0

    t0 = time.time()
    program_rows: list[dict] = []
    for d in program_surface:
        for g in d["groups"]:
            for branch in g:
                if branch["skipped"]:
                    continue
                v0 = branch["literal"]
                for op, mutant in [("seed", v0)] + mutate_literal(v0):
                    program_rows.append(run_program_case(
                        d["program"], branch, op, mutant, d["var"], d["file"], len(g)))

    belief_rows: list[dict] = []
    for e in belief_surface:
        for op, vec in [("seed", e["seed_vector"])] + belief_vector_mutants(e["seed_vector"]):
            belief_rows.append(run_belief_case(e, op, vec))

    n_violations = sum(1 for r in program_rows if not r["ok"] and r.get("op") != "seed") + \
                   sum(1 for r in belief_rows if not r["ok"] and r.get("op") != "seed")

    payload = {
        "library": "deepproblog",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
        "gate": gate,
        "test_root": str(test_root),
        "method": (
            "AST-discovered every `<number>::pred(args).` / `t(<number>)::pred(args).` "
            "probabilistic-fact clause and every literal DummyNet belief vector actually "
            "referenced inside a Model(...)/Network(...) call in DeepProbLog's own passing "
            "test files, mutated each literal in place, and re-ran the REAL Model -> "
            "ExactEngine -> solve pipeline exactly as those tests' own helpers do."
        ),
        "n_program_strings": len(program_surface),
        "n_program_cases": len(program_rows),
        "n_belief_surfaces": len(belief_surface),
        "n_belief_cases": len(belief_rows),
        "n_violations": n_violations,
        "program_results": program_rows,
        "belief_results": belief_rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "fuzz_results_deepproblog_auto.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    print(f"\n{len(program_rows)} program-literal cases + {len(belief_rows)} belief-vector cases "
          f"— {n_violations} oracle violations\n")
    for r in program_rows:
        mark = "!" if not r["ok"] else " "
        print(f"{mark}[program] {r['pred']:<28} {r['op']:<12} {r['verdict']:<45} "
              f"{str(r.get('result', r.get('error')))[:40]}")
    for r in belief_rows:
        mark = "!" if not r["ok"] else " "
        print(f"{mark}[belief]  {r['netname']:<10} {r['op']:<20} {r['verdict']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
