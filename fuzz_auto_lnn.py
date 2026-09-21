#!/usr/bin/env python3
"""
fuzz_auto_lnn.py — fuzzes the neural-parameter interface of IBM's Logical
Neural Networks (LNN), same discipline as fuzz_auto_ltn.py / fuzz_auto_deepstochlog.py:
baseline-gated, calls the REAL production classes directly, and never invents
an undocumented contract.

Why this looks different from the LTN/DeepStochLog scripts:
  - LNN's own passing test suite (tests/**) was checked by AST/grep for
    literal `alpha=` / `weights=` / `bias=` keyword arguments — there are
    NONE. Every test that touches a connective's neural configuration passes
    it nested inside an `activation={...}` dict, and even then only ever
    sets `bias_learning` / `weights_learning` (booleans), never a literal
    weight, bias, or alpha value. So the "discover real literal call sites
    in the library's own tests" approach that worked for LTN/DeepStochLog
    finds nothing fuzzable on this surface.
  - Instead, this script targets the REAL, documented `activation: dict`
    parameter of `lnn.And` / `lnn.Or` directly (confirmed from their
    docstrings and by tracing `n_ary_neuron.py` -> `connective_neuron.py` ->
    `_NeuronParameters.__init__`, where `activation` is splatted straight
    into the constructor as `weights=`, `bias=`, `alpha=`). This is the real
    production API, just not one any passing test happens to exercise with
    a literal value — so the corpus here is hand-declared (like
    dpl_fuzz_study/fuzz_inputs.py's DeepProbLog corpus) rather than
    AST-discovered, and every entry is grounded in `lnn/_exceptions.py`
    (read directly from source before writing this) or in the library's
    OWN three-valued truth table, lifted verbatim from its passing tests
    `tests/reasoning/logic/propositional/test_tri_and_1.py` and
    `test_tri_or_1.py`.

Grounded oracle (read directly from lnn/_exceptions.py before writing this):
  AssertWeights: checks `weights` is a tuple of length == arity. Does NOT
    check the individual values are non-negative, finite, or in any range.
  AssertBias: checks `bias` is a float. Does NOT check its value.
  AssertAlphaNodeValue: alpha must satisfy 0.5 < alpha <= 1 (raises
    ValueError otherwise) -- this one IS enforced at construction.
  AssertAlphaNeuronArityValue: alpha must be >= arity/(arity+1) -- also
    enforced at construction.
  => weights/bias are DOC-INVALID-but-silently-accepted candidates (type
     checked, value unchecked); alpha is a DOC-INVALID-correctly-rejected
     control case, confirmed live against the real exceptions above.

  Semantic oracle: `lnn.And`/`lnn.Or` with the default weights=(1,1) exactly
  reproduce the crisp three-valued truth table in test_tri_and_1.py /
  test_tri_or_1.py (e.g. And(FALSE, TRUE) == FALSE). This is not an invented
  property -- it is the library's own oracle, already asserted in its own
  passing test file. This script asks: does that same truth table still
  hold once weights/bias take a value AssertWeights/AssertBias silently
  accepted but never validated?

Usage:
    python fuzz_auto_lnn.py --list      # print the corpus, run nothing
    python fuzz_auto_lnn.py             # gate-check, then run
    python fuzz_auto_lnn.py --force     # skip the gate check (debug only)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
GATE_FILE = RESULTS / "lnn_tests.json"

NAN, INF = float("nan"), float("inf")

# ---------------------------------------------------------------------------
# Gate: refuse to run unless LNN's own suite actually passed something
# ---------------------------------------------------------------------------
def check_gate(force: bool) -> dict:
    if force:
        return {"gate": "skipped", "reason": "--force"}
    if not GATE_FILE.exists():
        sys.exit(f"REFUSING TO RUN: {GATE_FILE} missing. Run the baseline test "
                  f"suite first (see results/lnn_tests.json).")
    data = json.loads(GATE_FILE.read_text())
    status = data.get("status")
    passed = data.get("junit", {}).get("counts", {}).get("passed", 0)
    if status not in ("ok", "tests_failed") or passed == 0:
        sys.exit(f"REFUSING TO RUN: baseline shows status={status!r}, passed={passed}.")
    print(f"Gate OK: baseline passed={passed} — proceeding.")
    return {"gate": "passed", "baseline_passed": passed, "baseline_status": status}


# ---------------------------------------------------------------------------
# Ground truth, lifted verbatim from the library's own passing tests.
# Fact bounds: TRUE=(1,1) FALSE=(0,0) UNKNOWN=(0,1)  (lnn/constants.py)
# ---------------------------------------------------------------------------
T, F, U = (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)

TRUTH_TABLES = {
    # source: tests/reasoning/logic/propositional/test_tri_and_1.py::test_upward
    "And": [
        (T, T, T), (F, T, F), (F, F, F), (U, T, U), (U, U, U), (U, F, F),
    ],
    # source: tests/reasoning/logic/propositional/test_tri_or_1.py::test_upward
    "Or": [
        (T, T, T), (F, T, T), (F, F, F), (U, T, T), (U, U, U), (U, F, U),
    ],
}


def mutate_weights(arity: int):
    """(op, weights tuple). 'seed' is the library's own default -- included
    as a control so every row of the corpus has a known-good comparison."""
    return [
        ("seed_default", (1.0,) * arity),
        ("negative", (-1.0,) + (1.0,) * (arity - 1)),
        ("all_negative", (-1.0,) * arity),
        ("zero", (0.0,) * arity),
        ("nan", (NAN,) + (1.0,) * (arity - 1)),
        ("pos_inf", (INF,) + (1.0,) * (arity - 1)),
        ("neg_inf", (-INF,) + (1.0,) * (arity - 1)),
        ("huge", (1e6,) + (1.0,) * (arity - 1)),
    ]


def mutate_bias():
    return [
        ("seed_default", 1.0),
        ("negative", -3.0),
        ("zero", 0.0),
        ("nan", NAN),
        ("pos_inf", INF),
        ("huge", 1e6),
    ]


ALPHA_CASES = [
    # (op, alpha, expected) -- expected drawn directly from AssertAlphaNodeValue
    # (0.5 < alpha <= 1) and AssertAlphaNeuronArityValue (alpha >= n/(n+1)).
    # For arity=2, n/(n+1) = 0.667, which is stricter than 0.5 here, so it is
    # the binding constraint.
    ("seed_default", None, "DOC-VALID"),         # library default (erf-based), always valid
    ("valid_high", 0.9, "DOC-VALID"),
    ("boundary_at_0.5", 0.5, "DOC-INVALID"),      # excluded endpoint: 0.5 < alpha required
    ("below_range", 0.4, "DOC-INVALID"),
    ("above_range", 1.1, "DOC-INVALID"),
    ("below_arity_bound", 0.6, "DOC-INVALID"),    # valid vs (.5,1] but < 2/3 arity bound
]


def run_weight_bias_case(conn_name: str, kind: str, op: str, value, tag: str) -> dict:
    """Construct the REAL And/Or with a mutated weight or bias, then check
    every row of the library's OWN truth table for that connective still
    holds. `kind` is 'weights' or 'bias'."""
    from lnn import Proposition, And, Or, Model  # noqa: local import, real package

    out = {"connective": conn_name, "kind": kind, "op": op, "value": repr(value)}
    conn_cls = {"And": And, "Or": Or}[conn_name]
    activation = {kind: value}

    rows_ok = []
    error = None
    try:
        for i, (a_bound, b_bound, expected) in enumerate(TRUTH_TABLES[conn_name]):
            A = Proposition(f"A_{tag}_{i}")
            B = Proposition(f"B_{tag}_{i}")
            AB = conn_cls(A, B, activation=activation)
            model = Model()
            model.add_knowledge(AB)
            model.add_data({A: a_bound, B: b_bound})
            AB.upward()
            got = AB.get_data()
            L, U_ = float(got[0]), float(got[1])
            finite = math.isfinite(L) and math.isfinite(U_)
            matches = finite and abs(L - expected[0]) < 1e-6 and abs(U_ - expected[1]) < 1e-6
            rows_ok.append({
                "a": a_bound, "b": b_bound, "expected": expected,
                "got": [L, U_], "finite": finite, "matches": matches,
            })
        accepted = True
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False

    out["accepted_at_construction"] = accepted
    if error:
        out["error"] = error

    if op == "seed_default":
        # control row: must reproduce the table exactly, or the harness itself is broken
        out["verdict"] = "ok" if accepted and all(r["matches"] for r in rows_ok) else "HARNESS BROKEN"
        out["ok"] = out["verdict"] == "ok"
    elif not accepted:
        # AssertWeights/AssertBias only check type+length/type, so a
        # same-type numeric mutation should NOT be rejected at construction.
        out["verdict"] = "unexpectedly rejected at construction (re-check _exceptions.py)"
        out["ok"] = True  # rejecting is not itself a finding either way
    else:
        n_bad = sum(1 for r in rows_ok if not r["matches"])
        out["rows"] = rows_ok
        if n_bad == 0:
            out["verdict"] = "ok (truth table still holds despite unchecked weight/bias)"
            out["ok"] = True
        else:
            out["verdict"] = (f"SILENTLY ACCEPTED -- {n_bad}/{len(rows_ok)} truth-table rows "
                               f"wrong under this {kind} value")
            out["ok"] = False
    return out


def run_alpha_case(conn_name: str, op: str, alpha, expected_validity: str, tag: str) -> dict:
    from lnn import Proposition, And, Or, Model  # noqa

    out = {"connective": conn_name, "kind": "alpha", "op": op, "value": alpha,
           "validity": expected_validity}
    conn_cls = {"And": And, "Or": Or}[conn_name]
    activation = {} if alpha is None else {"alpha": alpha}
    A = Proposition(f"AA_{tag}")
    B = Proposition(f"BB_{tag}")
    try:
        AB = conn_cls(A, B, activation=activation)
        accepted = True
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        accepted = False

    if expected_validity == "DOC-VALID":
        out["verdict"] = "ok" if accepted else "FALSE REJECT"
        out["ok"] = accepted
    else:  # DOC-INVALID -- _exceptions.py says this must raise
        out["verdict"] = "REJECTED (correct)" if not accepted else "SILENTLY ACCEPTED"
        out["ok"] = not accepted
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gate = check_gate(args.force)

    corpus_preview = []
    for conn in ("And", "Or"):
        for op, w in mutate_weights(2):
            corpus_preview.append(f"[{conn}] weights {op:<14} {w}")
        for op, b in mutate_bias():
            corpus_preview.append(f"[{conn}] bias    {op:<14} {b}")
        for op, a, exp in ALPHA_CASES:
            corpus_preview.append(f"[{conn}] alpha   {op:<18} {a}  ({exp})")

    if args.list:
        print(f"{len(corpus_preview)} corpus entries\n")
        for line in corpus_preview:
            print(" ", line)
        return 0

    t0 = time.time()
    rows = []
    tag = 0
    for conn in ("And", "Or"):
        for op, w in mutate_weights(2):
            tag += 1
            rows.append(run_weight_bias_case(conn, "weights", op, w, str(tag)))
        for op, b in mutate_bias():
            tag += 1
            rows.append(run_weight_bias_case(conn, "bias", op, b, str(tag)))
        for op, a, exp in ALPHA_CASES:
            tag += 1
            rows.append(run_alpha_case(conn, op, a, exp, str(tag)))

    n_violations = sum(1 for r in rows if not r["ok"])
    payload = {
        "library": "lnn",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 2),
        "gate": gate,
        "method": (
            "Hand-declared corpus (AST discovery of LNN's own passing tests found no "
            "literal weight/bias/alpha call sites) targeting the real, documented "
            "`activation` dict parameter of lnn.And/lnn.Or, grounded in lnn/_exceptions.py "
            "(AssertWeights, AssertBias, AssertAlphaNodeValue, AssertAlphaNeuronArityValue) "
            "and in the library's own crisp three-valued truth table "
            "(tests/reasoning/logic/propositional/test_tri_and_1.py, test_tri_or_1.py)."
        ),
        "n_cases": len(rows),
        "n_violations": n_violations,
        "results": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "fuzz_results_lnn_auto.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    print(f"\n{len(rows)} cases — {n_violations} oracle violations\n")
    for r in rows:
        mark = "!" if not r["ok"] else " "
        conn = r.get("connective", "")
        kind = r.get("kind", "")
        op = r.get("op", "")
        print(f"{mark}[{conn:<3}] {kind:<8} {op:<18} {r['verdict']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
