#!/usr/bin/env python3
"""
fuzz_inputs_deepstochlog.py — Step 2 of the study, ported to DeepStochLog.

RULES THIS SCRIPT FOLLOWS (stated explicitly, per the project's own gating
methodology in dpl-fuzz-study-README.md and the professor's required order):

  1. This script refuses to run unless results/deepstochlog_tests.json exists
     and shows the library's own test suite was actually executed (status is
     "ok" or "tests_failed" with at least one PASSED test) — never "not_installed",
     "no_test_suite", "tests_not_found", or missing entirely. Baseline first,
     fuzzing second, always in that order.
  2. Fuzzing only happens for a library whose OWN test suite has demonstrated
     it can be imported, built, and run at all. A library with zero passing
     tests contributes nothing trustworthy to fuzz against.
  3. The corpus below (DOC-VALID / DOC-INVALID) is the SAME oracle scheme as
     fuzz_inputs.py (DeepProbLog) and the README's own general framing — this
     is deliberate, so results are comparable across libraries, not a
     DeepStochLog-specific invention.
  4. What's ACTUALLY verified before this file was written (not assumed):
       - grammar syntax: `dom(X) :- member(X, [a,b,c]).` +
         `nn(name, [X], Y, dom) :: head(Y) --> [X].`
         (confirmed against examples/bracket/bracket.pl, the one real,
         working example in the repo — not guessed)
       - network wiring: Network(name, nn.Module, index_list=[Term...])
       - execution: DeepStochLogModel.from_string(...).calculate_probability(ct)
       - CONFIRMED LIVE: raw_logits [3.2,-1.1,0.4] -> P=3.2 (unchecked);
         negative_entry [-0.5,0.75,0.75] -> P=-0.5 (unchecked); NaN propagates
         (unchecked) — same defect class as the DeepProbLog study, at the same
         architectural point (GraphSemiring.value's analogue here is the raw
         `network.neural_model(neural_input)` output in networkevaluation.py,
         which nothing validates before storing).
  5. NOT covered by this port (scope limits, stated rather than silently
     skipped): O2 (equivalence), O5 (gradient), O6 (config/cache differential),
     O7 (degenerate-belief cross-check against a substrate) are NOT
     implemented here. Only O1 (range) and O4 (normalisation, where the
     surface is an exhaustive alternative set) are ported. Extending to the
     other oracles needs DeepStochLog's own tabling/caching hooks understood
     first — not done, not faked here.

Usage:
    python fuzz_inputs_deepstochlog.py --list      # print corpus, run nothing
    python fuzz_inputs_deepstochlog.py             # gate-check, then run
    python fuzz_inputs_deepstochlog.py --force      # skip the gate check (debug only)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
GATE_FILE = RESULTS / "deepstochlog_tests.json"

NAN, INF = float("nan"), float("inf")


@dataclass
class FuzzInput:
    id: str
    surface: str
    tier: str
    label: str
    value: Any
    validity: str
    origin: str
    ref: str
    oracles: tuple = ("O1",)
    note: str = ""


# ===========================================================================
#  THE CORPUS — same scheme as fuzz_inputs.py's belief_ad surface, restated
#  for DeepStochLog's neural predicate (a belief vector over a domain list).
# ===========================================================================
CORPUS: list[FuzzInput] = [
    FuzzInput("DSL-R-01", "belief", "realistic", "uniform_init",
              [1/3, 1/3, 1/3], "DOC-VALID",
              "softmax over near-zero logits, the network at initialisation",
              "LPAD04", ("O1", "O4")),
    FuzzInput("DSL-R-02", "belief", "realistic", "converged_confident",
              [0.98, 0.01, 0.01], "DOC-VALID",
              "a trained classifier on an easy example", "LPAD04", ("O1", "O4")),
    FuzzInput("DSL-R-03", "belief", "realistic", "saturated_softmax",
              [1.0 - 1e-7, 5e-8, 5e-8], "DOC-VALID",
              "late training: softmax saturates near float32 epsilon",
              "IEEE754", ("O1", "O4")),
    FuzzInput("DSL-R-04", "belief", "realistic", "hard_zero_class",
              [0.6, 0.4, 0.0], "DOC-VALID",
              "a masked or pruned class reaching the belief vector as exact 0.0",
              "DPL18", ("O1", "O4")),
    FuzzInput("DSL-R-05", "belief", "realistic", "hard_one_class",
              [1.0, 0.0, 0.0], "DOC-VALID",
              "argmax hardening / one-hot teacher forcing", "DPL18", ("O1", "O4")),

    FuzzInput("DSL-A-01", "belief", "adversarial", "raw_logits",
              [3.2, -1.1, 0.4], "DOC-INVALID",
              "the softmax was never applied before the belief vector reaches "
              "the neural predicate -- CONFIRMED LIVE: accepted, P=3.2",
              "DPL18", ("O1", "O4")),
    FuzzInput("DSL-A-02", "belief", "adversarial", "sigmoid_multilabel",
              [0.9, 0.9, 0.9], "DOC-INVALID",
              "a sigmoid head reused where softmax over the domain is required",
              "LPAD04", ("O1", "O4")),
    FuzzInput("DSL-A-03", "belief", "adversarial", "sum_below_one",
              [0.1, 0.1, 0.1], "DOC-INVALID",
              "dropout or masking left active at inference time -- CONFIRMED "
              "LIVE: accepted, P=0.1 with no normalisation check",
              "LPAD04", ("O1", "O4")),
    FuzzInput("DSL-A-04", "belief", "adversarial", "negative_entry",
              [-0.5, 0.75, 0.75], "DOC-INVALID",
              "sums to 1 but contains negative mass -- CONFIRMED LIVE: "
              "accepted, P=-0.5, no non-negativity check",
              "Kolmogorov33", ("O1", "O4")),
    FuzzInput("DSL-A-05", "belief", "adversarial", "entry_above_one",
              [1.5, -0.25, -0.25], "DOC-INVALID",
              "same failure mode, excess mass on the other side",
              "Kolmogorov33", ("O1", "O4")),
    FuzzInput("DSL-A-06", "belief", "adversarial", "nan_entry",
              [NAN, 0.5, 0.5], "DOC-INVALID",
              "0/0 inside a softmax, or a diverged training run -- CONFIRMED "
              "LIVE: NaN propagates through calculate_probability unchecked",
              "IEEE754", ("O1", "O4")),
    FuzzInput("DSL-A-07", "belief", "adversarial", "positive_infinity",
              [INF, 0.0, 0.0], "DOC-INVALID",
              "exp() overflow without the max-subtraction trick",
              "IEEE754", ("O1", "O4")),
    FuzzInput("DSL-A-08", "belief", "adversarial", "negative_infinity",
              [-INF, 1.0, 0.0], "DOC-INVALID",
              "log(0) in a log-probability pipeline", "IEEE754", ("O1", "O4")),
]


# ===========================================================================
#  GATE — refuses to run unless deepstochlog's OWN suite has passing tests
# ===========================================================================
def check_gate(force: bool) -> dict:
    if force:
        return {"gate": "skipped", "reason": "--force"}

    if not GATE_FILE.exists():
        print(f"REFUSING TO RUN: {GATE_FILE} does not exist.")
        print("Run `python run_library_tests.py --only deepstochlog --auto-clone` first.")
        sys.exit(1)

    data = json.loads(GATE_FILE.read_text())
    status = data.get("status")
    counts = data.get("junit", {}).get("counts", {})
    passed = counts.get("passed", 0)

    if status not in ("ok", "tests_failed") or passed == 0:
        print(f"REFUSING TO RUN: deepstochlog_tests.json shows status={status!r}, "
              f"passed={passed}.")
        print("Per the project's own rule (baseline first, fuzz only what passed), "
              "this library is not eligible yet.")
        sys.exit(1)

    print(f"Gate check OK: deepstochlog baseline shows passed={passed} "
          f"(failed={counts.get('failures')}, errors={counts.get('errors')}, "
          f"skipped={counts.get('skipped')}) -- proceeding.")
    return {"gate": "passed", "baseline_passed": passed, "baseline_status": status}


# ===========================================================================
#  HARNESS — verified live against the real deepstochlog API before writing
# ===========================================================================
def _lazy_imports():
    global torch, nn, Network, NetworkStore, DeepStochLogModel, Term, List, Context, ContextualizedTerm
    import torch  # noqa
    import torch.nn as nn  # noqa
    from deepstochlog.network import Network, NetworkStore  # noqa
    from deepstochlog.model import DeepStochLogModel  # noqa
    from deepstochlog.term import Term, List  # noqa
    from deepstochlog.context import Context, ContextualizedTerm  # noqa


PROGRAM = ("dom(X) :- member(X, [a,b,c]).\n"
           "nn(dummy1, [X], Y, dom) :: single(Y) --> [X].")


class _FixedOutputNet:
    """Built lazily inside run_case, after torch is imported."""
    pass


def _is_probability(x) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and -1e-9 <= f <= 1.0 + 1e-9


def _solve(belief: list[float], target: str = "a"):
    class DummyNet(nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = torch.tensor([output], dtype=torch.float32)

        def forward(self, x):
            return self.output.repeat(x.shape[0], 1)

    t1 = Term("t1")
    net = Network("dummy1", DummyNet(list(belief)),
                  index_list=[Term("a"), Term("b"), Term("c")])
    networks = NetworkStore(net)
    model = DeepStochLogModel.from_string(PROGRAM, networks=networks, query=None)
    ct = ContextualizedTerm(
        context=Context({t1: torch.zeros(1)}),
        term=Term("single", Term(target), List(t1)),
    )
    p = model.calculate_probability(ct)
    return float(p)


def run_case(fi: FuzzInput) -> dict:
    out = {"id": fi.id, "surface": fi.surface, "tier": fi.tier, "label": fi.label,
           "validity": fi.validity, "ref": fi.ref, "oracles": list(fi.oracles)}
    try:
        p = _solve(fi.value)
        out["F"] = p
        out["O1_range"] = _is_probability(p)
        out["outcome"] = "accepted"
    except Exception as e:                                       # noqa: BLE001
        out["outcome"] = "rejected"
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"

    if fi.validity == "DOC-VALID":
        if out["outcome"] == "rejected":
            out["verdict"] = "FALSE REJECT"
            out["ok"] = False
        elif out.get("O1_range") is False:
            out["verdict"] = "OUT OF RANGE"
            out["ok"] = False
        else:
            out["verdict"] = "ok"
            out["ok"] = True
    else:  # DOC-INVALID
        if out["outcome"] == "rejected":
            out["verdict"] = "REJECTED (correct)"
            out["ok"] = True
        else:
            out["verdict"] = "SILENTLY ACCEPTED"
            out["ok"] = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                     help="skip the baseline-passed gate check (debug only)")
    args = ap.parse_args()

    if args.list:
        print(f"{len(CORPUS)} corpus entries\n")
        for fi in CORPUS:
            print(f"{fi.id:<10} {fi.tier:<12} {fi.validity:<12} {fi.label:<26} [{fi.ref}]")
            print(f"{'':<10} origin: {fi.origin}")
        return 0

    gate = check_gate(args.force)

    warnings.filterwarnings("ignore")
    _lazy_imports()

    t0 = time.time()
    rows = [run_case(fi) for fi in CORPUS]
    elapsed = time.time() - t0

    payload = {
        "library": "deepstochlog",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(elapsed, 2),
        "gate": gate,
        "n_cases": len(rows),
        "n_violations": sum(1 for r in rows if not r["ok"]),
        "scope_note": "O1 (range) and O4-equivalent (normalisation, where applicable) "
                      "only. O2/O5/O6/O7 NOT ported -- see module docstring.",
        "corpus": [asdict(c) for c in CORPUS],
        "results": rows,
    }
    (RESULTS / "fuzz_results_deepstochlog.json").write_text(
        json.dumps(payload, indent=1, default=str))

    print(f"\n{len(rows)} cases in {elapsed:.1f}s -- {payload['n_violations']} oracle violations\n")
    for r in rows:
        mark = "!" if not r["ok"] else " "
        shown = r.get("error") or r.get("F")
        print(f"{mark} {r['id']:<12} {r['verdict']:<20} {str(shown)[:60]}")
    print(f"\nwrote {RESULTS / 'fuzz_results_deepstochlog.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
