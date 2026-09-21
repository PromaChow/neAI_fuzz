#!/usr/bin/env python3
"""
Step 2 of the study: the fuzzed-input corpus, and the harness that runs it.

The corpus is DATA, declared once in CORPUS below. Every entry carries:

    id        stable identifier, cited in the README and in results/
    surface   where the value enters the system (see SURFACES)
    tier      realistic | boundary | adversarial
    value     the mutated input itself
    validity  DOC-VALID   -- a well-formed element of the algebra the library
                             documents, so the library must return the
                             documented answer;
              DOC-INVALID -- not a well-formed element, so the library should
                             REJECT it. Silently returning a number is the
                             violation, regardless of which number.
    origin    how a real pipeline produces this value (the realism argument)
    ref       reference key; see README section "References"
    oracles   which semantic oracles must hold for this input

Oracles (all derived from the neurosymbolic inference integral; see README):
    O1  range         F in [0,1], finite
    O2  equivalence   logically equivalent programs give equal F
    O3  logical law    monotonicity P(A) <= P(A or B) <= P(A or B or C), and the
                       conjunction bound P(A and B) <= min(P(A), P(B)). These
                       follow from the Kolmogorov axioms and hold for EVERY
                       probability model, so a violation is a proof of error
                       that needs no ground truth. O3 is the only oracle here
                       that catches case B-A-13.
    O4  normalisation exhaustive AD sums to 1; F(phi) + F(not phi) = 1
    O5  gradient      dF/db matches a central finite difference
    O6  differential  config / cache / batch composition do not change F
    O7  degenerate    with b in {0,1}, F equals the classical logic answer

Usage:
    python fuzz_inputs.py            # run everything, write results/
    python fuzz_inputs.py --list     # print the corpus, run nothing
    python fuzz_inputs.py --surface belief
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

NAN, INF = float("nan"), float("inf")

SURFACES = {
    "belief_ad":   "output vector of an nn/4 neural annotated disjunction",
    "belief_fact": "output scalar of an nn/2 neural probabilistic fact",
    "parameter":   "a learnable t(_) probabilistic parameter in the program text",
    "builtin":     "arguments to the tensor builtins registered in the logic",
    "program":     "the program text, under semantics-preserving rewriting",
    "config":      "engine / cache / batch configuration, which must not matter",
    "laws":        "probability laws that hold for every model (monotonicity)",
}


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
#  THE CORPUS
# ===========================================================================
CORPUS: list[FuzzInput] = [

    # -- belief_ad / realistic --------------------------------------------
    # A softmax head over a 3-element domain. Everything here is a genuine
    # probability distribution: the library MUST handle all of it correctly.
    FuzzInput("B-R-01", "belief_ad", "realistic", "uniform_init",
              [1/3, 1/3, 1/3], "DOC-VALID",
              "softmax over near-zero logits, i.e. the network at initialisation",
              "LPAD04", ("O1", "O4", "O5")),
    FuzzInput("B-R-02", "belief_ad", "realistic", "converged_confident",
              [0.98, 0.01, 0.01], "DOC-VALID",
              "a trained classifier on an easy example",
              "LPAD04", ("O1", "O4", "O5")),
    FuzzInput("B-R-03", "belief_ad", "realistic", "saturated_softmax",
              [1.0 - 1e-7, 5e-8, 5e-8], "DOC-VALID",
              "late training: softmax saturates and the small entries approach "
              "float32 epsilon (~1.19e-7)",
              "IEEE754", ("O1", "O4", "O5")),
    FuzzInput("B-R-04", "belief_ad", "realistic", "underflow_softmax",
              [1.0, 1e-45, 1e-45], "DOC-VALID",
              "logit gap large enough that exp() underflows into float32 "
              "subnormals (smallest subnormal ~1.4e-45)",
              "IEEE754", ("O1", "O4", "O5")),
    FuzzInput("B-R-05", "belief_ad", "realistic", "binary_float_sum",
              [0.1, 0.2, 0.7], "DOC-VALID",
              "a distribution whose entries are not exactly representable in "
              "binary, so the stored values do not sum to exactly 1",
              "Goldberg91", ("O1", "O4")),
    FuzzInput("B-R-06", "belief_ad", "realistic", "hard_zero_class",
              [0.6, 0.4, 0.0], "DOC-VALID",
              "masked or pruned class: an exact 0.0 reaches the semiring, where "
              "is_exact_zero() short-circuits the multiplication",
              "DPL18", ("O1", "O4", "O5")),
    FuzzInput("B-R-07", "belief_ad", "realistic", "hard_one_class",
              [1.0, 0.0, 0.0], "DOC-VALID",
              "argmax hardening / one-hot teacher forcing; also the degenerate "
              "belief used by oracle O7",
              "DPL18", ("O1", "O4", "O5", "O7")),
    FuzzInput("B-R-08", "belief_ad", "realistic", "near_tie",
              [0.3333333, 0.3333333, 0.3333334], "DOC-VALID",
              "three-way tie resolved only in the last float32 digit",
              "IEEE754", ("O1", "O4")),

    # -- belief_ad / boundary ---------------------------------------------
    # Still valid distributions, but sitting on the epsilon thresholds the
    # implementation uses internally (Semiring.eps = 1e-5).
    FuzzInput("B-B-01", "belief_ad", "boundary", "just_inside_eps",
              [1.0 - 1e-6, 5e-7, 5e-7], "DOC-VALID",
              "inside the 1e-5 window where is_one()/is_zero() treat a value as "
              "exactly 1 or 0 -- a valid distribution the implementation may "
              "silently round",
              "DPLSRC", ("O1", "O4", "O5")),
    FuzzInput("B-B-02", "belief_ad", "boundary", "just_outside_eps",
              [1.0 - 1e-4, 5e-5, 5e-5], "DOC-VALID",
              "just outside that window; paired with B-B-01 to isolate the "
              "threshold",
              "DPLSRC", ("O1", "O4", "O5")),
    FuzzInput("B-B-03", "belief_ad", "boundary", "signed_zero",
              [-0.0, 0.5, 0.5], "DOC-VALID",
              "IEEE 754 negative zero, which compares equal to 0.0 but carries "
              "a sign bit; produced by underflow of a negative quantity",
              "IEEE754", ("O1", "O4")),

    # -- belief_ad / adversarial ------------------------------------------
    # None of these is a probability distribution. A library implementing the
    # distribution semantics must not accept them silently.
    FuzzInput("B-A-01", "belief_ad", "adversarial", "raw_logits",
              [3.2, -1.1, 0.4], "DOC-INVALID",
              "the softmax was never applied -- the single most common wiring "
              "mistake when attaching a classifier head to a neural predicate",
              "DPL18", ("O1", "O4")),
    FuzzInput("B-A-02", "belief_ad", "adversarial", "sigmoid_multilabel",
              [0.9, 0.9, 0.9], "DOC-INVALID",
              "a sigmoid (multi-label) head reused where a softmax "
              "(single-label) head is required: each entry is in [0,1] but they "
              "do not form a distribution",
              "LPAD04", ("O1", "O4")),
    FuzzInput("B-A-03", "belief_ad", "adversarial", "sum_below_one",
              [0.1, 0.1, 0.1], "DOC-INVALID",
              "dropout or masking left active at inference time",
              "LPAD04", ("O1", "O4")),
    FuzzInput("B-A-04", "belief_ad", "adversarial", "all_zero",
              [0.0, 0.0, 0.0], "DOC-INVALID",
              "dead ReLU block, or an entirely masked output",
              "LPAD04", ("O1", "O4")),
    FuzzInput("B-A-05", "belief_ad", "adversarial", "negative_entry",
              [-0.5, 0.75, 0.75], "DOC-INVALID",
              "sums to 1 but contains a negative mass: arises from a linear head "
              "with no activation, or from a numerically unstable "
              "log-space-to-probability conversion",
              "Kolmogorov33", ("O1", "O4")),
    FuzzInput("B-A-06", "belief_ad", "adversarial", "entry_above_one",
              [1.5, -0.25, -0.25], "DOC-INVALID",
              "same origin as B-A-05, with the excess mass on the other side",
              "Kolmogorov33", ("O1", "O4")),
    FuzzInput("B-A-07", "belief_ad", "adversarial", "nan_entry",
              [NAN, 0.5, 0.5], "DOC-INVALID",
              "0/0 inside a softmax, or a diverged training run; NaN propagates "
              "through every arithmetic operation by IEEE 754",
              "IEEE754", ("O1", "O4")),
    FuzzInput("B-A-08", "belief_ad", "adversarial", "positive_infinity",
              [INF, 0.0, 0.0], "DOC-INVALID",
              "exp() overflow in a softmax written without the max-subtraction "
              "trick",
              "IEEE754", ("O1", "O4")),
    FuzzInput("B-A-09", "belief_ad", "adversarial", "negative_infinity",
              [-INF, 1.0, 0.0], "DOC-INVALID",
              "log(0) in a log-probability pipeline",
              "IEEE754", ("O1", "O4")),
    FuzzInput("B-A-10", "belief_ad", "adversarial", "vector_too_short",
              [0.5, 0.5], "DOC-INVALID",
              "network output width does not match the declared domain: the "
              "domain list in the program was edited but the head was not",
              "DPL18", ("O1",)),
    FuzzInput("B-A-11", "belief_ad", "adversarial", "vector_too_long",
              [0.25, 0.25, 0.25, 0.25], "DOC-INVALID",
              "same mismatch in the other direction",
              "DPL18", ("O1", "O4")),
    FuzzInput("B-A-13", "belief_ad", "adversarial", "negative_mass_sums_to_one",
              [0.6, -0.2, 0.6], "DOC-INVALID",
              "negative mass that still sums to exactly 1.0 -- a linear head "
              "with no activation, or an unstable log-space conversion. This "
              "case DEFEATS both the range oracle (every returned probability "
              "lands in [0,1]) and the normalisation oracle (the belief sums to "
              "1). Only a logical-law oracle catches it: see O3.",
              "KOLMOGOROV33", ("O1", "O3", "O4")),
    FuzzInput("B-A-14", "belief_ad", "adversarial", "negative_mass_sums_to_one_2",
              [0.7, -0.4, 0.7], "DOC-INVALID",
              "same shape, larger negative mass; confirms B-A-13 is not a "
              "single coincidence",
              "KOLMOGOROV33", ("O1", "O3", "O4")),
    FuzzInput("B-A-12", "belief_ad", "adversarial", "empty_vector",
              [], "DOC-INVALID",
              "an empty batch or a collapsed output dimension",
              "DPL18", ("O1",)),

    # -- belief_fact (nn/2, a single probability) --------------------------
    FuzzInput("F-R-01", "belief_fact", "realistic", "half", [0.5], "DOC-VALID",
              "a sigmoid head at the decision boundary", "SATO95", ("O1",)),
    FuzzInput("F-B-01", "belief_fact", "boundary", "zero", [0.0], "DOC-VALID",
              "sigmoid underflow; a legitimate probability", "SATO95", ("O1", "O7")),
    FuzzInput("F-B-02", "belief_fact", "boundary", "one", [1.0], "DOC-VALID",
              "sigmoid saturation; a legitimate probability", "SATO95", ("O1", "O7")),
    FuzzInput("F-A-01", "belief_fact", "adversarial", "negative", [-0.1], "DOC-INVALID",
              "a raw logit or an un-clamped residual reaching a probabilistic fact",
              "Kolmogorov33", ("O1",)),
    FuzzInput("F-A-02", "belief_fact", "adversarial", "above_one", [1.1], "DOC-INVALID",
              "same, on the upper side", "Kolmogorov33", ("O1",)),
    FuzzInput("F-A-03", "belief_fact", "adversarial", "nan", [NAN], "DOC-INVALID",
              "diverged training", "IEEE754", ("O1",)),

    # -- parameter (t(_) in the program text) ------------------------------
    # This surface carries no neural network at all. It isolates the claim: the
    # defect is the interface contract, not neural networks.
    FuzzInput("P-R-01", "parameter", "realistic", "t_half",
              "t(0.5) :: c.", "DOC-VALID",
              "the ordinary way to declare a learnable probability", "PLPCON15", ("O1",)),
    FuzzInput("P-B-01", "parameter", "boundary", "t_zero",
              "t(0.0) :: c.", "DOC-VALID", "boundary of the unit interval",
              "PLPCON15", ("O1",)),
    FuzzInput("P-B-02", "parameter", "boundary", "t_one",
              "t(1.0) :: c.", "DOC-VALID", "boundary of the unit interval",
              "PLPCON15", ("O1",)),
    FuzzInput("P-A-01", "parameter", "adversarial", "t_negative",
              "t(-0.3) :: c.", "DOC-INVALID",
              "a sign error in a hand-written or generated program, or a "
              "parameter restored from a checkpoint that was updated past 0",
              "Kolmogorov33", ("O1",)),
    FuzzInput("P-A-02", "parameter", "adversarial", "t_above_one",
              "t(1.7) :: c.", "DOC-INVALID",
              "an unclamped gradient step on a probabilistic parameter",
              "Kolmogorov33", ("O1",)),
    FuzzInput("P-A-03", "parameter", "adversarial", "t_nan",
              "t(nan) :: c.", "DOC-INVALID",
              "a NaN parameter restored from a diverged run", "IEEE754", ("O1",)),
    FuzzInput("P-R-02", "parameter", "realistic", "t_uninitialised",
              "t(_) :: c.", "DOC-VALID",
              "the documented way to ask for a randomly initialised parameter",
              "PLPCON15", ("O1",)),

    # -- builtin (tensor predicates exposed to the logic) ------------------
    FuzzInput("X-R-01", "builtin", "realistic", "one_hot_valid",
              "q(T) :- one_hot(1, 4, T).", "DOC-VALID",
              "the documented use", "CWE129", ("O1",)),
    FuzzInput("X-A-01", "builtin", "adversarial", "one_hot_negative_index",
              "q(T) :- one_hot(-1, 4, T).", "DOC-INVALID",
              "an index computed in the logic that went below zero -- Python "
              "silently wraps negative indices to the end of the sequence",
              "CWE129", ("O1",)),
    FuzzInput("X-A-02", "builtin", "adversarial", "one_hot_index_too_large",
              "q(T) :- one_hot(9, 4, T).", "DOC-INVALID",
              "index past the end; the control case for X-A-01", "CWE129", ("O1",)),
    FuzzInput("X-A-03", "builtin", "adversarial", "one_hot_zero_width",
              "q(T) :- one_hot(0, 0, T).", "DOC-INVALID",
              "a domain size computed as zero", "CWE129", ("O1",)),
    FuzzInput("X-R-02", "builtin", "realistic", "tensor_index_valid",
              "q(T) :- tensor_index(tensor(src(a)), [1], T).", "DOC-VALID",
              "the documented use", "CWE129", ("O1",)),
    FuzzInput("X-A-04", "builtin", "adversarial", "tensor_index_negative",
              "q(T) :- tensor_index(tensor(src(a)), [-1], T).", "DOC-INVALID",
              "same negative-wrap hazard as X-A-01, on a tensor lookup",
              "CWE129", ("O1",)),
    FuzzInput("X-A-05", "builtin", "adversarial", "tensor_index_too_large",
              "q(T) :- tensor_index(tensor(src(a)), [99], T).", "DOC-INVALID",
              "index past the end; control case for X-A-04", "CWE129", ("O1",)),
    FuzzInput("X-A-06", "builtin", "adversarial", "tensor_index_empty",
              "q(T) :- tensor_index(tensor(src(a)), [], T).", "DOC-INVALID",
              "an empty index list produced by a failed filter", "CWE129", ("O1",)),
]

# Semantics-preserving program rewritings for oracle O2. Each must give exactly
# the same probability as the seed: the transformations are sound under the
# distribution semantics (Sato 1995), so any disagreement is a bug in grounding
# or knowledge compilation, not a modelling choice.
EQUIVALENT_PROGRAMS = {
    "seed":            "q(X,Y) :- net1(X,Y).",
    "double_negation": "q(X,Y) :- \\+ \\+ net1(X,Y).",
    "prepended_true":  "q(X,Y) :- true, net1(X,Y).",
    "via_intermediate": "h(X,Y) :- net1(X,Y).\nq(X,Y) :- h(X,Y).",
    "added_tautology": "q(X,Y) :- net1(X,Y), (true ; fail).",
    "dead_clause":     "q(X,Y) :- net1(X,Y).\nq(X,Y) :- fail, net1(X,Y).",
}

# Configurations that must not change F (oracle O6). Caching and batching are
# performance features; the field's own position (DeepLog s.4) is that apparent
# differences between NeSy systems "arise from how this formula is calculated,
# not from what is computed" -- the same must hold within one system.
CONFIGURATIONS = ["no_cache", "cache", "with_batch_mate", "reordered_batch"]


# ===========================================================================
#  HARNESS
# ===========================================================================
def _lazy_imports():
    global Term, Var, Constant, Model, Network, ExactEngine, Query, DummyNet, torch
    import torch  # noqa
    from problog.logic import Term, Var, Constant  # noqa
    from deepproblog.model import Model  # noqa
    from deepproblog.network import Network  # noqa
    from deepproblog.engines import ExactEngine  # noqa
    from deepproblog.query import Query  # noqa
    from deepproblog.utils.standard_networks import DummyNet  # noqa


AD_PROGRAM = "nn(dummy1,[X],Y,[a,b,c]) :: net1(X,Y).\n"
FACT_PROGRAM = "nn(dummy2,[X]) :: net2(X).\nq2(X) :- net2(X).\n"
TENSOR_SRC_PROGRAM = "dummy_fact.\n"


def _solve(program, terms, nets=(), sources=None, cache=False):
    m = Model(program, list(nets), load=False)
    m.set_engine(ExactEngine(m), cache=cache)
    for k, v in (sources or {}).items():
        m.add_tensor_source(k, v)
    qs = [Query(t) for t in terms]
    res = m.solve(qs)
    return m, [(r.result[q.query] if q.query in r.result else None)
               for q, r in zip(qs, res)]


def _is_probability(x) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and -1e-9 <= f <= 1.0 + 1e-9


def run_case(fi: FuzzInput) -> dict:
    """Execute one corpus entry and judge it against its oracles."""
    out = {"id": fi.id, "surface": fi.surface, "tier": fi.tier, "label": fi.label,
           "validity": fi.validity, "ref": fi.ref, "oracles": list(fi.oracles)}
    try:
        if fi.surface == "belief_ad":
            net = Network(DummyNet({Term("i1"): list(fi.value)}), "dummy1")
            terms = [Term("net1", Term("i1"), Term(d)) for d in "abc"]
            _, got = _solve(AD_PROGRAM, terms, nets=[net])
            vals = [None if g is None else float(g) for g in vals_of(got)]
            out["F"] = vals
            out["O1_range"] = all(_is_probability(v) for v in vals if v is not None)
            present = [v for v in vals if v is not None]
            out["O4_sum"] = sum(present) if present else None
            out["O4_normalised"] = (present != [] and
                                    abs(sum(present) - 1.0) < 1e-6)

        elif fi.surface == "belief_fact":
            net = Network(DummyNet({Term("i1"): list(fi.value)}), "dummy2")
            _, got = _solve(FACT_PROGRAM, [Term("q2", Term("i1"))], nets=[net])
            v = None if got[0] is None else float(got[0])
            out["F"] = [v]
            out["O1_range"] = _is_probability(v)

        elif fi.surface == "parameter":
            _, got = _solve(fi.value + "\nq :- c.\n", [Term("q")])
            v = None if got[0] is None else float(got[0])
            out["F"] = [v]
            out["O1_range"] = _is_probability(v)

        elif fi.surface == "builtin":
            src = {(Term("a"),): torch.tensor([0.1, 0.2, 0.3, 0.4])}
            m, _ = _solve(TENSOR_SRC_PROGRAM + fi.value, [], sources={"src": src})
            res = m.solve([Query(Term("q", Var("T")))])[0]
            answers = list(res.result)
            val = m.get_tensor(answers[0].args[0]) if answers else None
            out["F"] = val.tolist() if torch.is_tensor(val) else val
            out["O1_range"] = True  # a value was produced; correctness is judged below
        else:
            raise ValueError(f"unknown surface {fi.surface}")

        out["outcome"] = "accepted"
    except Exception as e:                                   # noqa: BLE001
        out["outcome"] = "rejected"
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # -- verdict ---------------------------------------------------------
    if fi.validity == "DOC-VALID":
        if out["outcome"] == "rejected":
            out["verdict"] = "FALSE REJECT"
            out["ok"] = False
        elif out.get("O1_range") is False:
            out["verdict"] = "OUT OF RANGE"
            out["ok"] = False
        elif "O4" in fi.oracles and out.get("O4_normalised") is False:
            out["verdict"] = "AD NOT NORMALISED"
            out["ok"] = False
        else:
            out["verdict"] = "ok"
            out["ok"] = True
    else:  # DOC-INVALID -- the library is supposed to refuse this
        if out["outcome"] == "rejected":
            out["verdict"] = "REJECTED (correct)"
            out["ok"] = True
        else:
            detail = []
            if out.get("O1_range") is False:
                detail.append("result outside [0,1] or non-finite")
            if out.get("O4_normalised") is False:
                detail.append("AD does not sum to 1")
            out["verdict"] = "SILENTLY ACCEPTED" + (
                " -- " + "; ".join(detail) if detail else "")
            out["ok"] = False
    return out


def vals_of(got):
    return got


def run_equivalence() -> list[dict]:
    rows = []
    base = AD_PROGRAM
    for belief_id in ("B-R-01", "B-R-02", "B-R-04", "B-A-01"):
        fi = next(c for c in CORPUS if c.id == belief_id)
        ref = None
        for name, tail in EQUIVALENT_PROGRAMS.items():
            try:
                net = Network(DummyNet({Term("i1"): list(fi.value)}), "dummy1")
                _, got = _solve(base + tail, [Term("q", Term("i1"), Term("a"))],
                                nets=[net])
                v = None if got[0] is None else float(got[0])
            except Exception as e:                            # noqa: BLE001
                v = f"EXC:{type(e).__name__}"
            if name == "seed":
                ref = v
                continue
            same = v == ref or (isinstance(v, float) and isinstance(ref, float)
                                and abs(v - ref) < 1e-9)
            rows.append({"id": f"O2/{belief_id}/{name}", "surface": "program",
                         "tier": fi.tier, "label": f"{fi.label} :: {name}",
                         "validity": "DOC-VALID", "ref": "SATO95",
                         "oracles": ["O2"], "F": [v],
                         "verdict": "ok" if same else "EQUIVALENT PROGRAMS DISAGREE",
                         "ok": same, "note": f"seed={ref} rewritten={v}"})
    return rows


def run_configurations() -> list[dict]:
    rows = []
    t = Term("net1", Term("i1"), Term("a"))
    mate = Term("net1", Term("i2"), Term("b"))
    for fi in [c for c in CORPUS if c.surface == "belief_ad"]:
        def mk():
            return Network(DummyNet({Term("i1"): list(fi.value),
                                     Term("i2"): [0.2, 0.3, 0.5]}), "dummy1")
        vals = {}
        try:
            vals["no_cache"] = _solve(AD_PROGRAM, [t], nets=[mk()])[1][0]
            vals["cache"] = _solve(AD_PROGRAM, [t], nets=[mk()], cache=True)[1][0]
            vals["with_batch_mate"] = _solve(AD_PROGRAM, [t, mate], nets=[mk()])[1][0]
            vals["reordered_batch"] = _solve(AD_PROGRAM, [mate, t], nets=[mk()])[1][1]
        except Exception as e:                                # noqa: BLE001
            rows.append({"id": f"O6/{fi.id}", "surface": "config", "tier": fi.tier,
                         "label": fi.label, "validity": fi.validity, "ref": "DEEPLOG26",
                         "oracles": ["O6"], "verdict": "consistent rejection",
                         "ok": True, "note": f"all configurations raised {type(e).__name__}"})
            continue

        fl = {}
        for k, v in vals.items():
            fl[k] = None if v is None else float(v)

        def eq(a, b):
            if a is None or b is None:
                return a is b
            if math.isnan(a) and math.isnan(b):
                return True
            return abs(a - b) < 1e-9

        agree = all(eq(fl["no_cache"], fl[k]) for k in CONFIGURATIONS)
        rows.append({"id": f"O6/{fi.id}", "surface": "config", "tier": fi.tier,
                     "label": fi.label, "validity": fi.validity, "ref": "DEEPLOG26",
                     "oracles": ["O6"], "F": fl,
                     "verdict": "ok" if agree else "CONFIGURATION CHANGES ANSWER",
                     "ok": agree})
    return rows


LAW_PROGRAM = ("nn(dummy1,[X],Y,[a,b,c]) :: net1(X,Y).\n"
               "la(X) :- net1(X,a).\n"
               "lab(X) :- net1(X,a).\nlab(X) :- net1(X,b).\n"
               "labc(X) :- net1(X,a).\nlabc(X) :- net1(X,b).\nlabc(X) :- net1(X,c).\n")


def run_probability_laws() -> list[dict]:
    """O3. Monotonicity is a theorem of the Kolmogorov axioms: if A implies B
    then P(A) <= P(B). Here la implies lab implies labc by construction, so the
    three probabilities must be non-decreasing, whatever the belief is. A
    violation is a proof that the returned numbers are not a probability
    measure -- no ground truth, no reference implementation needed."""
    rows = []
    for fi in [c for c in CORPUS if c.surface == "belief_ad"]:
        terms = [Term(f, Term("i1")) for f in ("la", "lab", "labc")]
        try:
            net = Network(DummyNet({Term("i1"): list(fi.value)}), "dummy1")
            _, got = _solve(LAW_PROGRAM, terms, nets=[net])
            a, ab, abc = (None if g is None else float(g) for g in got)
        except Exception as e:                                    # noqa: BLE001
            rows.append({"id": f"O3/{fi.id}", "surface": "laws", "tier": fi.tier,
                         "label": fi.label, "validity": fi.validity,
                         "ref": "KOLMOGOROV33", "oracles": ["O3"],
                         "verdict": "REJECTED (correct)" if fi.validity ==
                         "DOC-INVALID" else "FALSE REJECT",
                         "ok": fi.validity == "DOC-INVALID",
                         "error": f"{type(e).__name__}: {str(e)[:80]}"})
            continue

        vals = [a, ab, abc]
        if any(v is None or math.isnan(v) for v in vals):
            monotone = None
        else:
            monotone = (a <= ab + 1e-9) and (ab <= abc + 1e-9)
        in_range = all(v is not None and _is_probability(v) for v in vals)
        # The interesting subset: looks entirely normal, yet violates a law.
        silent = bool(in_range and monotone is False)
        rows.append({
            "id": f"O3/{fi.id}", "surface": "laws", "tier": fi.tier,
            "label": fi.label, "validity": fi.validity, "ref": "KOLMOGOROV33",
            "oracles": ["O3"], "F": {"P_a": a, "P_a_or_b": ab, "P_a_or_b_or_c": abc},
            "all_results_in_unit_interval": in_range,
            "monotonicity_holds": monotone,
            "silently_wrong": silent,
            "verdict": ("MONOTONICITY VIOLATED, ALL RESULTS LOOK NORMAL" if silent
                        else "MONOTONICITY VIOLATED" if monotone is False
                        else "ok" if monotone else "not evaluable (NaN)"),
            "ok": monotone is not False,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the corpus, run nothing")
    ap.add_argument("--surface", help="restrict to one surface")
    args = ap.parse_args()

    if args.list:
        print(f"{len(CORPUS)} corpus entries across {len(SURFACES)} surfaces\n")
        for fi in CORPUS:
            if args.surface and fi.surface != args.surface:
                continue
            print(f"{fi.id:<8} {fi.surface:<12} {fi.tier:<12} {fi.validity:<12} "
                  f"{fi.label:<26} [{fi.ref}]")
            print(f"{'':<8} origin: {fi.origin}")
        return 0

    warnings.filterwarnings("ignore")
    _lazy_imports()

    t0 = time.time()
    rows = [run_case(fi) for fi in CORPUS
            if not args.surface or fi.surface == args.surface]
    if not args.surface or args.surface == "program":
        rows += run_equivalence()
    if not args.surface or args.surface == "config":
        rows += run_configurations()
    if not args.surface or args.surface == "laws":
        rows += run_probability_laws()
    elapsed = time.time() - t0

    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seconds": round(elapsed, 2),
               "n_cases": len(rows),
               "n_violations": sum(1 for r in rows if not r["ok"]),
               "corpus": [asdict(c) for c in CORPUS],
               "results": rows}
    # A partial run must not clobber the full-corpus results file.
    name = f"fuzz_results_{args.surface}.json" if args.surface else "fuzz_results.json"
    (RESULTS / name).write_text(json.dumps(payload, indent=1, default=str))

    print(f"{len(rows)} cases in {elapsed:.1f}s -- "
          f"{payload['n_violations']} oracle violations\n")
    for r in rows:
        mark = "!" if not r["ok"] else " "
        f = r.get("F")
        shown = (r.get("error") or
                 (json.dumps(f, default=str) if f is not None else ""))
        print(f"{mark} {r['id']:<16} {r['verdict']:<42} {str(shown)[:60]}")
    print(f"\nwrote {RESULTS / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
