#!/usr/bin/env python3
"""
Step 3: from "accepted invalid input" to "demonstrably wrong inference".

"The library accepted a malformed belief" is a finding about input validation.
"The library returned a plausible-looking answer that is provably wrong" is a
finding about correctness. This script establishes the second, and explains the
first.

Six parts:

  1  ROOT CAUSE     why invalid values are accepted -- three validation gates
                    that ProbLog implements and DeepProbLog's semiring disables.
  2  STAGE TRACE    one case followed input -> network -> evaluate_nn ->
                    semiring.value -> the actual semiring arithmetic -> result.
  3  SUBSTRATE DIFF the same values through plain ProbLog. It rejects them.
  4  LAW VIOLATION  a belief that passes BOTH the range and the normalisation
                    oracle, and still produces P(A) > P(A or B).
  5  DECISION FLIP  a realistic wiring bug that changes the predicted answer
                    while every reported probability stays inside [0,1].
  6  LEARNING       what an invalid belief does to the loss and the gradient.

Usage:  python trace_case.py            (writes results/downstream_effects.json)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from problog.errors import InvalidValue
from problog.evaluator import SemiringProbability
from problog.logic import Constant, Term
from problog.program import PrologString
from problog import get_evaluatable

from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.semiring.graph_semiring import GraphSemiring
from deepproblog.utils.standard_networks import DummyNet

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
OUT: dict = {}

AD3 = "nn(m,[X],Y,[a,b,c]) :: d(X,Y).\n"
softmax = lambda z: (lambda e: [x / sum(e) for x in e])(
    [math.exp(x - max(z)) for x in z])
sigmoid = lambda z: [1 / (1 + math.exp(-x)) for x in z]


def solve(program, terms, nets=(), cache=False):
    m = Model(program, list(nets), load=False)
    m.set_engine(ExactEngine(m), cache=cache)
    qs = [Query(t) for t in terms]
    res = m.solve(qs)
    return m, [(r.result[q.query] if q.query in r.result else None)
               for q, r in zip(qs, res)]


def belief_net(vectors: dict, name="m"):
    return Network(DummyNet({Term(k): list(v) for k, v in vectors.items()}), name)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 1. ROOT CAUSE
# ===========================================================================
def part1_root_cause():
    hr("1. ROOT CAUSE -- why an invalid probability is accepted at all")

    print("""
ProbLog validates probabilities at three separate gates. DeepProbLog's
GraphSemiring subclasses problog.evaluator.Semiring and disables all three --
one by overriding a method and dropping its check, two by not overriding
methods whose base-class defaults are permissive.
""")

    gates = [
        {"gate": "Semiring.value(a)",
         "problog": "SemiringProbability.value() raises InvalidValue unless "
                    "0 <= v <= 1  (problog/evaluator.py:226-232)",
         "deepproblog": "GraphSemiring.value() OVERRIDES it and returns the "
                        "network output with no range check "
                        "(semiring/graph_semiring.py:64-89)",
         "effect": "any real number can enter the computation as a probability"},
        {"gate": "Semiring.in_domain(a)",
         "problog": "SemiringProbability.in_domain() is the range check; "
                    "ProbLog's AD handling calls it to reject an annotated "
                    "disjunction whose weights exceed 1 (constraint.py:216)",
         "deepproblog": "NOT OVERRIDDEN -- inherits the base default "
                        "`return True` (evaluator.py:134-136)",
         "effect": "the annotated-disjunction sum constraint never fires"},
        {"gate": "Semiring.result_in_domain(a)",
         "problog": "SemiringProbability.result_in_domain() checks the FINAL "
                    "answer; _check_result() raises CompilationError otherwise "
                    "(evaluator.py:458-480)",
         "deepproblog": "NOT OVERRIDDEN -- inherits the base default "
                        "`return True` (evaluator.py:138-145)",
         "effect": "a final query probability outside [0,1] is returned to the "
                   "caller"},
    ]
    for g in gates:
        print(f"  GATE  {g['gate']}")
        print(f"    ProbLog      : {g['problog']}")
        print(f"    DeepProbLog  : {g['deepproblog']}")
        print(f"    consequence  : {g['effect']}\n")

    # Demonstrate gate 1 directly, on the same values, side by side.
    pl = SemiringProbability()
    demo = []
    for v in (0.5, -0.2, 1.7, float("nan")):
        try:
            pl_out = pl.value(Constant(v))
            pl_res = f"accepted {pl_out}"
        except InvalidValue as e:
            pl_res = f"REJECTED InvalidValue: {str(e)[:52]}"
        except Exception as e:                                  # noqa: BLE001
            pl_res = f"REJECTED {type(e).__name__}"
        dpl = GraphSemiring(model=None, substitution={}, values={})
        try:
            dpl_res = f"accepted {dpl.value(Constant(v))}"
        except Exception as e:                                  # noqa: BLE001
            dpl_res = f"REJECTED {type(e).__name__}"
        demo.append({"value": v, "problog_semiring": pl_res,
                     "deepproblog_semiring": dpl_res})
        print(f"  value({v!r:>6})   ProbLog: {pl_res:<52} DeepProbLog: {dpl_res}")

    print("""
  So this is not a missing feature. The validation exists in the parent class
  and is switched off by the subclass. The fix is three small methods.""")
    OUT["root_cause"] = {"gates": gates, "semiring_value_comparison": demo}


# ===========================================================================
# 2. STAGE TRACE
# ===========================================================================
def part2_stage_trace(belief=(0.6, -0.2, 0.6)):
    hr(f"2. STAGE TRACE -- belief {list(belief)} followed end to end")

    ops: list[dict] = []
    orig_value = GraphSemiring.value
    orig_plus, orig_times = GraphSemiring.plus, GraphSemiring.times
    orig_negate = GraphSemiring.negate

    def f(x):
        try:
            return round(float(x), 8)
        except Exception:                                       # noqa: BLE001
            return str(x)

    def v_probe(self, a, key=None):
        out = orig_value(self, a, key)
        ops.append({"op": "value", "term": str(a)[:60], "out": f(out)})
        return out

    def p_probe(self, a, b):
        out = orig_plus(self, a, b)
        ops.append({"op": "plus", "a": f(a), "b": f(b), "out": f(out)})
        return out

    def t_probe(self, a, b):
        out = orig_times(self, a, b)
        ops.append({"op": "times", "a": f(a), "b": f(b), "out": f(out)})
        return out

    def n_probe(self, a):
        out = orig_negate(self, a)
        ops.append({"op": "negate", "a": f(a), "out": f(out)})
        return out

    GraphSemiring.value = v_probe
    GraphSemiring.plus = p_probe
    GraphSemiring.times = t_probe
    GraphSemiring.negate = n_probe

    program = AD3 + "q0(X) :- d(X,a).\nq01(X) :- d(X,a).\nq01(X) :- d(X,b).\n"
    net_out = None
    try:
        net = belief_net({"i": belief})
        orig_call = Network.__call__
        captured = {}

        def call_probe(self, to_evaluate):
            r = orig_call(self, to_evaluate)
            captured["network_output"] = [f(x) for x in
                                          (r[0] if isinstance(r, list) else r)]
            return r

        Network.__call__ = call_probe
        _, got = solve(program, [Term("q01", Term("i"))], nets=[net])
        Network.__call__ = orig_call
        net_out = captured.get("network_output")
        final = f(got[0])
    finally:
        GraphSemiring.value = orig_value
        GraphSemiring.plus = orig_plus
        GraphSemiring.times = orig_times
        GraphSemiring.negate = orig_negate

    print(f"""
  STAGE 1  input                 belief vector {list(belief)}
                                 sum = {sum(belief)}  -> passes a sum-to-one check
                                 entries in [0,1]? {all(0 <= x <= 1 for x in belief)}
  STAGE 2  Network.__call__      emitted {net_out}
                                 no validation here (network.py returns the
                                 module's output verbatim)
  STAGE 3  Model.evaluate_nn     stores the tensor against (net, inputs);
                                 no validation here either
  STAGE 4  GraphSemiring.value   reads entry i of that tensor and returns it as
                                 a probability -- this is the gate that ProbLog
                                 would have closed
  STAGE 5  SDD evaluation        the arithmetic actually performed:""")
    for o in ops[:14]:
        if o["op"] == "value":
            print(f"             value({o['term']}) = {o['out']}")
        elif o["op"] == "negate":
            print(f"             negate({o['a']}) = {o['out']}")
        else:
            print(f"             {o['op']}({o['a']}, {o['b']}) = {o['out']}")
    if len(ops) > 14:
        print(f"             ... {len(ops) - 14} more operations")
    print(f"""
  STAGE 6  result                P(q01) = {final}
                                 in [0,1]? {0 <= float(final) <= 1}
                                 -> returned to the caller with no warning

  The negative value is never rejected, never clamped, and never flagged. It is
  simply added: the disjunction q01 = d(a) or d(b) sums 0.6 and -0.2.""")
    OUT["stage_trace"] = {"belief": list(belief), "network_output": net_out,
                          "semiring_operations": ops, "final": final}


# ===========================================================================
# 3. SUBSTRATE DIFFERENTIAL
# ===========================================================================
def part3_substrate():
    hr("3. SUBSTRATE DIFFERENTIAL -- the same values through plain ProbLog")

    cases = {
        "valid AD":             "0.6::d(a); 0.2::d(b); 0.2::d(c).",
        "negative mass, sum=1": "0.6::d(a); -0.2::d(b); 0.6::d(c).",
        "sigmoid (sum > 1)":    "0.9::d(a); 0.9::d(b); 0.9::d(c).",
        "raw logits":           "3.2::d(a); -1.1::d(b); 0.4::d(c).",
        "negative fact":        "-0.1::f.  q0 :- f.  q01 :- f.",
        "fact above one":       "1.1::f.  q0 :- f.  q01 :- f.",
    }
    rows = []
    for label, head in cases.items():
        src = head
        if "::f." not in head:
            src += "\nq0 :- d(a).\nq01 :- d(a).\nq01 :- d(b)."
        src += "\nquery(q0).\nquery(q01)."
        try:
            r = get_evaluatable().create_from(PrologString(src)).evaluate()
            verdict = "ACCEPTED -> " + str({str(k): round(float(v), 6)
                                            for k, v in r.items()})
        except Exception as e:                                  # noqa: BLE001
            verdict = f"REJECTED {type(e).__name__}: {str(e)[:70]}"
        rows.append({"case": label, "problog": verdict})
        print(f"  {label:<22} {verdict}")

    print("""
  Every malformed case is rejected by ProbLog with a named, specific error.
  DeepProbLog accepts all of them and returns a number. Both run on the same
  grounder and the same knowledge compiler; the only difference is the semiring
  object, which is where the checks live.""")
    OUT["substrate_differential"] = rows


# ===========================================================================
# 4. PROBABILITY-LAW VIOLATION WITH A PLAUSIBLE-LOOKING RESULT
# ===========================================================================
def part4_law_violation():
    hr("4. LAW VIOLATION -- passes range AND normalisation, still provably wrong")

    program = (AD3 + "q0(X) :- d(X,a).\n"
               "q01(X) :- d(X,a).\nq01(X) :- d(X,b).\n"
               "q012(X) :- d(X,a).\nq012(X) :- d(X,b).\nq012(X) :- d(X,c).\n")
    rows = []
    for label, v in [("valid", [0.6, 0.2, 0.2]),
                     ("negative mass, sums to 1.0", [0.6, -0.2, 0.6]),
                     ("negative mass, sums to 1.0 (2)", [0.7, -0.4, 0.7])]:
        _, got = solve(program, [Term(f, Term("i")) for f in
                                 ("q0", "q01", "q012")], nets=[belief_net({"i": v})])
        a, ab, abc = (float(x) for x in got)
        in_range = all(0.0 <= x <= 1.0 for x in (a, ab, abc))
        monotone = a <= ab + 1e-9 <= abc + 1e-9
        rows.append({"case": label, "belief": v, "belief_sum": sum(v),
                     "P_a": a, "P_a_or_b": ab, "P_a_or_b_or_c": abc,
                     "all_results_in_unit_interval": in_range,
                     "monotonicity_holds": bool(monotone)})
        print(f"\n  {label}")
        print(f"    belief {v}   sum = {sum(v)}")
        print(f"    P(a)         = {a:.6f}")
        print(f"    P(a or b)    = {ab:.6f}")
        print(f"    P(a or b or c) = {abc:.6f}")
        print(f"    every result in [0,1]:      {in_range}")
        print(f"    monotonicity P(A) <= P(A|B): {monotone}"
              f"{'' if monotone else '   <-- IMPOSSIBLE IN ANY PROBABILITY MODEL'}")

    print("""
  This is the important case. The belief [0.6, -0.2, 0.6] sums to exactly 1.0,
  so it passes the normalisation oracle. Every probability the library returns
  is inside [0,1], so it passes the range oracle. A user sees three ordinary
  numbers. And yet a disjunction is strictly less probable than one of its own
  disjuncts, which no probability measure permits (monotonicity follows directly
  from the Kolmogorov axioms).

  So range and normalisation oracles are NOT sufficient. A NeSy test suite needs
  logical-law oracles -- monotonicity, the conjunction bound, inclusion-exclusion
  -- because they are the only ones that catch this class.""")
    OUT["law_violation"] = rows


# ===========================================================================
# 5. DECISION FLIP
# ===========================================================================
def part5_decision_flip(trials=60, seed=7):
    hr("5. DECISION FLIP -- a wrong predicted answer, with no visible symptom")

    program = ("nn(m,[X],Y,[0,1,2]) :: digit(X,Y).\n"
               "addition(A,B,S) :- digit(A,X), digit(B,Y), S is X+Y.\n")

    def sums(v1, v2):
        net = belief_net({"i1": v1, "i2": v2})
        ts = [Term("addition", Term("i1"), Term("i2"), Constant(s))
              for s in range(5)]
        _, got = solve(program, ts, nets=[net])
        return [float(x) if x is not None else 0.0 for x in got]

    print("""
  Setup: MNIST-addition in miniature. Two images, digits from {0,1,2}, query the
  sum. The wiring bug is a sigmoid head (multi-label) where a softmax head
  (single-label) belongs -- so every belief entry is individually inside [0,1]
  and a per-entry range check passes. Only the SUM is wrong.

  We keep only trials where every reported P(sum) also lands inside [0,1], i.e.
  where the user can see nothing unusual.
""")
    rnd = random.Random(seed)
    hits, examined, inrange = [], 0, 0
    for _ in range(trials):
        z1 = [rnd.uniform(-2.5, 2.5) for _ in range(3)]
        z2 = [rnd.uniform(-2.5, 2.5) for _ in range(3)]
        examined += 1
        good, bad = sums(softmax(z1), softmax(z2)), sums(sigmoid(z1), sigmoid(z2))
        if not all(0 <= x <= 1 for x in bad):
            continue
        inrange += 1
        ag = max(range(5), key=lambda s: good[s])
        ab = max(range(5), key=lambda s: bad[s])
        if ag != ab:
            hits.append({"logits": [z1, z2],
                         "intended_belief": [softmax(z1), softmax(z2)],
                         "buggy_belief": [sigmoid(z1), sigmoid(z2)],
                         "P_sum_intended": good, "P_sum_buggy": bad,
                         "predicted_intended": ag, "predicted_buggy": ab})

    print(f"  {examined} trials; {inrange} produced only in-range probabilities; "
          f"{len(hits)} of those flipped the predicted sum.\n")
    for h in hits[:3]:
        print(f"  intended belief {[round(x,4) for x in h['intended_belief'][0]]} "
              f"{[round(x,4) for x in h['intended_belief'][1]]}")
        print(f"  buggy    belief {[round(x,4) for x in h['buggy_belief'][0]]} "
              f"{[round(x,4) for x in h['buggy_belief'][1]]}  (all entries in [0,1])")
        print(f"    P(sum) intended {[round(x,5) for x in h['P_sum_intended']]}")
        print(f"    P(sum) buggy    {[round(x,5) for x in h['P_sum_buggy']]}")
        print(f"    PREDICTED SUM   intended={h['predicted_intended']}  "
              f"buggy={h['predicted_buggy']}   <-- WRONG ANSWER\n")

    print("""  Standard evaluation for this task takes the argmax. The argmax
  discards the magnitudes, so the only symptom -- that the probabilities do not
  sum to 1 -- is thrown away before anyone looks at it. Accuracy drops and
  nothing in the stack reports a fault.""")
    OUT["decision_flip"] = {"trials": examined, "in_range_trials": inrange,
                            "flips": len(hits), "examples": hits[:5]}


# ===========================================================================
# 6. LEARNING
# ===========================================================================
def part6_learning():
    hr("6. LEARNING -- finite loss, zero gradient")

    class Head(torch.nn.Module):
        def __init__(self, logits, normalise):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor(logits))
            self.normalise = normalise

        def forward(self, *_):
            return (torch.softmax(self.logits, -1) if self.normalise
                    else self.logits)

    program = AD3 + "q(X,Y) :- d(X,Y).\n"
    rows = []
    for label, normalise in [("correct (softmax head)", True),
                             ("BUG (no softmax; a logit is negative)", False)]:
        mod = Head([1.0, 0.2, -0.5], normalise)
        net = Network(mod, "m", optimizer=torch.optim.SGD(mod.parameters(), lr=0.1))
        m = Model(program, [net], load=False)
        m.set_engine(ExactEngine(m), cache=False)
        t = Term("q", Term("i"), Term("c"))   # train toward the negative class
        m.optimizer.zero_grad()
        res = m.solve([Query(t)])[0]
        p = res.result[t]
        loss = GraphSemiring.cross_entropy(res, target=1.0, weight=1.0, q=t)
        g = mod.logits.grad
        gn = 0.0 if g is None else float(g.norm())
        rows.append({"case": label, "F": float(p), "loss": loss,
                     "loss_finite": math.isfinite(loss),
                     "gradient": None if g is None else [float(x) for x in g],
                     "gradient_norm": gn})
        print(f"\n  {label}")
        print(f"    F               = {float(p):.6f}")
        print(f"    loss            = {loss:.6f}   (finite: {math.isfinite(loss)})")
        print(f"    gradient        = {None if g is None else [round(float(x), 8) for x in g]}")
        print(f"    gradient norm   = {gn:.3e}")

    print("""
  GraphSemiring.cross_entropy computes  pos = p.clamp(min=0.0)  and then
  -log(pos + eps). For a negative p the clamp returns exactly 0, so the loss is
  -log(1e-12) = 27.63 -- a large but perfectly finite number -- and clamp has
  zero derivative in the clamped region, so the gradient is exactly zero.

  The training loop therefore reports a plausible loss and learns nothing from
  that example. No exception, no NaN, no warning. A run that silently stops
  learning looks exactly like a run that is training badly, which is the hardest
  kind of fault to attribute.""")
    OUT["learning"] = rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, help="run only one part (1-6)")
    args = ap.parse_args()
    parts = {1: part1_root_cause, 2: part2_stage_trace, 3: part3_substrate,
             4: part4_law_violation, 5: part5_decision_flip, 6: part6_learning}
    for i, fn in parts.items():
        if args.part in (None, i):
            fn()
    path = RESULTS / "downstream_effects.json"
    path.write_text(json.dumps(OUT, indent=1, default=str))
    print(f"\n\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
