#!/usr/bin/env python3
"""
Step 4: the minimal, reviewer-proof proof-of-concept.

Four parts, each answering a specific objection to the §5 findings.

  A  CONTRACT   What does DeepProbLog actually require of a neural output?
                Classified into: stated in the defining publication / enforced
                by construction in the shipped examples / stated in the API
                docstrings / checked at runtime. These are four different
                things and the distinction decides how the finding is worded.

  B  SEMANTICS  Proof that P(A) and P(A or B) refer to the same events in the
                same probability model, by printing the GROUND FORMULA the
                library compiles. Containment is then syntactic, not a matter
                of interpretation.

  C  TABLE      Three cases, one table: valid / invalid-but-sums-to-one /
                invalid-sum. Input, per-entry validity, sum validity,
                acceptance, outputs, oracle verdicts, law verdict.

  D  GRADIENT   How far the zero-gradient result generalises: swept over
                belief values, target classes and BOTH loss functions, so the
                claim can be scoped to what was actually measured.

Environment is recorded in the JSON; validation behaviour is version-specific.

Usage:  python minimal_poc.py      (writes results/minimal_poc.json)
"""
from __future__ import annotations

import importlib.metadata as md
import json
import math
import pathlib
import platform
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from problog.logic import Constant, Term
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

# The program used throughout. la is one disjunct of lab by construction.
PROGRAM = ("nn(m,[X],Y,[a,b,c]) :: d(X,Y).\n"
           "la(X)   :- d(X,a).\n"
           "lab(X)  :- d(X,a).\nlab(X)  :- d(X,b).\n"
           "labc(X) :- d(X,a).\nlabc(X) :- d(X,b).\nlabc(X) :- d(X,c).\n")


def env() -> dict:
    pkgs = {}
    for d in ("deepproblog", "problog", "torch", "pysdd", "pytest"):
        try:
            pkgs[d] = md.version(d)
        except md.PackageNotFoundError:
            pkgs[d] = None
    return {"packages": pkgs, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine()}


def build(belief):
    net = Network(DummyNet({Term("i"): list(belief)}), "m")
    m = Model(PROGRAM, [net], load=False)
    m.set_engine(ExactEngine(m), cache=False)
    return m


def query(m, functor):
    t = Term(functor, Term("i"))
    r = m.solve([Query(t)])[0]
    return float(r.result[t]) if t in r.result else None


def hr(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


# ===========================================================================
# A. THE CONTRACT
# ===========================================================================
def part_a_contract():
    hr("A. WHAT IS THE CONTRACT ON A NEURAL OUTPUT?")

    print("""
  Four distinct questions, four different answers. Conflating them is what a
  reviewer will attack, so they are separated here.

  A1. Stated in the defining publication?                        YES
      Manhaeve et al., "Neural Probabilistic Logic Programming in DeepProbLog",
      AIJ 298:103504 (arXiv:1907.08194), describes the neural predicate as

        "allowing atomic expressions to be labeled with neural networks whose
         outputs can be considered probability distributions."

      and, in its deep-learning background,

        "The softmax outputs are well-suited to model a probability
         distribution (i.e. 0 < y_i < 1 and sum_i y_i = 1)."

      So the requirement is stated, in prose, by the paper that defines the
      construct. It is an explicit semantic requirement, not an inference of
      ours.

  A2. Enforced by construction in the shipped examples?           YES""")

    # Measured, not asserted. AST-based, per CLASS, so an __init__.py that
    # merely imports torch is not counted as a network. Scans the repo checkout
    # when DPL_SOURCE_EXAMPLES points at one (the installed wheel ships fewer
    # example files than the repository), otherwise the installed package.
    import ast, os
    import deepproblog
    ex = pathlib.Path(os.environ.get("DPL_SOURCE_EXAMPLES")
                      or (pathlib.Path(deepproblog.__file__).parent / "examples"))
    print(f"      scanned: {ex}")
    classes = []
    for path in sorted(ex.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        src = path.read_text(errors="ignore")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                elif isinstance(b, ast.Name):
                    bases.append(b.id)
            if "Module" not in bases:
                continue
            body = ast.get_source_segment(src, node) or ""
            norm = [k for k in ("Softmax", "softmax", "Sigmoid", "sigmoid")
                    if k in body]
            classes.append({"file": str(path.relative_to(ex)), "class": node.name,
                            "normalisers": sorted(set(norm))})
    normalised = [c for c in classes if c["normalisers"]]
    unnormalised = [c for c in classes if not c["normalisers"]]
    print(f"      {len(classes)} classes inherit nn.Module across the examples.")
    print(f"      with a normalising layer in the class body: {len(normalised)}")
    for c in normalised:
        print(f"        {c['file']}::{c['class']}  {c['normalisers']}")
    print(f"      without:                                     {len(unnormalised)}")
    for c in unnormalised:
        print(f"        {c['file']}::{c['class']}")
    # The per-class scan is necessary but not sufficient: several example
    # networks normalise by COMPOSITION, inheriting the layer from
    # deepproblog.utils.standard_networks rather than declaring it in their own
    # body. Check those helpers directly.
    import deepproblog.utils.standard_networks as sn
    helper_src = pathlib.Path(sn.__file__).read_text()
    mlp_default_softmax = "softmax=True" in helper_src
    smallnet_final = "self.final = nn.Sigmoid() if num_classes == 1 else nn.Softmax(1)" \
        in helper_src.replace("\n", " ").replace("  ", " ")
    print(f"""
      Composition check (deepproblog/utils/standard_networks.py):
        MLP defaults to softmax=True and appends nn.Softmax(-1):   {mlp_default_softmax}
        SmallNet final layer is Sigmoid (1 class) else Softmax(1): {smallnet_final}

      That resolves the six classes above that carry no normaliser in their own
      body. Resolved by hand, with references, because composition is not
      something the textual scan can follow:

        Forth/__init__.py::EncodeModule   composes standard_networks.MLP, which
                                          appends nn.Softmax(-1) by default;
                                          registered for nn/4 in
                                          Forth/Sort/compare.pl:1 and
                                          Forth/Add/choose.pl:1-2
        Forth/WAP/wap_network.py::RNN     registered for nn/3 DETERMINISTIC
                                          (wap.pl:1, an embedding, not a belief)
                                          -- correctly unnormalised
        CLUTRR/architecture.py::Encoder   an internal LSTM encoder
        HWF/network.py::SymbolEncoder     an internal encoder
        MNIST/network.py::MNIST_CNN       an internal backbone
        MNIST/neural_baseline/...::Separate_Baseline_Multi
                                          a NEURAL BASELINE, not part of any
                                          DeepProbLog program

      Poker and Coins register standard_networks.smallnet, covered by the
      SmallNet check above.

      CONCLUSION, as measured plus the resolution above: no shipped example
      registers an unnormalised module against a PROBABILISTIC neural predicate
      (nn/2 or nn/4). The single unnormalised registration is the WAP RNN, and
      it feeds a deterministic nn/3 predicate, where any real value is correct.

      Note also MNIST_Net (examples/MNIST/network.py:49-53):
          if n == 1:  self.softmax = nn.Sigmoid()
          else:       self.softmax = nn.Softmax(1)
      The maintainers themselves use sigmoid for a single Bernoulli fact and
      softmax for a categorical domain. That distinction is the library's own,
      and it is the basis of case B-A-02 -- sigmoid is not "wrong", it is wrong
      FOR A CATEGORICAL ANNOTATED DISJUNCTION.
""")

    print("""  A3. Stated in the API docstrings?                              NO
      Network.__init__ documents network_module, name, optimizer, scheduler, k
      and batching. Nothing about the range, normalisation or shape of the
      module's output. A user reading only the API has no statement of it.

  A4. Checked at runtime?                                        NO
      See part B and trace_case.py part 1.

  CLASSIFICATION. This is a semantic requirement stated in the defining
  publication and honoured throughout the library's own examples, but not
  restated at the API boundary and not enforced in code. It is therefore
  accurate to call it a documented invariant that is unchecked, and NOT
  accurate to call it merely "behaviour outside the intended input domain" --
  the intended domain is stated, and the library silently computes outside it.""")

    OUT["contract"] = {
        "stated_in_publication": True,
        "publication": "Manhaeve et al., AIJ 298:103504 (arXiv:1907.08194)",
        "quotes": [
            "allowing atomic expressions to be labeled with neural networks "
            "whose outputs can be considered probability distributions",
            "The softmax outputs are well-suited to model a probability "
            "distribution (i.e. 0 < y_i < 1 and sum_i y_i = 1)"],
        "examples_scanned": str(ex),
        "nn_module_classes_total": len(classes),
        "classes_with_normaliser": normalised,
        "classes_without_normaliser": unnormalised,
        "stated_in_api_docstrings": False,
        "checked_at_runtime": False,
        "mlp_default_softmax": mlp_default_softmax,
        "smallnet_final_layer_checked": smallnet_final,
        "classification": "documented invariant (publication + examples), "
                          "unstated at the API boundary, unenforced in code",
        "caveat": "A2 combines an automated per-class scan with a hand "
                  "resolution of six classes that normalise by composition or "
                  "are internal encoders; the resolution is listed with source "
                  "references and is not machine-checked.",
    }


# ===========================================================================
# B. THE SEMANTICS ARE THE SAME
# ===========================================================================
def part_b_semantics():
    hr("B. DO P(A) AND P(A or B) REFER TO THE SAME EVENTS?")

    print("""
  The objection: maybe `la` and `lab` are evaluated in different models, or the
  annotated disjunction gives the atoms some library-specific meaning that makes
  the comparison invalid.

  Settled by printing the ground formula the library actually compiles.
""")
    m = build([0.6, 0.2, 0.2])
    formulas = {}
    for f in ("la", "lab"):
        q = Query(Term(f, Term("i")))
        ac = m.solver.build_ac(q)
        text = str(ac.proof)
        formulas[f] = {"formula": text,
                       "named_nodes": {str(k): v for k, v in ac.get_named().items()}}
        print(f"  --- ground formula for {f}(i) ---")
        for line in text.splitlines():
            print("   ", line)
        print()

    print("""  Read it off:

    la(i)  is node 1: atom(identifier=(8,(i,),0), name=choice(8,0,d(i,a),i))
    lab(i) is node 4: disj(children=(1, 2), name=lab(i))

  Node 1 is the SAME node in both formulas -- same identifier (8,(i,),0), same
  group (8,(i,)), same probability term nn(m,[i],0). lab is compiled as the
  disjunction of node 1 with node 2, where node 2 is the atom for d(i,b) in the
  same annotated-disjunction group.

  So in the compiled formula, the event of la is literally one of the two
  disjuncts of the event of lab:

        la  =  n1
        lab =  n1 v n2       hence   la |= lab      (syntactically)

  Monotonicity (A subset of B implies P(A) <= P(B)) is a theorem of the
  Kolmogorov axioms. Containment here is a property of the formula the library
  compiled, not of our reading of the program. The comparison is therefore
  sound, and P(la) > P(lab) is a contradiction in the library's own model.

  Note also the constraint line: annotated_disjunction([1, 2], 3), with node 3
  the "extra" null choice. That constraint is exactly where ProbLog consults
  Semiring.in_domain -- the gate GraphSemiring leaves at the permissive base
  default.""")
    OUT["semantics"] = formulas


# ===========================================================================
# C. THE THREE-CASE TABLE
# ===========================================================================
ORACLES = {
    "O1 range": "every probability the library RETURNS is in [0,1] and finite",
    "O4 normalisation": "the belief vector supplied to an exhaustive nn/4 "
                        "annotated disjunction sums to 1 (checked on the INPUT "
                        "belief, which is the observation point a user or a "
                        "wrapper can see)",
    "O3 logical law": "monotonicity P(la) <= P(lab) <= P(labc), where la |= lab "
                      "|= labc holds syntactically in the compiled formula "
                      "(part B)",
}


def part_c_table():
    hr("C. THREE CASES, ONE TABLE")
    print("\n  Oracle definitions, stated exactly, at the observation point:")
    for k, v in ORACLES.items():
        print(f"    {k:<18} {v}")

    cases = [
        ("A", "valid distribution", [0.6, 0.2, 0.2]),
        ("B", "invalid entry, sum = 1", [0.6, -0.2, 0.6]),
        ("C", "valid entries, invalid sum", [0.9, 0.9, 0.9]),
    ]
    rows = []
    for tag, label, belief in cases:
        entries_ok = all(0.0 <= x <= 1.0 for x in belief)
        sum_ok = abs(sum(belief) - 1.0) < 1e-6
        try:
            m = build(belief)
            a, ab, abc = (query(m, f) for f in ("la", "lab", "labc"))
            accepted = True
            err = None
        except Exception as e:                                  # noqa: BLE001
            a = ab = abc = None
            accepted = False
            err = f"{type(e).__name__}: {e}"

        o1 = (accepted and all(v is not None and math.isfinite(v)
                               and -1e-9 <= v <= 1 + 1e-9 for v in (a, ab, abc)))
        o4 = sum_ok
        o3 = (accepted and a is not None
              and a <= ab + 1e-9 and ab <= abc + 1e-9)
        rows.append({"case": tag, "label": label, "belief": belief,
                     "entries_in_unit_interval": entries_ok,
                     "belief_sums_to_one": sum_ok, "accepted": accepted,
                     "error": err, "P_la": a, "P_lab": ab, "P_labc": abc,
                     "O1_range_passes": o1, "O4_normalisation_passes": o4,
                     "O3_monotonicity_passes": o3})

    print(f"\n  {'':<4}{'belief':<22}{'entries':<9}{'sum':<7}{'accepted':<10}"
          f"{'P(la)':>9}{'P(lab)':>9}{'P(labc)':>9}  {'O1':<5}{'O4':<5}{'O3':<5}")
    for r in rows:
        chk = lambda b: " ok " if b else " X  "                 # noqa: E731
        print(f"  {r['case']:<4}{str(r['belief']):<22}"
              f"{chk(r['entries_in_unit_interval']):<9}"
              f"{chk(r['belief_sums_to_one']):<7}"
              f"{chk(r['accepted']):<10}"
              f"{r['P_la']:>9.4f}{r['P_lab']:>9.4f}{r['P_labc']:>9.4f}  "
              f"{chk(r['O1_range_passes']):<5}"
              f"{chk(r['O4_normalisation_passes']):<5}"
              f"{chk(r['O3_monotonicity_passes']):<5}")

    print("""
  Case A is the control: valid in, valid out, all three oracles pass.

  Case B is the finding. Its belief has a negative entry, so it is not a
  probability distribution -- but the vector sums to 1, so O4 passes, and every
  probability the library returns lies in [0,1], so O1 passes. Only O3 fails.
  Range and normalisation checks together CERTIFY an output that violates a law
  of probability.

  Case C fails O4 at the input and O3 at the output, so it is the easier case:
  a wrapper checking the belief vector would catch it. Case B would not be
  caught by that wrapper.""")
    OUT["three_case_table"] = {"oracle_definitions": ORACLES, "rows": rows}


# ===========================================================================
# D. HOW FAR DOES THE ZERO GRADIENT GENERALISE?
# ===========================================================================
def part_d_gradient():
    hr("D. SCOPE OF THE ZERO-GRADIENT RESULT")

    print("""
  The earlier result was a single configuration. Claiming "training silently
  stops learning" from one point would be an overreach, so it is swept here over
  belief values, target classes and both loss functions the library ships.
""")

    class Head(torch.nn.Module):
        def __init__(self, vec):
            super().__init__()
            self.vec = torch.nn.Parameter(torch.tensor(vec))

        def forward(self, *_):
            return self.vec

    prog = "nn(m,[X],Y,[a,b,c]) :: d(X,Y).\nq(X,Y) :- d(X,Y).\n"
    beliefs = {
        "valid":            [0.6, 0.2, 0.2],
        "small negative":   [0.6, -0.01, 0.41],
        "negative":         [0.6, -0.2, 0.6],
        "large negative":   [0.9, -0.8, 0.9],
        "above one":        [1.4, -0.2, -0.2],
    }
    rows = []
    print(f"  {'belief':<18}{'target':<9}{'loss fn':<16}{'F':>10}"
          f"{'loss':>12}{'|grad|':>12}  note")
    for blabel, vec in beliefs.items():
        for target_class in ("a", "b", "c"):
            for lname, lfn in (("cross_entropy", GraphSemiring.cross_entropy),
                               ("mse", GraphSemiring.mse)):
                mod = Head(vec)
                net = Network(mod, "m",
                              optimizer=torch.optim.SGD(mod.parameters(), lr=0.1))
                m = Model(prog, [net], load=False)
                m.set_engine(ExactEngine(m), cache=False)
                t = Term("q", Term("i"), Term(target_class))
                m.optimizer.zero_grad()
                res = m.solve([Query(t)])[0]
                p = res.result[t]
                try:
                    loss = lfn(res, target=1.0, weight=1.0, q=t)
                except Exception as e:                          # noqa: BLE001
                    loss = float("nan")
                g = mod.vec.grad
                gn = 0.0 if g is None else float(g.norm())
                dead = gn == 0.0
                note = "ZERO GRADIENT" if dead else ""
                rows.append({"belief": blabel, "belief_vector": vec,
                             "target_class": target_class, "loss_fn": lname,
                             "F": float(p), "loss": loss, "grad_norm": gn,
                             "zero_gradient": dead})
                print(f"  {blabel:<18}{target_class:<9}{lname:<16}"
                      f"{float(p):>10.4f}{loss:>12.4f}{gn:>12.3e}  {note}")

    ce = [r for r in rows if r["loss_fn"] == "cross_entropy"]
    mse = [r for r in rows if r["loss_fn"] == "mse"]
    ce_dead_neg = [r for r in ce if r["F"] < 0 and r["zero_gradient"]]
    ce_neg = [r for r in ce if r["F"] < 0]
    mse_dead_neg = [r for r in mse if r["F"] < 0 and r["zero_gradient"]]

    print(f"""
  SCOPE OF THE CLAIM, as measured:

    cross_entropy, F < 0 : {len(ce_dead_neg)}/{len(ce_neg)} produced a zero gradient
    mse,           F < 0 : {len(mse_dead_neg)}/{len(ce_neg)} produced a zero gradient

  The zero gradient is a property of GraphSemiring.cross_entropy, whose
  `p.clamp(min=0.0)` has zero derivative below 0. GraphSemiring.mse has no
  clamp, so it keeps a gradient -- on a value that is not a probability, which
  is its own problem, but a different one.

  So the defensible statement is:

    "When a neural predicate emits a negative value and cross_entropy is used,
     the loss is finite and the gradient with respect to that value is exactly
     zero, so that example contributes nothing to the update."

  NOT: "DeepProbLog training silently stops learning." Whether this occurs in a
  real training run depends on how often a negative value is produced, which
  this experiment does not measure.

  One further observation from the sweep, not part of the original claim: the
  belief [1.4, -0.2, -0.2] with target class a gives F = 1.4 and a cross-entropy
  loss of -0.3365. A NEGATIVE cross-entropy is impossible for a probability,
  since -log(p) >= 0 for p <= 1. It arises because F > 1 makes log(F) positive.
  A training curve that dips below zero is a visible symptom -- unlike the
  others in this study -- but only if someone is watching for it.""")
    OUT["gradient_scope"] = {"rows": rows,
                             "cross_entropy_negative_F_zero_grad":
                                 [len(ce_dead_neg), len(ce_neg)],
                             "mse_negative_F_zero_grad":
                                 [len(mse_dead_neg), len(ce_neg)]}


def main() -> int:
    OUT["environment"] = env()
    print(f"environment: { {k: v for k, v in OUT['environment']['packages'].items()} }"
          f"  python {OUT['environment']['python']}")
    part_a_contract()
    part_b_semantics()
    part_c_table()
    part_d_gradient()
    path = RESULTS / "minimal_poc.json"
    path.write_text(json.dumps(OUT, indent=1, default=str))
    print(f"\n\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
