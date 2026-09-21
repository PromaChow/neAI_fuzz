# Fuzzing Neurosymbolic AI Libraries: LTN, LNN, and DeepProbLog

This project investigates input validation in three neurosymbolic AI libraries:

- **LTNtorch (LTN)**
- **IBM Logical Neural Networks (LNN)**
- **DeepProbLog**

The main objective was to determine whether documented input constraints are actually enforced by the libraries at runtime.

The experiments were performed using the real installed libraries. No replacement implementations or mock versions were used.

---

## 1. Methodology

The same basic principles were followed for all three libraries.

### Rule 1: Tests Run

Fuzzing was performed only after the library's existing tests had been run.

This was done to make sure that failures found during fuzzing were not simply caused by an already broken installation or test environment.

### Rule 2: Documentation

Only rules that could be connected to one of the following were tested:

- the library's documentation
- the library's source code
- the library's existing tests

For example, if a parameter was documented as needing to be between 0 and 1, values outside that range were treated as DOC-INVALID.

### Rule 3: Library Methods invoked by tests are run

Every fuzz case was sent through the actual installed library.

The behavior was therefore measured from the real implementation rather than from a reimplementation of the expected behavior.

### Rule 4: Inputs were classified as DOC-VALID or DOC-INVALID

A **DOC-VALID** input satisfies the documented requirement and should be accepted.

A **DOC-INVALID** input violates the documented requirement and should be rejected.

If an invalid input was silently accepted and a numerical result was returned, it was treated as an incorrect result.

---

# 2. Fuzzing Algorithm

Two general approaches were used to select and test inputs.

## Finding Real Literals

The library's own test files were searched for real numeric literals.

The following process was used:

1. The test file was parsed.
2. Calls to library classes or functions were located.
3. Imported names were resolved to their actual library definitions.
4. The real constructor or function signature was checked.
5. Only calls containing explicit literal values were kept.

This approach was used because a literal value can be mutated without having to guess what a dynamically generated value should contain.

## Mutation

After a suitable value had been identified:

1. The original value was recorded.
2. It was replaced with boundary or invalid values.
3. The real library object was constructed.
4. The real operation was executed.
5. The result was compared with the documented rule.

Typical mutations included:

- `0`
- negative numbers
- fractional numbers
- values slightly outside a valid range
- `NaN`
- positive infinity
- negative infinity
- very large numbers

The number of fuzz cases was determined by:

**number of targets × number of mutations**

When more than one rule was being checked for the same input, the resulting checks were counted separately.

---

# 3. Overall Results

| Library / Fuzzer | Own Tests | Fuzz Cases | Incorrect Cases |
|---|---:|---:|---:|
| LTNtorch | 11 / 11 | 418 | 115 |
| LNN | 117 / 117 | 40 | 12 |
| DeepProbLog, deep pipeline | 65 / 76 | 116 | 24 |
| DeepProbLog, literal scan | 65 / 76 | 56 | 24 |

\* 11 DeepProbLog tests were skipped.

The fuzz-case totals were obtained directly from the targets and mutations used in each experiment. They were not selected as arbitrary numbers.

---

# 4. LTNtorch

The LTNtorch experiments were performed during the current session using a freshly installed copy of the actual library.

Two separate fuzzing methods were used.

---

## 4.1 Automatic Literal Discovery

The first experiment was based on Algorithm A.

The LTNtorch test file was searched for calls that constructed real library classes using explicit numeric values.

A total of **27 call sites** were found.

Of these:

- 23 constructed `AggregPMean` or `AggregPMeanError` using `p=2`.
- 4 used `stable=False` for other operators.

The four `stable=False` cases were executed as control cases, but they were not considered bug cases because `stable=False` is a supported option.

The important documented rule for `AggregPMean` and `AggregPMeanError` is:

> `p` must be greater than or equal to 1.

The 23 relevant call sites were each tested with eight mutations:

1. zero
2. negative value
3. half
4. `0.5`
5. `NaN`
6. positive infinity
7. negative infinity
8. a very large value

Therefore:

```text
23 call sites × 8 mutations = 184 scored cases
```

The four control cases were also executed once:

```text
184 + 4 = 188 total cases
```

This is why **188 cases** were produced by this experiment.

### Result

Out of the 184 scored cases, **115 incorrect results** were observed.

Negative, fractional, NaN, and infinite values were accepted and produced numerical outputs even though they violated the documented constraint on `p`.

For `p=0`, a division-related error was produced. The invalid value was therefore not rejected according to the documented parameter constraint. Instead, it was allowed to reach a later calculation.

The main problem was that several invalid values were silently accepted.

---

## 4.2 Truth-Value Validation

A second LTNtorch experiment was performed because the truth-value rule applied to a broader set of classes and could not be obtained cleanly through Algorithm A.

The target list was constructed from the source code.

It contained:

- 18 connective configurations
- 4 aggregator configurations
- 1 predicate mechanism

Therefore:

```text
18 + 4 + 1 = 23 targets
```

Ten values were tested for every target.

### Valid values

- `0.0`
- `1.0`
- `0.5`

### Invalid finite values

- `-1e-6`
- `-0.5`
- `1 + 1e-6`
- `1.5`

### Invalid non-finite values

- `NaN`
- positive infinity
- negative infinity

Therefore:

```text
23 targets × 10 values = 230 cases
```

This is why the second experiment contains **230 cases**.

### Result

All **230 cases were handled correctly**.

Valid values were accepted, and invalid values were rejected.

This produced an interesting contrast within the same library.

The truth-value constraint was enforced correctly, while the `p >= 1` constraint was not enforced consistently.

### LTNtorch total

The two experiments produced:

```text
188 + 230 = 418 total fuzz cases
```

There were:

```text
115 incorrect cases
```

All 115 were found in the experiment involving `p`.

---

# 5. IBM Logical Neural Networks (LNN)

For LNN, Algorithm A did not produce suitable numeric targets.

The existing tests mainly used boolean configuration options. Explicit numeric values for weights, biases, and similar parameters were not present in the required form.

Therefore, a manual target list was created directly from the LNN source code and its existing truth-table tests.

Two connectives were tested:

- `And`
- `Or`

Three groups of parameters were examined.

---

## 5.1 Weight Values

Eight weight cases were tested:

1. default
2. negative
3. all-negative
4. zero
5. NaN
6. positive infinity
7. negative infinity
8. very large value

## 5.2 Bias Values

Six bias cases were tested:

1. default
2. negative
3. zero
4. NaN
5. positive infinity
6. very large value

## 5.3 Alpha Values

Six alpha cases were tested:

1. default
2. valid high value
3. exact boundary
4. below the allowed range
5. above the allowed range
6. valid-range value below the arity requirement

For each connective:

```text
8 weight cases
+ 6 bias cases
+ 6 alpha cases
= 20 cases
```

Since two connectives were tested:

```text
2 connectives × 20 cases = 40 cases
```

This is why **40 fuzz cases** were produced for LNN.

### Result

**12 incorrect cases** were found.

For weights, negative, all-negative, and zero values caused the expected truth-table behavior to fail for both connectives:

```text
3 problematic weight values × 2 connectives = 6 cases
```

For biases, negative, zero, and very large values caused incorrect results for both connectives:

```text
3 problematic bias values × 2 connectives = 6 cases
```

Therefore:

```text
6 + 6 = 12 incorrect cases
```

No incorrect behavior was observed in the alpha tests.

An additional observation was made for NaN and infinity. These values were not silently accepted. They were eventually rejected by a downstream range check.

This indicates that some protection is present for non-finite values, while finite but invalid values such as negative or zero weights can still pass through and affect the result.

---

# 6. DeepProbLog

Two separate fuzzers were used for DeepProbLog.

The first was a deeper five-script pipeline that had been developed earlier in the project.

The second was a lighter literal-scanning fuzzer that was run during the current session.

The two approaches were independent, but the same main validation problem was identified.

---

# 6.1 DeepProbLog Deep Pipeline

This experiment was completed earlier in the project.

Five scripts were used.

### Step 1: Running the real test suite

The actual DeepProbLog test suite was executed.

The result was:

```text
65 passing tests
11 skipped tests
76 total tests
```

This baseline was established before fuzzing was performed.

### Step 2: Runtime instrumentation

Two real library points were instrumented:

- `Network.__call__`
- `GraphSemiring.value`

The values passing through these points during normal test execution were recorded.

This was useful because simply reading the test source code had previously underestimated the actual range of values reaching these parts of the system.

### Step 3: Fuzzing inputs

A total of **44 inputs** were selected across six different surfaces.

The inputs were checked against seven different types of rules.

These included:

- probability range
- probability normalization
- monotonicity
- equivalence of sound program rewrites
- consistency between engine settings
- gradient behavior
- agreement with ordinary ProbLog for degenerate cases

Because a single input could be checked against multiple rules, the 44 selected inputs resulted in:

```text
116 individual fuzz cases
```

This explains why the reported number is 116 rather than 44.

### Result

There were:

```text
24 incorrect cases out of 116
```

The comparison-based checks were clean:

```text
20 / 20 sound rewrite comparisons passed
22 / 22 engine-setting comparisons passed
```

The failures were therefore concentrated around the handling of neural-network outputs as probabilities rather than around the general reasoning process.

---

# 6.2 DeepProbLog Literal-Scanning Fuzzer

The second DeepProbLog fuzzer was run during the current session.

Algorithm A was applied to the actual DeepProbLog test files.

Two types of inputs were searched for:

1. Explicit probability literals in DeepProbLog programs.
2. Explicit numeric lists used as fake neural-network outputs.

The scan found:

- 6 explicit probability facts
- 2 fake network output vectors

Each probability fact was tested with seven mutations:

1. original value
2. negative
3. value above 1
4. zero
5. NaN
6. positive infinity
7. negative infinity

Therefore:

```text
6 facts × 7 mutations = 42 cases
```

The two fake network vectors were also tested with seven mutations each:

```text
2 vectors × 7 mutations = 14 cases
```

Therefore:

```text
42 + 14 = 56 total cases
```

This is why **56 fuzz cases** were produced.

### Result

There were:

```text
24 incorrect cases out of 56
```

The failures were divided as follows:

```text
12 program-text cases
12 network-output cases
```

---

# 7. DeepProbLog Network Output Validation

The network-output experiment produced an important result.

Invalid values such as:

- negative numbers
- values greater than 1
- all-zero vectors
- NaN
- infinity

were accepted without an exception.

This indicates that a neural network output can reach the probability-handling stage without being explicitly checked as a valid probability.

The program-text experiment behaved slightly differently.

NaN and infinity were rejected, but this rejection does not appear to be caused by a probability-range check.

Instead, `nan` and `inf` are not valid numeric literals in the DeepProbLog program syntax.

For example:

```text
-0.5
```

and:

```text
1.7
```

are syntactically valid numbers and can therefore pass through.

The rejection of NaN and infinity is consequently a parser-level effect rather than evidence of a proper probability validation rule.

---

# 8. Why the Two DeepProbLog Fuzzers Agree

The two DeepProbLog fuzzers used different methods.

The first one observed values during actual library execution.

The second one searched the library's test files for explicit numeric literals.

Despite this difference, the same central issue was found:

> Neural network outputs can be treated as probabilities without an explicit validation that they are valid probability values.

The agreement between two independent methods provides stronger evidence that the issue is related to the library's input validation rather than being caused by one particular fuzzing method.

---

# 9. Main Findings

Several observations were made across the three libraries.

### Passing unit tests does not guarantee complete input validation

The libraries' own tests were able to pass while invalid boundary values were still accepted in several cases.

For example, all 11 LTNtorch tests passed, but 115 problematic fuzz cases were still found.

### Finite invalid values were particularly important

Many of the failures involved ordinary numerical values such as:

- negative numbers
- zero
- values greater than 1

These values do not necessarily cause numerical exceptions, so explicit validation is needed if they are outside the documented range.

### Validation was not always consistent within the same library

In LTNtorch, the truth-value rule was correctly enforced in all 230 tested cases, while the `p >= 1` rule was not enforced consistently.

### Independent fuzzers produced matching results

This was particularly clear in DeepProbLog.

The runtime-based fuzzer and the literal-scanning fuzzer used different methods, but both identified the same probability-validation problem.

---

# 10. Final Results

| Library / Experiment | Fuzz Cases | Incorrect Cases |
|---|---:|---:|
| LTNtorch, literal discovery | 188 | 115 |
| LTNtorch, truth-value validation | 230 | 0 |
| **LTNtorch total** | **418** | **115** |
| **LNN** | **40** | **12** |
| **DeepProbLog, deep pipeline** | **116** | **24** |
| **DeepProbLog, literal scan** | **56** | **24** |

The fuzz-case counts were obtained directly from the experimental design:

- LTNtorch: `23 × 8 + 4 = 188`, followed by `23 × 10 = 230`
- LNN: `2 × (8 + 6 + 6) = 40`
- DeepProbLog deep pipeline: 44 selected inputs produced 116 rule-based checks
- DeepProbLog literal scan: `6 × 7 + 2 × 7 = 56`

The experiments therefore were not based on an arbitrary number of random test cases. Each case was connected to a real library target, a documented rule, and a specific mutation.

Overall, the experiments demonstrate that fuzzing based on real library contracts can expose input-validation problems that are not necessarily detected by conventional unit tests.# neAI_fuzz
