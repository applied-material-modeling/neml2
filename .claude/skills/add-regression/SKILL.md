---
name: add-regression
description: Add a NEML2 regression test under `tests/regression/<submodule>/<scenario>/` that pins a model's current behavior against a checked-in `gold/result.pt` reference. Use this skill whenever the user wants to scaffold a regression test, set up gold reference data, capture a model's current output for drift detection, or add a new scenario directory under `tests/regression/`. Trigger on phrases like "add a regression test for X", "scaffold a regression scenario", "create a TransientRegression for this model", "the regression suite is missing coverage for Y", or any setup involving `gold/result.pt`. For tests that compare against a *known* truth (analytical solution, benchmark data) rather than the model's own previous output, use `add-verification` instead — those are different in purpose.
---

# add-regression

Set up a regression-style test that pins a model's behavior to a checked-in reference. The `regression_tests` Catch2 binary auto-discovers any `.i` file under `tests/regression/<submodule>/` and runs whichever driver is named `regression` in it.

## When to use

A regression test answers: *did the model's output drift from what it produced last time?* It's a change-detection tool — useful for catching unintended numerical changes from refactors, dependency bumps, or unrelated edits. The reference (`gold/result.pt`) is the model's own previous output, not external truth.

If you want to check whether the model produces the *correct* answer (against an analytical solution, an established benchmark code, or even a strong physical-intuition result), use the `add-verification` skill instead. Verification and regression complement each other: verification confirms correctness, regression freezes that correct behavior so future changes don't silently break it.

## Directory layout

One directory per scenario:

```
tests/regression/<submodule>/<domain>/<scenario>/
  ├── model.i                Driver + model + EquationSystems + Solvers
  └── gold/
      └── result.pt          Reference output, produced by running model.i once
```

`model.i` declares two drivers in the `[Drivers]` block — a `TransientDriver` (which writes `result.pt`) and a `TransientRegression` (which reads `gold/result.pt` and compares):

```hit
[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]
```

Mirror an existing scenario in the same `<submodule>/` for the rest of the layout (`[Tensors]`, `[Models]`, `[EquationSystems]`, `[Solvers]`, the second `[Models]` block with `predictor` + `update` + final `model`). The recovery and viscoelasticity scenarios under `tests/regression/solid_mechanics/` are good templates depending on whether internal state is involved.

## Bootstrap the gold file

Before the test can run, `gold/result.pt` must exist. Generate it by running `model.i` through `neml2-run` once:

```bash
cd tests/regression/<submodule>/<domain>/<scenario>
neml2-run model.i driver        # produces result.pt; ignore the warning about TransientRegression
mv result.pt gold/result.pt
```

`neml2-run` will print `Warning: Object of type 'TransientRegression' is not registered in the NEML2 registry. This object will be ignored.` That's expected — `TransientRegression` is only linked into the `regression_tests` binary, not the standalone CLI. The warning means "I'm skipping that block," not "something failed."

## Running the test

```bash
./build/dev/tests/regression/regression_tests "<submodule-test-name>" -c "<domain>/<scenario>/model.i"
```

The test-case name is the string passed to `TEST_CASE(…)` in `tests/regression/<submodule>/regression_<submodule>.cxx` — typically a simple phrase like `"solid mechanics"`. Find it with `grep TEST_CASE tests/regression/<submodule>/*.cxx` if you're not sure.

The `-c` flag selects the Catch2 dynamic section, which is the path of the `.i` file relative to `tests/regression/<submodule>/`. **A wrong section pattern silently passes with zero assertions** — if you see `1 test case passed, 0 assertions`, the section pattern didn't match anything; double-check the relative path.

## Clean up before committing

Only `model.i` and `gold/result.pt` should be checked in. Stray `result.pt` files left in the scenario directory after `neml2-run` will be picked up by `git status` if you forget — quick to clean:

```bash
find tests/regression/<submodule>/<domain> -maxdepth 2 -name 'result.pt' -delete
```

## When you change the model and the regression starts failing

Two cases — distinguish them deliberately:

- **The change is intentional and the new behavior is correct.** Regenerate the gold: `cd <scenario-dir> && neml2-run model.i driver && mv result.pt gold/result.pt`. Commit the new `gold/result.pt` alongside the code change so a reviewer can correlate the two.
- **The change is unintentional or you're not sure.** Don't regenerate the gold yet — that's the test telling you something drifted. Run the corresponding verification test (or write one if there isn't one) to check whether the *new* behavior is still physically correct. Only after that's confirmed should you regenerate the gold.

## Choosing what scenarios to add

A regression test pins a *specific load history* through a *specific composition*. For broad coverage of one model, focus on scenarios that exercise distinct code paths rather than scaling up the same path:

- One scenario per qualitatively different composition (e.g., for a viscoelastic model: one with internal state + `ImplicitUpdate`, one as a pure forward evaluation).
- Scenarios that involve composition with auxiliary objects users will actually combine (e.g., elasticity composed with a viscous flow rule, mixed stress/strain control, anisothermal load).
- A scenario per regime change in the constitutive law (loading vs unloading, elastic-only vs plastic, etc.) when one regime would catch a bug the other wouldn't.

Avoid creating many scenarios that differ only in numerical parameters — they all break together for any real change and contribute little independent signal.
