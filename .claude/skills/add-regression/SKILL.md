---
name: add-regression
description: Add a NEML2 regression test under `tests/regression/<submodule>/<scenario>/` that pins a model's current behavior against a checked-in `gold/result.pt` reference. Trigger on phrases like "add a regression test for X", "scaffold a regression scenario", "create a TransientRegression for this model", "the regression suite is missing coverage for Y". For tests that compare against a *known* truth (analytical solution, benchmark data), use `add-verification` instead.
---

# add-regression

Set up a regression scenario that pins a model's behavior to a
checked-in `gold/result.pt`. The `tests/regression/test_regression.py`
driver walks every `.i` under `tests/regression/` and runs the
`TransientRegression` block inside; one `.i` per scenario, no Python
wrapper to write.

## Layout

```
tests/regression/<submodule>/<scenario>/
├── model.i           # [Drivers] (TransientDriver + TransientRegression) + [Models] + ...
└── gold/
    └── result.pt     # checked-in reference, produced by neml2-run on first author
```

## `model.i` skeleton

```ini
[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
    # rtol = 1e-10  # override for noisy scenarios; default is 1e-10
    # atol = 1e-10
  []
[]

[Tensors]
  ...
[]

[Models]
  ...
[]
```

The two driver sub-blocks are load-bearing:

- `driver` (a `TransientDriver`) runs the actual time-stepping and
  holds the result in memory (the committed `.i` sets no `save_as`).
- `regression` (a `TransientRegression`) compares that result to
  `gold/result.pt` under the declared tolerances.

The driver block can be named anything — `test_regression.py` finds
it by `type = TransientRegression`, not by name.

## Workflow

1. **Author `model.i`** with both driver blocks. Verify it parses
   and loads:

   ```bash
   neml2-inspect tests/regression/<submodule>/<scenario>/model.i model
   ```

2. **Produce the gold reference** — a deliberate, manual step. The
   committed `.i` carries no `save_as`, so temporarily add
   `save_as = 'result.pt'` to the `[driver]` block, run it, move the
   output into `gold/`, then drop the `save_as` line again:

   ```bash
   cd tests/regression/<submodule>/<scenario>
   mkdir -p gold
   # 1. add `save_as = 'result.pt'` to the [driver] block, then:
   neml2-run model.i driver
   mv result.pt gold/result.pt
   # 2. remove the `save_as` line from [driver] again
   ```

3. **Run the regression** to confirm the gold round-trips:

   ```bash
   pytest -v tests/regression/test_regression.py -k <scenario>
   ```

   The parametrized ID is the `.i` file's path relative to
   `tests/regression/`, so `-k isokinharden` matches
   `solid_mechanics/rate_independent_plasticity/isokinharden/model.i`.

## Hard rules

- **Gold is a `result.pt`**, not a hand-written reference. Produce it
  from `neml2-run` once and check it in. Re-run + re-check-in when
  the model output legitimately changes; never hand-edit the file.
- **Tolerances default to `rtol = atol = 1e-10`** on
  `TransientRegression`. Tighten or loosen per scenario only when
  numerical noise demands it.
- **One scenario per directory.** The test parametrizer keys on the
  `.i` path; two `.i` files in the same directory clash.
- **No `save_as` in the committed `.i`.** The regression compares the
  driver's in-memory result to `gold/result.pt`, so a `save_as` write
  would be dead weight on every run — add it only transiently when
  regenerating the gold (step 2).

## When to use `add-verification` instead

If the reference comes from an analytical solution, an external code
(NEML1, WARP3D, …), or a closed-form benchmark — anything *not*
NEML2's own output — use the `add-verification` skill. Verification
catches algorithm drift; regression catches output drift.

## See also

- `tests/regression/test_regression.py` — the parametrized driver.
- Existing scenarios under `tests/regression/<submodule>/` for working
  examples of `[Tensors]` setups, mixed-control wiring, etc.
- `add-verification` for external-truth tests.
- `add-model` for the upstream model authoring contract.
