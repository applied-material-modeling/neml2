---
name: add-verification
description: Add a NEML2 verification test under `tests/verification/<submodule>/<scenario>/` that compares model output against an external ground truth — analytical solution, established benchmark code (NEML1, WARP3D, etc.), or a closed-form physical-intuition result. Trigger on phrases like "add a verification test for X", "verify against the analytical solution", "set up a Verification driver", "compare to benchmark data". For tests that pin current behavior to a `gold/result.pt` of the model's own output, use `add-regression` instead.
---

# add-verification

Set up a verification scenario that compares NEML2's output to an
*external* truth (closed-form, NEML1, experiment-fitted CSV, …). The
`tests/verification/test_verification.py` driver walks every `.i`
under `tests/verification/`, locates its `[Drivers]` block of
`type = Verification`, and runs it; pass iff every declared
`variables` row matches its `references` row inside the declared
tolerances.

## Layout

```
tests/verification/<submodule>/<scenario>/
├── <scenario>.i        # [Drivers] (TransientDriver + Verification) + [Models] + [Tensors] (...CSV...)
└── reference.csv       # one column per reference variable; column names match the `references` list
```

`<submodule>` matches the source directory (`solid_mechanics`,
`chemical_reactions`, …); `<scenario>` is a short descriptive name
(`chaboche`, `mixed_control`, `pyrolysis`, …).

## `<scenario>.i` skeleton

```ini
[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'reference.csv'
    variable = 'time'
  []
  [strain_ref]
    type = CSVSR2
    csv_file = 'reference.csv'
    variable = 'strain'
  []
  [stress_ref]
    type = CSVSR2
    csv_file = 'reference.csv'
    variable = 'stress'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strain_ref'
    save_as = 'result.pt'
  []
  [verification]
    type = Verification
    driver = 'driver'
    variables = 'stress'             # NEML2-side variable name(s) to check
    references = 'stress_ref'         # parallel list of [Tensors] entry names supplying ground truth
    # rtol = 1e-6   # override per scenario
    # atol = 1e-8
  []
[]

[Models]
  ...
[]
```

- `variables` / `references` are parallel lists; each entry pairs one
  NEML2 output name to one `[Tensors]` reference entry.
- `CSVScalar` / `CSVSR2` / etc. pull a named column out of
  `reference.csv` (or whichever file you point at). Use these for any
  external truth that fits in CSV — analytical solutions evaluated
  in Python and dumped, NEML1 stress histories converted to CSV,
  experimental data, …

## Workflow

1. **Produce / convert the reference data** to CSV. For analytical
   truth, a small Python script that writes the columns is usually
   the path of least resistance. For external code output (NEML1
   `.vtest`, etc.), convert via `scripts/vtest_to_csv.py` if a
   conversion path exists.
2. **Author `<scenario>.i`** with the `[Tensors]` CSV readers, the
   `TransientDriver`, and the `Verification` block.
3. **Run the verification** to confirm it loads + matches:

   ```bash
   pytest -v tests/verification/test_verification.py -k <scenario>
   ```

   The parametrized ID is the `.i` path relative to
   `tests/verification/`, so `-k chaboche` matches
   `solid_mechanics/chaboche/chaboche.i`.

## Hard rules

- **Reference is external truth**, not NEML2's own output. If the
  ground truth is "what NEML2 produced last time", use
  `add-regression` instead.
- **CSV is the universal format.** Any reference source — analytical,
  external code, experimental — should land in a CSV column. This
  keeps the pipeline tool-agnostic and the diff reviewable.
- **`variables` / `references` are parallel lists.** Order matters.
- **One scenario per directory.** The parametrizer keys on the `.i`
  path.
- **Tolerances are scenario-specific.** External truth carries its
  own noise; pick `rtol` / `atol` that reflect the physical
  comparison being made, not the framework defaults.

## When to use `add-regression` instead

If the reference is "what the model produced previously" — pinning
output to detect drift — use the `add-regression` skill instead.

## See also

- `tests/verification/test_verification.py` — the parametrized driver.
- Existing scenarios under `tests/verification/<submodule>/` for
  worked CSV-reader patterns.
- `scripts/vtest_to_csv.py` — converter for NEML1-style `.vtest`
  reference files, where available.
- `add-regression` for output-drift tests.
