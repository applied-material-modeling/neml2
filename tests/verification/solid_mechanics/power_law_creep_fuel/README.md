# Verification: NEML2 power-law creep vs the MOOSE UPuZr fuel simulation

The NEML2 power-law creep model (`PowerLawCreepFlowRate`) is verified against the
original MOOSE metallic-fuel creep simulation by running the full 1.79-year coupled
fuel problem **both ways** and comparing the outputs. The ground truth is the MOOSE
simulation itself -- not a synthetic or closed-form reference.

> This is a full-simulation (finite-element BVP) cross-check, not a NEML2 material-point
> `Verification`-driver scenario, so it has no `model.i` and is not collected by the
> `tests/verification` pytest sweep. The input decks carry a `.txt` suffix for the same
> reason (the sweep globs `*.i`); they are ordinary HIT inputs -- run them with `-i <file>`
> as shown below, or strip the suffix.

## Contents
- `moose_x447_fuel.csv` -- MOOSE reference output (original deck).
- `neml2_fuel.csv`      -- NEML2 output (creep stress supplied by the NEML2 model).
- `compare.py`          -- regenerates the numbers below from the two CSVs.
- `inputs/`             -- everything needed to re-run both simulations:
  - `x447_dp11_fuel_moose.i.txt`  -- the original MOOSE deck (reference).
  - `fuel_old_coupled.i.txt`      -- the same problem, stress supplied by NEML2.
  - `fuel_creep_neml2.i.txt`      -- the NEML2 power-law creep model (loaded by the deck above).
  - `fuel_mesh.e`, `data/`        -- shared mesh + loading histories (both decks use these).
  - `moose_coupling/`             -- a **custom** MOOSE material we authored
    (`ComputeNEML2StressOldSystem.{h,C}`) that feeds the NEML2 stress into MOOSE's legacy
    mechanics system. It is not part of MOOSE or NEML2 -- kept here only as reference so
    the coupled run is reproducible (see `moose_coupling/README.md`). Compile it into a
    local MOOSE + NEML2-v3 build to run `fuel_old_coupled.i.txt`.

## Re-running

MOOSE reference:
```
cd inputs && <moose-app> -i x447_dp11_fuel_moose.i.txt
```

NEML2-coupled (MOOSE built against NEML2 v3 with the bridge material compiled in):
```
cd inputs && <app> -i fuel_old_coupled.i.txt \
    Executioner/nl_rel_tol=1e-6 Executioner/nl_abs_tol=1e-8 Executioner/nl_max_its=50 \
    Executioner/line_search=bt Executioner/residual_and_jacobian_together=false
```
Both use nl_rel 1e-6 / nl_abs 1e-8; the adaptive stepper yields identical time grids
(15522 steps), so the comparison is exact row-by-row.

## Result -- average von Mises stress, full 1.79 yr (rel-to-peak)

| metric | value |
| --- | --- |
| median | 1.15e-04 |
| 90th percentile | 1.67e-03 |
| 99th percentile | 9.95e-03 |
| max | 3.34e-02 |

Temperature rel error 2.3e-06; burnup identical. NEML2 reproduces the MOOSE creep
response to a **median of 0.011%** over the full fuel life (99% of steps < 1%); the
3.3% maximum is at a few isolated power-ramp transients. Run `python compare.py` to
regenerate.
