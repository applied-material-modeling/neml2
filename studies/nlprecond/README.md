# Nonlinear-preconditioning testbed

Reproduces and quantifies the **cold-start Newton pathology** in NEML2's
viscoplasticity and crystal-plasticity models — the behaviour that predictors
and line search currently paper over.

This directory contains no preconditioner. It exists so that "did nonlinear
preconditioning help?" becomes a one-command comparison against a committed
baseline.

```bash
python -m studies.nlprecond.ablate --smoke     # wiring check, ~1 min
python -m studies.nlprecond.ablate             # full sweep (136 points, ~2 min)
python -m studies.nlprecond.report             # the three headline tables
```

Runs are short by construction: each covers **6 time steps**, not the parent
scenario's 99. The history is *truncated*, not stretched, so every increment is
byte-identical to the parent's — a 6-step run reproduces the parent's step-1
behaviour exactly, at 1/16 the cost. Pass `--nsteps 99` for a full-history run.

## The diagnosis

### 1. The first time step costs an order of magnitude more than any other

Driving the committed regression scenarios under `NEML2_LOGS=newton=debug`:

| scenario | step-1 iterations | median over later steps |
| --- | --- | --- |
| `viscoplasticity/isoharden` (Perzyna n=2, U=7) | 15 | 1 |
| `viscoplasticity/chaboche` (n=2, U=19) | 15 | 3 |
| `crystal_plasticity/single_crystal_coupled` (power law n=8, U=10) | 16 | 3 |

### 2. A predictor cannot help at step 1 — by construction

Removing the predictor leaves the step-1 residual trace *byte-identical* and
makes every later step as expensive as the first. The per-step iteration counts
say it plainly (`vp_isoharden`, no line search):

| arm | iterations per step | total |
| --- | --- | --- |
| `pred-ls` | `15-4-3-3-3-2` | 30 |
| `nopred-ls` | `15-15-15-15-15-15` | 90 |

Over the parent's full 99-step history (`--nsteps 99`) the same two arms cost
**129** and **1485** iterations — the gap widens with every step, because the
predictor keeps paying off and the cold start never stops being a cold start.

With no predictor the initial guess is the unknown's own model input, which
`TransientDriver` zero-fills. At step 1 the constant-extrapolation predictor
extrapolates *from* the zero initial state, so it produces that same zero seed.
There is nothing to warm-start from.

So the "bad first step" is not really about the first step. It is a **cold-start
penalty**, and step 1 is simply the one step where a predictor cannot hide it.

### 3. Root cause: a cold start lands in the far field of the power law

The step-1 trace is a large overshoot followed by a long constant-rate plateau:

```
vp_chaboche step 1:  9.52e1 -> 6.47e7 -> 1.62e7 -> 4.04e6 -> 1.01e6 -> ... -> 3.20e-2
                            ^ overshoot        ^^^^^ exactly 4x per iteration ^^^^^
```

For a residual dominated by a power-law flow rate $\dot\gamma \sim (\sigma/\sigma_0)^n$,
an undamped Newton step contracts the residual by a *constant* factor
$(1-1/n)^n$ per iteration — linear, not quadratic convergence. Sweeping the
rate-sensitivity exponent confirms it to four digits:

| n | measured contraction | $(1-1/n)^{-n}$ |
| --- | --- | --- |
| 2 | 4.004 | 4.000 |
| 4 | 3.162 | 3.160 |
| 8 | 2.912 | 2.910 |
| 12 | 2.842 | 2.841 |
| 20 | 2.791 | 2.790 |

Newton spends 13–15 iterations crawling down this plateau before it re-enters
the quadratic basin. **This is the unbalanced nonlinearity that nonlinear
preconditioning targets**: one strongly-nonlinear component dictates the step
for the whole system.

### 4. Line search does not fix it — it trades the overshoot for a stall

On `cp_coupled`, line search on vs off gives the *same* 16 iterations. Without
it, |R| spikes by ~90 orders of magnitude and the solve diverges outright; with
it, the residual creeps down a plateau at a measured contraction of **1.056**
per iteration — a near-total stall. Both arms pay for the same underlying
imbalance.

## What a nonlinear preconditioner has to beat

Two distinct targets, and the second is the more interesting one:

1. **Iteration count** — collapse the 13–15 plateau iterations at a cold start.
2. **Basin size** — the increment at which each arm stops converging at all.
   From the committed baseline (total iterations, `DIV` = diverged):

   | case | arm | x1 | x2 | x5 | x10 | x20 |
   | --- | --- | --- | --- | --- | --- | --- |
   | `vp_isoharden` | `pred+ls` | 26 | 26 | 26 | 24 | 25 |
   | `vp_chaboche` | `pred+ls` | 27 | 28 | 35 | 56 | 83 |
   | `cp_decoupled` | `pred+ls` | 52 | 49 | 61 | 78 | DIV |
   | `cp_coupled` | `pred+ls` | 34 | DIV | DIV | DIV | DIV |
   | `cp_coupled` | `pred-ls` | DIV | DIV | DIV | DIV | DIV |

   `cp_coupled` cannot take even *twice* its own increment with both
   globalizations on, and cannot take its own increment at all without line
   search. Pushing that boundary outward would remove the dependence on
   globalization rather than just making it cheaper — and, unlike an iteration
   count, it is not something a better predictor can buy.

   Note the viscoplastic contrast: `vp_isoharden` is flat at ~25 iterations from
   x1 to x20. Its whole cost *is* the cold start; the increment barely matters.

## Candidate fix: state the flow rule in inverted form

`vp_isoharden_inverted` is `vp_isoharden` with the Perzyna rule written as an
implicit residual in *inverted* form and `flow_rate` carried as an unknown:

```
  eta * gdot^(1/n) - <f> = 0        instead of        gdot = (<f>/eta)^n
```

Same root — `check_equivalence` compares every output series at every step and
finds a worst relative difference of **2e-11** (n=2) / **5e-11** (n=8). What
changes is only what Newton sees: substituting the explicit map makes every
residual degree-n in the stress; carrying `gdot` as an unknown relocates that
to a single 1/n power and leaves the other residuals affine in it.

The payoff tracks stiffness, which is the whole point — step-1 iterations,
`vp_isoharden` `pred-ls`:

| Δt multiple | rate form | inverted |
| --- | --- | --- |
| x1 | 15 | 9 |
| x2 | 16 | 8 |
| x5 | 17 | 7 |
| x10 | 18 | 7 |
| x20 | **19** | **7** |

The rate form degrades as the step stiffens; the inverted form improves and
then flattens. Total iterations over the run with no predictor go 90 → 114 for
the rate form and 49 → 42 for the inverted one — 2.7x fewer at x20.

Two secondary effects worth noting:

* **Line search becomes irrelevant.** `pred+ls` and `pred-ls` give byte-identical
  counts for the inverted case (24 and 24; 49 and 49). There is no overshoot
  left for it to damp.
* **The gain shrinks with n, opposite to the scalar model.** Step-1 counts are
  15 → 9 at n=2 but 17 → 17 at n=20; totals still improve 40-46% throughout.
  The coupled system keeps nonlinearity (flow direction, normality) that the
  scalar radial-return model does not, so the flow rule is not the only
  degree-n term. Worth chasing if more is wanted.

Regularization knobs live in `harness.py`: `GDOT_CUTOFF` (where the fractional
power hands over to its tangent line) and `GDOT_SEED` (the strictly-positive
initial condition, since at step 1 there is nothing to extrapolate from).

### On crystal plasticity it buys robustness, not iterations

`cp_coupled_inverted` applies the same idea to `PowerLawSlipRule`
(`PowerLawSlipRuleResidual`). The law is odd and has no yield threshold, so the
regularization below the cutoff is the unique odd cubic matching the power's
value and slope -- not the tangent line used for Perzyna.

The trade is the **opposite** of the viscoplastic case:

| | `cp_coupled` | `cp_coupled_inverted` |
| --- | --- | --- |
| dt x1 | 16 step-1 / 34 total | 36 / 100 |
| dt x2 | **diverges** | 108 |
| dt x20 | **diverges** | 148 |

Per step it is roughly 3x more expensive. But the baseline cannot take even
*twice* its own increment, while the inverted form runs to x20. Compared at
equal load coverage (120 parent increments, `pred+ls`):

| formulation | steps x dt | total iterations |
| --- | --- | --- |
| `cp_coupled` | 120 x1 | 383 |
| `cp_coupled` | 60 x2 | diverges |
| `cp_coupled_inverted` | 120 x1 | 1670 |
| `cp_coupled_inverted` | 6 x20 | **128** |

So it is 3x cheaper for the same physics *if* the time-stepper exploits the
larger step, and 4.4x more expensive if it does not. For a driver that cuts
back on failure, the basin matters more than the per-step count.

Both forms need line search here; neither converges without it. An earlier
sweep of this concluded the reformulation failed outright for CP -- that was an
artifact of testing at `ls_iters=1`, which the *baseline* also fails at.

Accuracy is governed by the cutoff: 2.9e-10 vs the baseline at 1e-20 and 1e-10,
degrading to 1.7e-3 at 1e-4.

**The 12 extra unknowns are nearly free.** Each slip rate's residual involves
only its own rate, so the `(slip_rates, slip_rates)` Jacobian block is *exactly*
diagonal -- measured off-diagonal is identically zero. The case therefore splits
the equation system into two groups and condenses that block out with
`SchurComplement`, taking the linear solve from a dense 22x22 to per-site
scalar solves plus the ordinary 10x10 (22^3 = 10648 against 10^3 + 12 = 1012,
about 10.5x). Newton iterates are unchanged -- Schur is an exact solve -- so
this is purely a per-iteration cost win.

Two ordering constraints, both easy to get wrong:

* The BLOCK group must come **first** and be the Schur *primary*
  (`structure = 'block dense'`, `residual_primary_group = 0`). The coupling is
  an arrowhead -- `A(0,1)` is per-site `(12, 10, 1)`, not a plain rectangular
  block -- and that is the orientation `SchurComplement` supports (see
  `test_per_instance_matvec_arrowhead_inverts_schur`, and `taylor`, which uses
  the same convention). Declaring `'dense block'` with the dense group primary
  assembles fine and then fails to converge.
* Line search is required, in both formulations. `check_equivalence` enables it
  by default for exactly this reason.

### The log form: fixes the rate, not the basin

`cp_coupled_log` carries `u = log(|gdot|/gamma0)` as the unknown
(`PowerLawSlipRuleLogResidual` + `SlipRateFromLog`). Taking the log of the flow
law's magnitude relation makes the residual **exactly affine in the unknown**
(`dr/du = 1` identically) and collapses the ~70-decade slip-rate range onto a
few hundred units. Agreement with the baseline: 2.7e-09.

It is the only form that beats the baseline on **convergence rate**, and it
barely moves the **basin** -- the mirror image of the inverted form:

| | step-1 iters | per-step | basin (largest dt) |
| --- | --- | --- | --- |
| `cp_coupled` | 16 | `16-5-4-3-3-3` | x1 |
| `cp_coupled_inverted` | 36 | `36-13-15-15-13-8` | **x20** |
| `cp_coupled_log` | **12** | `12-6-5-13-4-5` | x2 |

So the two reformulations are complementary: log space fixes the cold-start
iteration count, the 1/n-power form fixes robustness.

**But the log form is fragile, and does not win in practice.** Over 120 parent
increments it diverges partway in both configurations it can attempt, while the
inverted form completes by taking x20 steps:

| | steps x dt | total |
| --- | --- | --- |
| `cp_coupled` | 120 x1 | 390 |
| `cp_coupled_log` | 120 x1 | **diverges** |
| `cp_coupled_log` | 60 x2 | **diverges** |
| `cp_coupled_inverted` | 6 x20 | **148** |

Two further costs, both recorded rather than tuned away:

* It needs **deeper line search** -- `linesearch_iters = 20` against everything
  else's 5, and it fails outright at 5. Undamped it fails completely (99+
  iterations on a scalar model); the residual is nearly flat in `u` far from the
  root, with a barrier where the slip absorbs the whole trial shear.
* It is **sensitive to the initial guess and not to the tau floor**. Sweeping
  both at nbatch=16: every `taufloor` from 1e-12 to 1e-2 behaves identically,
  while `u_seed` of -60 or -30 leaves 3-4 batch members unconverged and -10 or
  -3 converges all of them. A very dormant seed sits ~54 units from a root near
  `u = +24`, which 20 halvings cannot walk back for some orientations.

### Where the CP iterations actually go

Instrumenting one step-1 solve of `cp_coupled_inverted` (per-group and
per-variable residuals, the Newton step, and the accepted line-search alpha)
splits its 36 iterations into two distinct phases.

**Phase 1, ~12 iterations: climbing out of the seed.** The slip-rate unknown is
seeded at zero, 20 decades below its converged value of ~0.07. That sits inside
the regularization cutoff, where the odd cubic's slope is ~1e17, so each Newton
step multiplies the slip rate by only ~100:

```
it 0: 0.0e+00   it 4: 6.7e-12   it 8: 5.6e-06   it 12: 1.8e-02
it 2: 3.0e-16   it 6: 1.5e-08   it 10: 4.8e-04  it 13: 7.5e-02  <- converged
```

Throughout, alpha is pinned at 1/32 -- the floor after five halvings -- and the
residual *grows*, 118 to 347. Seeding helps but less than that trace suggests:
36 iterations at zero, 31 at 1e-12, 28 at 1e-2, and divergence at 1e-1. The seed
is now a knob (`gdot_seed`) rather than hardcoded zeros. The log form's `u_seed`
sensitivity is the same effect in log coordinates.

**Phase 2, ~25 iterations: damped decay.** alpha oscillates in [0.125, 0.5] and
the residual contracts only ~1.2x per iteration. The line search rejects the
full Newton step nearly every iteration. **This is the dominant cost and it is
not yet explained.**

Ruled out along the way: a units mismatch between the residual groups. At
iteration 0 the slip-rate residual (58, stress units) dwarfs the elastic-strain
residual (9e-3, strain units), which looks like it should mislead the
line-search merit function. Swapping that merit function for a per-group
*relative* norm makes things worse, not better (37 -> 60 iterations for
`cp_coupled_inverted`, 39 -> 200 for `cp_coupled_log`), so the mismatch is not
the mechanism.

A caveat on the affine-invariance argument used earlier in this document: it
rules out rescaling as a cure for the *undamped* Newton path, which is what the
viscoplastic diagnosis needed. It does **not** extend to a line-searched solve,
where the merit function is scale-dependent. Scaling is a legitimate lever
there -- it just is not the one that helps here.

### The incremental potential: right diagnosis, unusable as a merit

The inverted residual is not merely better conditioned -- it is the *gradient*
of an incremental potential (Ortiz-Stainier). For the power-law slip rule the
primal dissipation potential is

    phi(gdot) = n/(n+1) * gamma0 * tauc * |gdot/gamma0|^((n+1)/n)

and `dphi/dgdot` reproduces `PowerLawSlipRuleResidual` to 1e-12 for both signs
across five decades, with `phi'' > 0` everywhere. So

    I(gdot) = Psi(Ee_trial - dt*sum_i gdot_i M_i) + dt * sum_i phi(gdot_i)

satisfies `dI/dgdot_i = dt * r_i`: elastic energy plus plastic dissipation. The
rate form is the gradient of nothing, which is an independent reason to prefer
the inverted one.

Evaluating I along a step-1 Newton path (using `Ee_slaved = Ee - r_Ee`, which
avoids reconstructing trial strains and rotated Schmid tensors) **confirms that
max|R| is a bad merit function** -- through the whole stalled phase:

| iters | max\|R\| | I |
| --- | --- | --- |
| 0-11 | **grows** 116 -> 253 | **decreases monotonically** 6.122 -> 1.207 |
| 12-29 | falls 82 -> 0.08 | drifts *up* ~1e-3/step |

Those first 12 iterations are exactly the ones the line search wastes at
alpha = 1/32. The Newton steps are good descent steps on the potential; the
residual norm rejects them.

**But an incomplete potential cannot be used as the merit.** The converged root
sits at I = 1.1822, above the path minimum of 1.1652 -- the true solution is not
I's minimizer, off by ~1.5%. A line search demanding I-descent therefore rejects
every step once phase 2 begins and stalls completely: 200+ iterations at
ls = 5, 10 and 20, against 32 for max\|R\|.

The gap is the *implicit hardening coupling*. Voce itself is variational --
`h(alpha) = h_s(1 - exp(-theta0*alpha/h_s))` integrates to a genuine stored
energy `Psi_h(alpha)`, and its derivative is exactly NEML2's rate form. What
breaks stationarity is evaluating `tauc` inside `phi` at the *new* alpha, which
adds a `dphi/dtauc * dtauc/dgdot` term the residual omits. Adding `Psi_h` to the
merit does not repair this -- it makes phase 2 worse, as expected if the missing
piece is the coupling rather than an energy.

Two routes follow, neither tried yet: freeze `tauc` at its step-n value inside
`phi` (explicit hardening), which restores exact variational structure at the
cost of an O(dt) physics change; or accept a non-monotone / hybrid merit that
uses I-descent where available.

Incidental but useful: **deeper backtracking on a bad merit is harmful.** With
max|R| the step-1 count goes 32 -> 82 -> 132 as `ls_iters` goes 5 -> 10 -> 20.
More halvings just lets the solver settle for a tiny alpha that marginally
improves the wrong quantity.

### Lagging tauc makes it exactly variational -- and it still does not help

`cp_coupled_variational` and `cp_decoupled_variational` evaluate the slip
strength from the previous step's hardening. With `tauc` frozen during the
solve the update is exactly variational; in the decoupled case, where the
orientation is already lagged, sub-system #1's potential is **complete** --
nothing else is an unknown, so `dI = 0` does imply `R = 0`.

The structure is confirmed: I now decreases **monotonically** along the whole
path, in both cases. Theory holds.

But a monotone-descent line search on I converges far worse than on max|R|:

| merit | ls=5 | ls=10 | ls=20 |
| --- | --- | --- | --- |
| `max\|R\|` | **34** | 83 | 127 |
| `I` | 168 | 218 | 300+ |

The path shows why: I falls 6.12 -> 1.00 in five iterations as elastic energy
converts to plastic work, then changes by ~1e-4 per step while max|R| still
oscillates around 45. I is dominated by a few easy directions and nearly flat
in the stiff ones, so *simple decrease* is too weak a criterion -- a full Newton
step that lands at |R| = 1168 is accepted because I dropped by 1e-3.

**Armijo does not rescue it, and the reason kills the trust-region idea too.**
Using the true directional derivative `grad(I).p = dt * r_gamma . dgamma` with
`c1 = 1e-4`, on the fully variational sub-system:

| merit | ls=5 | ls=10 | ls=20 | ls=40 |
| --- | --- | --- | --- | --- |
| `max\|R\|` | **34** | 83 | 127 | 127 |
| I, simple decrease | 168 | 218 | 300+ | 300+ |
| I, Armijo | 166 | 179 | 187 | 300+ |

The ratio of actual to predicted decrease, `rho`, explains it: rho sits between
0.1 and 1.0 for the whole path. **The quadratic model is accurate.** The step is
not overshooting the model's validity, so neither Armijo nor a trust region has
anything to correct -- a trust region would keep its radius large and reduce to
plain Newton.

What is actually wrong is that **I is nearly flat where the iterations are
spent**. At iteration 0 the directional derivative is `grad(I).p = -5.7e-08`
against `I = 6.12`: the cold start, with slip rates seeded near zero, is
essentially a stationary point of the potential. I carries almost no signal over
the region the solver traverses, while max|R| there moves by orders of
magnitude. Minimizing a flat function is intrinsically slow, whatever the
line-search rule.

So the potential-based merit is a dead end for this problem, and so is the
trust region built on it. The variational structure is real -- and worth keeping
as the reason to prefer the inverted residual -- but it does not make a useful
merit function here.

There is also a physics cost. Lagging `tauc` is *not* a small perturbation at
these step sizes: `check_equivalence` puts `cp_coupled_variational` 16% away
from the baseline at step 1. With `h_n = 0` the first step sees `tauc = 50`
against a converged ~59, and `n = 8` turns that into a 3.3x slip-rate error.
Shipping variational crystal plasticity would need either much smaller steps or
a variationally consistent implicit hardening treatment.

### The step-1 cost is the initial guess, essentially all of it

Seeding sub-system #1's slip rates directly (bypassing the predictor) on
`cp_decoupled_variational`:

| seed | max\|gdot\|_0 | step-1 iters |
| --- | --- | --- |
| IC, 1e-12 uniform | 1e-12 | **34** |
| converged answer (oracle) | 8.2e-02 | **1** |
| 0.5x converged | 4.1e-02 | 5 |
| 2x converged | 1.6e-01 | 5 |
| 0.1x converged | 8.2e-03 | 11 |

One iteration from a good guess. Order-of-magnitude accuracy is enough -- a
factor of 10 either way still gives 5-11.

**This retires the earlier phase-1 / phase-2 decomposition.** Those 36 iterations
were described above as ~12 climbing out of the seed plus ~25 of unexplained
damped decay. With a good seed there is no phase 2 either: both were
consequences of starting at a near-stationary point, not two separate phenomena.

**A uniform IC does not substitute for a real predictor.** The IC can only carry
one value for all 12 systems, and the solution has a few systems near 0.08 with
the rest near zero:

| uniform `gdot_seed` | 1e-12 | 1e-3 | 1e-2 | 4e-2 | 8e-2 | 2e-1 |
| --- | --- | --- | --- | --- | --- | --- |
| step-1 iters | 29 | 28 | **24** | 29 | diverges | 43 |

24 at best against the oracle's 1. So the per-system predictor is worth
building, and the IC workaround is not a way around it.

**What blocks it.** The natural implementation adds no new leaf: `predictor1` is
a `ComposedModel`, so chaining the *explicit* `PowerLawSlipRule` onto the elastic
strain `CrystalPlasticityStrainPredictor` already predicts gives each system its
own rate. That composes and loads, but the predicted `slip_rates` reaches the
implicit system without its per-slip sub-batch axis and `SumSlipRates` fails with
`IndexError: dim -1 out of range`.

The root cause is a missing *declaration*: NEML2 has no way to know a
sub-batched unknown's shape. The mechanism exists -- `[Settings]
example_batch_shape` supports per-variable specs, sub-batch (`'(2; 100)'`) and
even sub-batch labels -- but it is read only by `cli/aoti_export.py` and
`cli/aoti_compile.py`. The eager path never sees it, which is also why
`TransientDriver` zero-fills sub-batched inputs at base shape
(`supports_nopred`) and why every case here hand-shapes an IC. Closing that
parity gap is the prerequisite for the slip-rate predictor.

## Cases

Each case is a self-contained copy of a regression scenario, edited only to
expose knobs. The parents are untouched — they are pinned against a
`gold/result.pt` and must stay that way.

| case | parent (under `tests/regression/solid_mechanics/`) | unknowns | why |
| --- | --- | --- | --- |
| `vp_isoharden` | `viscoplasticity/isoharden` | 7 | minimal reproducer; fastest grid point |
| `vp_chaboche` | `viscoplasticity/chaboche` | 19 | nonlinearity spread across backstress groups |
| `cp_coupled` | `crystal_plasticity/single_crystal_coupled` | 10 | canonical CP; one fully-coupled group |
| `cp_decoupled` | `crystal_plasticity/single_crystal_decoupled` | 7 + 3 | two sequentially-solved sub-systems |
| `vp_isoharden_inverted` | `studies/nlprecond/cases/vp_isoharden` | 8 | the reformulation candidate (see above) |
| `cp_coupled_inverted` | `studies/nlprecond/cases/cp_coupled` | 22 | the 1/n-power form on CP -- big basin lift |
| `cp_coupled_log` | `studies/nlprecond/cases/cp_coupled` | 22 | the log form on CP -- best rate, fragile |

## Ablation arms and knobs

Arms are `{pred,nopred}` x `{+ls,-ls}`:

* `pred` / `nopred` — the `predictor = '...'` wiring, stripped by the harness.
* `+ls` / `-ls` — `max_linesearch_iterations` of 5 or **1**. One is not a
  degenerate line search: the C++ loop takes the full-step branch at
  `ls_max_iters <= 1`, making it exactly plain `Newton`. One integer spans both
  arms, so no solver block has to be swapped.

HIT `${...}` knobs, supplied via `load_input(pre=[...])`:

| knob | meaning |
| --- | --- |
| `nbatch` | dynamic batch members |
| `npoint` | time points; the driver takes `npoint-1` steps |
| `tfrac` | fraction of the parent's full load history to cover |
| `flow_n` | Perzyna `exponent` / `PowerLawSlipRule` `n` |
| `ls_iters` | `max_linesearch_iterations` (1 = no line search) |
| `max_its` | Newton iteration cap |

**Step count and increment size are independent**, which is the point of
`tfrac`. The harness sets `tfrac = dt_scale * nsteps / 99`, so:

* `--nsteps` changes only *how many* steps run. Each one keeps the parent's
  increment, so shortening a run never changes the physics it measures.
* `dt_scale` changes only *how big* each step is, as a multiple of the parent's
  increment. This is the stress knob that walks the solver off the edge of its
  convergence basin.

Stretching a short history over the full load range instead — the obvious first
implementation — silently conflates the two, so cheap runs would be measuring a
different problem than the one being reported on.

## Metrics

| metric | definition |
| --- | --- |
| `iters_step1` | Newton iterations at driver step 1 (summed over sub-systems) |
| `iters_by_step` | per-step counts, e.g. `15-4-3-3-3-2` — the cold-start signature |
| `median_iters_rest` | median iterations over steps 2..N |
| `imbalance` | `iters_step1 / median_iters_rest` |
| `total_iters` | Newton iterations over the whole run |
| `step1_overshoot` | `peak|R| / |R0|` at step 1; > 1 means the first step made things worse |
| `step1_contraction` | median residual ratio across the linear plateau, or NaN if there is no plateau |
| `theory_contraction` | `(1-1/n)^-n`; comparable only against `-ls` arms |

Per-solve metrics come from the step-1 solve that does the *most* work, not an
average: a step may hold several solves over unrelated sub-systems (e.g.
`cp_decoupled` solves elastic-strain+hardening, then orientation), and averaging
their conditioning is meaningless.

`imbalance` is a **lower bound** at small `--nsteps`. A step keeps getting
cheaper for the first ~10 steps as the state settles, so a short run's
`median_iters_rest` sits above the true steady state (`vp_isoharden` reads 5 at
6 steps and 15 at 99). `iters_by_step` shows the shape without this caveat,
which is why the report leads with it.

## Route: eager, CPU, float64

Iteration counts and residual histories are route-independent — the Newton loop
is the *same* shared C++ code for eager and AOTI
(`neml2/csrc/aoti/newton.cpp`, reached from `Newton.solve` via
`newton_solve_eager`). Eager avoids an Inductor compile per grid point. This is
a convergence study, not a wall-time benchmark, so the "benchmarks run through
AOTI" rule does not apply; `benchmark/_p3_precond_study.py` makes the same call
for its Krylov study. Any *timing* claim would have to move to AOTI.

## How the capture works

`NEML2_LOGS=newton=debug` plus `neml2.log.set_sink(...)`, which intercepts the
C++ solver's lines in-process — simpler than the fd-2 `dup2` juggling in
`benchmark/_p3_precond_study.py`.

Two subtleties worth knowing before editing `harness.py`:

* Solves are split on the `---- begin/end newton solve ----` banners, **not** on
  the `ITERATION 0` line. When `NEML2_CAPTURE_SOLVE_FAILURE` is set, a
  divergence triggers a masked re-run (`Newton::solve_masked`) that replays the
  same iterations with no banner and no iteration-0 line. Splitting on
  `ITERATION 0` silently concatenates that replay and doubles the count.
* The `ITERATION` regex requires the literal `, |R| = ` suffix, which excludes
  the indented `  LS ITERATION   n, min(alpha) = ...` line-search
  sub-iterations. Those are not Newton iterations.

## Adding a case

1. Copy the parent `model.i` into `cases/<name>/`.
2. Delete its `[Drivers]/[regression]` sub-block (no gold reference here).
3. Replace the batch/step literals with `${nbatch}` / `${npoint}`, the **upper
   bound** of the load-history `linspace` with `${tfrac}` (this is what makes
   the step count and the increment independent — see above), the flow exponent
   with `${flow_n}`, and rewrite `[Solvers]/[newton]` to `NewtonWithLineSearch`
   with `max_linesearch_iterations = '${ls_iters}'` and `max_its = '${max_its}'`.
4. Register it in `cases.py`, setting `solves_per_step` to the number of
   `ImplicitUpdate`s the driver evaluates per step. The harness asserts the
   captured solve count against it, so a wiring change fails loudly instead of
   misattributing every per-step metric.

**If the case has a sub-batched unknown** (a per-slip internal variable, say),
set `supports_nopred=False` with a reason. Without a predictor the unknown
becomes an ordinary model input, and `TransientDriver` zero-fills unmatched
inputs at *base* shape — dropping the sub-batch axis. The unknown vector then
has fewer rows than the residual and the linear solve dies with
`linalg.solve: A must be batches of square matrices`. The flag turns that
cryptic crash into a recorded `skipped` row. (This is a real NEML2 gap:
`predictor` is documented as optional, but omitting it is not actually viable
for a sub-batched system.)

## Output

`results/<name>/` holds three long-format CSVs:

| file | one row per |
| --- | --- |
| `summary.csv` | grid point, with every derived metric |
| `steps.csv` | (grid point, driver step) — iterations |
| `traces.csv` | (grid point, Newton iteration) — the step-1 residual trace |
