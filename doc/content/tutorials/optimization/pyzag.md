---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(tutorials-optimization-pyzag)=
# Recurrent calibration with pyzag

The [previous tutorial](tutorials-optimization-calibration) calibrates a
*single-step* model — one forward call per loss evaluation. Most
constitutive calibration problems are not like that. The data is a
**time history**: a strain (or strain + temperature, or mixed-control)
path with N steps, and the observable is the stress trajectory the
material traces out over those N steps. Recovering parameters from such
data means backpropagating through N coupled implicit solves, where the
state at step $k$ is a recursive function of the state at step $k-1$.

Two things break if you try to do this with plain PyTorch autograd:

1. **Memory.** Backward AD across the unrolled recursion stores every
   intermediate tensor from every Newton iteration of every step. Memory
   scales linearly in the number of steps and quickly outgrows GPU RAM.
2. **Throughput.** A naive unroll evaluates one step at a time — fine
   for a 5-stage NN, terrible for a viscoplastic update that runs in
   microseconds per step. The GPU sits idle most of the time.

The [`pyzag`](https://github.com/applied-material-modeling/pyzag)
companion package solves both:

- The **chunked adjoint method** evaluates the parameter gradient by
  solving a *backward* recursion (the adjoint problem) — peak memory is
  proportional to the chunk size, not the total number of steps. The
  result is mathematically equivalent to backward AD but uses orders of
  magnitude less memory.
- The **vectorized time integration** advances `block_size` steps at
  once instead of one at a time, feeding the device more work per
  iteration. The basic idea is documented in
  [this paper](https://arxiv.org/abs/2310.08649).

The NEML2 ↔ pyzag interface lives in the {mod}`neml2.pyzag` submodule.
Its job is to wrap a NEML2 nonlinear system as a pyzag
`NonlinearRecursiveFunction` so the same gradient-based optimizer loop
from the previous tutorial keeps working, only now the forward "model"
is an entire transient simulation.

## The model

We calibrate a small-strain Perzyna viscoplastic model with linear
isotropic hardening. The setup mirrors a
[transient driver](tutorials-models-transient-driver) problem but
strips the driver wrapper away — pyzag plays the role of the time
integrator.

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The implicit residual lives in the `[EquationSystems]` block:
`unknowns = 'stress equivalent_plastic_strain'` says we solve for the
seven scalar components of stress + plastic strain at every step,
treating `strain` and `t` as forcing variables. That's exactly the
contract `NEML2PyzagModel` consumes.

## Wrapping the system

`load_nonlinear_system` parses the input file and returns a native
`ModelNonlinearSystem`; `NEML2PyzagModel` adapts it to pyzag's
`NonlinearRecursiveFunction` interface. The wrapper mirrors each
HIT-named NEML2 parameter as a `torch.nn.Parameter` on itself, with a
flat `<block>_<leaf>` key (`yield_sy`, `flow_rate_eta`, …) so torch's
optimizers and parametrization hooks can reach them.

```{code-cell} ipython3
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

from neml2 import load_nonlinear_system
from neml2.pyzag import NEML2PyzagModel

# Three nuisance "parameters" that the underlying SR2LinearCombination
# registers internally (the relative weights of the strain-rate
# difference and a zero offset). They're framework plumbing rather than
# material parameters and we exclude them from the wrapper so they don't
# show up in named_parameters().
LINCOMB_INTERNALS = ["Eerate_offset", "Eerate_weight_0", "Eerate_weight_1"]

nsys = load_nonlinear_system("input.i", "eq_sys")
pmodel = NEML2PyzagModel(
    nsys,
    exclude_parameters=LINCOMB_INTERNALS + ["elasticity_nu"],
)
pmodel
```

Inspect the layout the wrapper exposes to pyzag:

```{code-cell} ipython3
print("State variables (nstate = {}): {}".format(pmodel.nstate, pmodel.svars))
print("Force variables (nforce = {}): {}".format(pmodel.nforce, pmodel.fvars))
print("Lookback                    : {}".format(pmodel.lookback))
print("Tunable parameters:")
for name, p in pmodel.named_parameters():
    if name in pmodel._param_targets:
        print(f"  {name:20s} = {p.item():g}")
```

A couple of things to read off this output:

- The wrapper has flattened the per-variable tensor layout the underlying
  model uses (`SR2` stress + `Scalar` equivalent plastic strain) into a
  single trailing `nstate = 7` axis. The strain forcing is similarly
  flattened: 6 Mandel components + a scalar time = 7.
- `lookback = 1` means the recursion needs one previous step to advance
  — that's just backward Euler. pyzag will allocate space for the
  current step *and* the previous step in every chunk.
- Five `nn.Parameter`s show up under their HIT block names. That's the
  optimization surface: torch's `Adam`, `LBFGS`, whatever you choose
  will see exactly these tensors via `pmodel.parameters()`.

## Running a forward simulation

Build a uniaxial tension history — `nbatch = 5` specimens, `nstep = 50`
time steps each, strain ramped from 0 to 1 % along the loaded axis:

```{code-cell} ipython3
nstep, nbatch = 50, 5

# Time: linear ramp shared across all specimens.
t = torch.linspace(0.0, 1.0, nstep).unsqueeze(-1).expand(nstep, nbatch)

# Strain: 1 % uniaxial tension, with Poisson lateral contraction.
end_strain = torch.tensor([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
strain = (
    torch.linspace(0.0, 1.0, nstep).reshape(nstep, 1, 1)
    * end_strain.reshape(1, 1, 6)
).expand(nstep, nbatch, 6).contiguous()

# Pack into the pyzag-flat (nstep, nbatch, nforce) layout, in pmodel.fvars
# order: strain (6 components) first, then t (1 component).
forces = torch.cat([strain, t.unsqueeze(-1)], dim=-1)

# Initial state: zero stress + zero plastic strain.
state0 = torch.zeros(nbatch, pmodel.nstate)

forces.shape, state0.shape
```

The pyzag solver couples the wrapper to a step generator, a predictor
that seeds each chunk's Newton iteration, and an inner nonlinear solver.
The block size controls how many steps are solved together — this is
the parallel-in-time knob from the paper.

```{code-cell} ipython3
from pyzag import nonlinear, chunktime

solver = nonlinear.RecursiveNonlinearEquationSolver(
    pmodel,
    step_generator=nonlinear.StepGenerator(block_size=10),
    predictor=nonlinear.PreviousStepsPredictor(),
    nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1e-8, atol=1e-10),
)

with torch.no_grad():
    trajectory = nonlinear.solve(solver, state0, nstep, forces)

trajectory.shape
```

The trajectory is `(nstep, nbatch, nstate)`. The first six components of
the trailing axis are the Mandel stress; the seventh is the equivalent
plastic strain. Read off the final state of the first specimen:

```{code-cell} ipython3
print("Final stress (Mandel components):", trajectory[-1, 0, :6].tolist())
print("Final equivalent plastic strain :", trajectory[-1, 0, 6].item())
```

The `block_size` controls how many steps the chunked Newton solver
attacks together. Going from 1 to 50 should change wall time
substantially but not the answer:

```{code-cell} ipython3
ref = trajectory

for block in (1, 5, 10, 25):
    sv = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        step_generator=nonlinear.StepGenerator(block_size=block),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1e-8, atol=1e-10),
    )
    with torch.no_grad():
        out = nonlinear.solve(sv, state0, nstep, forces)
    print(f"block_size = {block:>2}: max|Δ| vs block=10 = {(out - ref).abs().max().item():.2e}")
```

All four chunkings agree to ~1e-8 — exactly the absolute tolerance of
the nonlinear solver. Pick a block size that fits your memory budget.

## Synthetic ground truth

To demonstrate calibration we need data to calibrate *against*. Use the
as-loaded parameters as ground truth, generate a stress trajectory,
then start from a perturbed parameter guess and try to recover the
original. The "experiment" only observes stress — equivalent plastic
strain is internal.

```{code-cell} ipython3
truth_sys = load_nonlinear_system("input.i", "eq_sys")
truth = NEML2PyzagModel(
    truth_sys,
    exclude_parameters=LINCOMB_INTERNALS + ["elasticity_nu"],
)

with torch.no_grad():
    truth_solver = nonlinear.RecursiveNonlinearEquationSolver(
        truth,
        step_generator=nonlinear.StepGenerator(block_size=10),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1e-8, atol=1e-10),
    )
    obs_stress = nonlinear.solve(
        truth_solver, state0, nstep, forces
    )[..., :6].detach()

obs_stress.shape  # (nstep, nbatch, 6)
```

## Calibration with the chunked adjoint

We treat the yield stress and the Perzyna pre-exponential as the two
unknown parameters and pretend everything else is known. Exclude the
"known" parameters from the wrapper so pyzag's adjoint doesn't try to
differentiate through them (a parameter is either fully optimized or
fully excluded — pyzag accumulates gradients onto every
`requires_grad=True` leaf it finds, so freezing via
`requires_grad_(False)` doesn't work here):

```{code-cell} ipython3
FROZEN = ["elasticity_E", "elasticity_nu", "isoharden_K", "flow_rate_n"]

trial_sys = load_nonlinear_system("input.i", "eq_sys")
trial = NEML2PyzagModel(
    trial_sys,
    exclude_parameters=LINCOMB_INTERNALS + FROZEN,
)

# Perturb the trial parameters away from truth: yield stress 8 vs 5,
# pre-exponential 150 vs 100.
with torch.no_grad():
    trial.yield_sy.data = torch.tensor(8.0)
    trial.flow_rate_eta.data = torch.tensor(150.0)

[name for name, _ in trial.named_parameters() if name in trial._param_targets]
```

The calibration loop is a vanilla PyTorch optimizer loop, with one
substitution: `nonlinear.solve_adjoint` replaces the would-be
`nonlinear.solve(...).backward()`. The forward pass returns the
trajectory; the backward pass solves the chunked adjoint problem to
get gradients with respect to `trial`'s `nn.Parameter`s.

```{code-cell} ipython3
trial_solver = nonlinear.RecursiveNonlinearEquationSolver(
    trial,
    step_generator=nonlinear.StepGenerator(block_size=10),
    predictor=nonlinear.PreviousStepsPredictor(),
    nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1e-8, atol=1e-10),
)
opt = torch.optim.Adam(trial.parameters(), lr=0.5)

print(f"{'step':>4}  {'loss':>12}  {'yield_sy':>10}  {'flow_rate_eta':>14}")
for step in range(8):
    opt.zero_grad()
    pred = nonlinear.solve_adjoint(trial_solver, state0, nstep, forces)
    loss = ((pred[..., :6] - obs_stress) ** 2).mean()
    loss.backward()
    opt.step()
    print(f"{step:>4d}  {loss.item():12.4e}  "
          f"{trial.yield_sy.item():10.4f}  {trial.flow_rate_eta.item():14.4f}")
```

Loss decreases monotonically — adjoint gradients are pointing in the
right direction. But notice the parameters are barely moving: `yield_sy`
and `flow_rate_eta` get the same `lr` from Adam even though their
natural scales differ by 20×. This is a classic calibration failure
mode and the next section is its fix.

## Reparameterizing for scale

`pyzag.reparametrization.RangeRescale` wraps each parameter in a
bijection from the unit interval to a user-specified
`[lower_bound, upper_bound]` range. The optimizer then operates on the
unconstrained (~unit-scale) tensor; the wrapper transforms back to the
natural scale on every forward call. The mechanism is just PyTorch's
[`torch.nn.utils.parametrize`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html)
hook, so the wrapper-side parameter (the thing the optimizer sees) is
*not* the same tensor as the underlying NEML2 parameter (the thing the
model evaluates with). `NEML2PyzagModel._update_parameter_values` runs
on every forward call to copy the reparametrized value into the
underlying model.

```{code-cell} ipython3
from pyzag.reparametrization import RangeRescale, parametrize

trial_sys = load_nonlinear_system("input.i", "eq_sys")
trial = NEML2PyzagModel(
    trial_sys,
    exclude_parameters=LINCOMB_INTERNALS + FROZEN,
)

bounds = {"yield_sy": (1.0, 20.0), "flow_rate_eta": (10.0, 500.0)}
starts = {"yield_sy": 8.0,         "flow_rate_eta": 150.0}

for name, (lb, ub) in bounds.items():
    rescaler = RangeRescale(lb=torch.tensor(lb), ub=torch.tensor(ub))
    parametrize.register_parametrization(trial, name, rescaler)
    # Seed the *unconstrained* tensor so the natural-scale value matches
    # the desired start. RangeRescale doesn't implement right_inverse, so
    # we invert manually with rescaler.reverse.
    with torch.no_grad():
        getattr(trial.parametrizations, name).original.data = (
            rescaler.reverse(torch.tensor(starts[name]))
        )

for name in starts:
    print(f"{name:20s} natural = {getattr(trial, name).item():7.3f}    "
          f"unconstrained = {getattr(trial.parametrizations, name).original.item():.4f}")
```

Same calibration loop, same learning rate, but now Adam's update steps
are commensurate across parameters:

```{code-cell} ipython3
trial_solver = nonlinear.RecursiveNonlinearEquationSolver(
    trial,
    step_generator=nonlinear.StepGenerator(block_size=10),
    predictor=nonlinear.PreviousStepsPredictor(),
    nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=1e-8, atol=1e-10),
)
opt = torch.optim.Adam(trial.parameters(), lr=0.05)

print(f"{'step':>4}  {'loss':>12}  {'yield_sy':>10}  {'flow_rate_eta':>14}")
for step in range(30):
    opt.zero_grad()
    pred = nonlinear.solve_adjoint(trial_solver, state0, nstep, forces)
    loss = ((pred[..., :6] - obs_stress) ** 2).mean()
    loss.backward()
    opt.step()
    if step % 3 == 0 or step == 29:
        print(f"{step:>4d}  {loss.item():12.4e}  "
              f"{trial.yield_sy.item():10.4f}  "
              f"{trial.flow_rate_eta.item():14.4f}")
```

Both parameters converge to within a few percent of the truth
(`yield_sy = 5`, `flow_rate_eta = 100`) in 30 Adam steps. The
oscillation around step 12–20 is Adam overshooting on a fairly stiff
objective; switching to LBFGS or annealing the learning rate would
tighten the final values.

## Where to go next

- The pyzag package has its own
  [documentation](https://applied-material-modeling.github.io/pyzag/)
  with deeper coverage of the chunked adjoint algorithm, predictors,
  and bayesian extensions (variational inference, HMC) on top of the
  same wrapper.
- For uncertainty quantification, pyzag plugs into
  [`pyro`](https://pyro.ai/) — the `NEML2PyzagModel` wrapper is already
  an `nn.Module`, so any pyro guide that operates on torch modules
  works without further adaptation.
- The wrapper exposes additional escape hatches not shown here:
  `pmodel._param_targets` is the runtime sync table mapping wrapper-side
  flat names back to the owning native submodule, useful when you need
  to reach the underlying parameter object directly (for instance, to
  share a parameter across multiple loaded systems).
