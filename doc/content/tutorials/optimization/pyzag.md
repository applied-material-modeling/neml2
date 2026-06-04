(tutorials-optimization-pyzag)=
# Recurrent calibration with pyzag

The [previous tutorial](tutorials-optimization-calibration) calibrates a
*single-step* model — one forward call per loss evaluation. Constitutive
calibration problems usually aren't single-step. The data is a **time
history**: a strain (or strain + temperature, or mixed-control) path
with $N$ steps, and the observable is the stress trajectory the material
traces out over those $N$ steps. Recovering parameters from such data
means backpropagating through $N$ coupled implicit solves, where the
state at step $k$ is a recursive function of the state at step $k-1$.

## What hurts about plain PyTorch here

Two things go wrong if you try to do this with vanilla PyTorch autograd:

1. **Memory.** Backward AD across the unrolled recursion stores every
   intermediate tensor from every Newton iteration of every step. Peak
   memory scales linearly in $N$ and quickly outgrows GPU RAM for
   realistic load histories (hundreds to thousands of steps × hundreds
   of specimens × Newton-iterations-per-step).
2. **Throughput.** A naive unroll evaluates one step at a time. That's
   fine for a deep neural network — each layer is heavy and the GPU
   stays busy — but a viscoplastic update is microseconds per step, so
   the GPU sits idle waiting for serial Python work. The actual compute
   is a tiny fraction of wall time.

Either issue alone makes calibration painful; together they cap the
problem size at "small enough to embarrass-parallelize on CPU."

## How pyzag addresses both

The [`pyzag`](https://github.com/applied-material-modeling/pyzag)
companion package targets exactly this shape of problem:

- The **chunked adjoint method** computes the parameter gradient by
  solving a *backward* recursion (the adjoint problem) — peak memory is
  proportional to the chunk size, not the total number of steps. The
  result is mathematically equivalent to backward AD but uses orders of
  magnitude less memory.
- The **vectorized time integration** advances `block_size` steps at
  once instead of one at a time, feeding the device more work per
  iteration. The underlying scheme is documented in
  [this paper](https://arxiv.org/abs/2310.08649).

The NEML2 ↔ pyzag interface lives in the {mod}`neml2.pyzag` submodule.
{class}`~neml2.pyzag.NEML2PyzagModel` wraps a NEML2
{class}`~neml2.equation_systems.NonlinearSystem` as a pyzag
`NonlinearRecursiveFunction`, mirroring each HIT-named NEML2 parameter
as a flat `torch.nn.Parameter` on the wrapper. From that point onward
the calibration loop is a standard PyTorch optimizer loop — with the
single substitution that `nonlinear.solve_adjoint(...)` replaces
`solve(...).backward()`.

## Where to go from here

Two end-to-end notebooks demonstrate the workflow on a temperature- and
rate-dependent viscoplastic model. Both ship with pre-baked outputs, so
they're readable without re-execution:

- [](/notebooks/deterministic) — point-estimate calibration with the
  chunked adjoint, covering parameter rescaling, optimizer choice
  (Adam vs LBFGS), and goodness-of-fit diagnostics.
- [](/notebooks/statistical) — extends the same setup to a hierarchical
  Bayesian fit with Stochastic Variational Inference via
  [`pyro`](https://pyro.ai/), so you get parameter posteriors instead
  of point estimates.

For the underlying algorithms (chunked adjoint derivation, predictors,
block-size trade-offs, custom solver choices), see the
[pyzag documentation](https://applied-material-modeling.github.io/pyzag/).
