(tutorials-optimization-pyzag)=
# Recurrent calibration with pyzag

Time-history calibration multiplies the cost of plain PyTorch
autograd in two ways at once. This page explains the bottlenecks and
how the `pyzag` companion package addresses them.

The [previous tutorial](tutorials-optimization-calibration) calibrates a
model with a single forward call per loss evaluation. Real constitutive
data is a **time history**: a loading path with $N$ steps, and the
observable is the stress trajectory the material traces out.
Recovering parameters from that data means backpropagating through
$N$ coupled implicit solves — each step's state depends on the
previous one, so the time loop becomes a long autograd chain.

## Why plain PyTorch struggles at this scale

Two things go wrong if we try to backpropagate through the full
unrolled time loop with vanilla autograd:

1. **Memory.** Backward AD stores every intermediate tensor from every
   Newton iteration of every step. Peak memory scales linearly in $N$
   and quickly outgrows GPU RAM for realistic load histories (hundreds
   to thousands of steps × hundreds of specimens).
2. **Throughput.** Each step is solved one at a time, with control
   bouncing back to Python between steps. For a viscoplastic update
   that runs in microseconds per step, the GPU is starved.

Either issue alone makes calibration painful; together they cap the
problem size at "small enough to run on CPU."

## How pyzag addresses both

The [`pyzag`](https://github.com/applied-material-modeling/pyzag)
companion package fixes both by changing the shape of the time loop.
The key trick, from Messner, Hu & Chen
([arXiv:2310.08649](https://arxiv.org/abs/2310.08649)), is to solve
$n_\text{chunk}$ contiguous time steps together as a single
block-bidiagonal nonlinear system. Two consequences follow:

- **Vectorized time integration.** A full chunk's residual and Jacobian
  are assembled in one batched call to the constitutive model —
  `n_chunk` × `n_batch` instances at once. The paper reports >100×
  wall-time speedups over sequential integration on its ODE
  benchmarks; the actual gain for a given NEML2 model depends on
  `n_chunk`, batch size, and device.
- **Chunked adjoint gradient.** The parameter gradient is computed by
  a *backward* recursion (the adjoint problem) over the same chunks.
  Peak memory scales with `n_chunk`, not $N$ — mathematically
  equivalent to full backward AD, but with orders of magnitude less
  storage.

The NEML2 ↔ pyzag interface lives in the {mod}`neml2.pyzag` submodule.
{class}`~neml2.pyzag.NEML2PyzagModel` wraps a NEML2 nonlinear system as
a pyzag `NonlinearRecursiveFunction`, exposing each NEML2 parameter as
a `torch.nn.Parameter` on the wrapper. From there the calibration loop
is a standard PyTorch optimizer loop, with `nonlinear.solve_adjoint(...)`
in place of `solve(...).backward()`.

## Where to go from here

Two end-to-end notebooks demonstrate the workflow on a temperature- and
rate-dependent viscoplastic model. Both ship with pre-baked outputs, so
they're readable without re-execution:

- [](deterministic/main) — point-estimate calibration with the
  chunked adjoint, covering parameter rescaling
  (`reparametrization.RangeRescale`), an Adam optimization loop, and
  stress–strain comparison plots against the synthetic data.
- [](statistical/main) — extends the same setup to a hierarchical
  Bayesian fit with Stochastic Variational Inference via
  [`pyro`](https://pyro.ai/), so we get parameter posteriors instead
  of point estimates.

For the underlying algorithms (chunked adjoint derivation, predictors,
block-size trade-offs, custom solver choices), see the
[pyzag documentation](https://applied-material-modeling.github.io/pyzag/).
