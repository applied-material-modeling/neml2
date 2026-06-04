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

A backward-Euler integration of the constitutive ODE
$\dot{\boldsymbol{y}} = \boldsymbol{h}(\boldsymbol{y}, t; \boldsymbol{p})$
solves, at every step,

$$
\boldsymbol{y}_{i+1} - \boldsymbol{y}_i
  - \boldsymbol{h}(\boldsymbol{y}_{i+1}, t_{i+1}; \boldsymbol{p})\,\Delta t_{i+1}
  = \boldsymbol{0}
$$

via Newton's method on $\boldsymbol{y}_{i+1}$. Calibrating $\boldsymbol{p}$
against an $N$-step trajectory means evaluating this $N$ times forward and
then differentiating end-to-end through the recursion. Two things go wrong
if you try to do that with vanilla PyTorch autograd:

1. **Memory.** Backward AD across the unrolled recursion stores every
   intermediate tensor from every Newton iteration of every step. Peak
   memory scales linearly in $N$ and quickly outgrows GPU RAM for
   realistic load histories (hundreds to thousands of steps × hundreds
   of specimens × Newton-iterations-per-step).
2. **Throughput.** Each step is solved one at a time, with control
   bouncing back to Python between steps. For a viscoplastic update
   that runs in microseconds per step the GPU is starved — the actual
   compute is a tiny fraction of wall time.

Either issue alone makes calibration painful; together they cap the
problem size at "small enough to embarrass-parallelize on CPU."

## How pyzag addresses both

The [`pyzag`](https://github.com/applied-material-modeling/pyzag)
companion package solves both by changing what the time loop looks like.
The key trick, from Messner & Hu
([arXiv:2310.08649](https://arxiv.org/abs/2310.08649)), is to lift the
recursion into a **block-bidiagonal** nonlinear system over a chunk of
$n_\text{chunk}$ contiguous steps. Writing
$\boldsymbol{y}_{i+j} = \boldsymbol{y}_i + \Delta\boldsymbol{y}_j$, the
chunked residual is

$$
\begin{bmatrix}
  \Delta\boldsymbol{y}_1 - \boldsymbol{h}(\boldsymbol{y}_i + \Delta\boldsymbol{y}_1, t_{i+1}; \boldsymbol{p})\,\Delta t_{i+1} \\
  \Delta\boldsymbol{y}_2 - \Delta\boldsymbol{y}_1 - \boldsymbol{h}(\boldsymbol{y}_i + \Delta\boldsymbol{y}_2, t_{i+2}; \boldsymbol{p})\,\Delta t_{i+2} \\
  \vdots \\
  \Delta\boldsymbol{y}_{n_\text{chunk}} - \Delta\boldsymbol{y}_{n_\text{chunk}-1} - \boldsymbol{h}(\boldsymbol{y}_i + \Delta\boldsymbol{y}_{n_\text{chunk}}, t_{i+n_\text{chunk}}; \boldsymbol{p})\,\Delta t_{i+n_\text{chunk}}
\end{bmatrix} = \boldsymbol{0},
$$

and its Newton Jacobian has block-bidiagonal structure

$$
\boldsymbol{J} =
\begin{bmatrix}
  \boldsymbol{I} - \boldsymbol{j}_{i+1}\,\Delta t_{i+1} & & & \\
  -\boldsymbol{I} & \boldsymbol{I} - \boldsymbol{j}_{i+2}\,\Delta t_{i+2} & & \\
  & \ddots & \ddots & \\
  & & -\boldsymbol{I} & \boldsymbol{I} - \boldsymbol{j}_{i+n_\text{chunk}}\,\Delta t_{i+n_\text{chunk}}
\end{bmatrix}
$$

with the ODE Jacobian
$\boldsymbol{j}_{i+j} = \partial \boldsymbol{h}/\partial \boldsymbol{y}_{i+j}$
on each diagonal block. Two consequences follow:

- **Vectorized time integration.** A full chunk's residual and Jacobian
  are assembled and evaluated in a single batched call to the
  constitutive model — `n_chunk` × `n_batch` instances at once. The
  bidiagonal solve uses parallel cyclic reduction so even the linear
  algebra avoids serial fan-in. This is where the paper reports
  >100× wall-time speedups over sequential integration.
- **Chunked adjoint gradient.** The parameter gradient is computed by
  solving a *backward* recursion (the adjoint problem) over the same
  chunks. Peak memory scales with `n_chunk`, not $N$ — mathematically
  equivalent to backward AD over the full unrolled recursion, but with
  orders of magnitude less storage.

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
