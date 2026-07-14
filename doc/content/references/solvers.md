(solvers-reference)=
# Solvers

Solvers drive the implicit update at the heart of a rate-form constitutive
model. When a model has no closed form — the state at the end of a step depends
on itself — it is written as a residual $r(u, g) = 0$ (unknowns $u$, given
inputs $g$), wrapped in an [](models-ImplicitUpdate), and handed to a solver
that finds the root. This page is the conceptual map of the available solvers
and when to reach for each; the [`[Solvers]` catalog](solvers-syntax) is the
canonical per-type option list (every field, its meaning, its default).

Two kinds compose:

- a **nonlinear solver** runs the outer Newton iteration over $r(u) = 0$;
- a **linear solver** solves the linear system in each Newton step
  ($J\,\Delta u = -r$, where $J = \partial r / \partial u$). The nonlinear
  solver names it through its `linear_solver` field.

## Nonlinear solvers

- [](solvers-Newton) — Newton–Raphson. Each iteration assembles the residual
  and its Jacobian and takes one linear solve; it stops when the relative or
  absolute residual norm falls below `rel_tol` / `abs_tol`, or errors at
  `max_its`. `verbose` prints the per-iteration convergence history. This is the
  default choice.
- [](solvers-NewtonWithLineSearch) — Newton with a globalized step. When the
  full Newton step overshoots (stiff or far-from-solution problems), a line
  search (`BACKTRACKING` or `STRONG_WOLFE`) scales it back to guarantee residual
  decrease. Reach for it when plain Newton stalls or diverges; it inherits every
  `Newton` option and adds the `linesearch_*` controls.

## Linear solvers

Each Newton step solves $J\,\Delta u = -r$. The choice trades assembly +
factorization cost against per-iteration cost.

**Direct** — assemble $J$ and factorize it:

- [](solvers-DenseLU) — a dense LU solve of a single-group system. The default,
  and the right choice for the common small dense update (e.g. a $6\times6$
  stress return).
- [](solvers-SchurComplement) — a block factorization for a two-group system
  (one `BLOCK` + one `DENSE` group, e.g. crystal plasticity's per-grain state
  coupled to a global unknown). It inverts the primary block and solves the
  Schur complement, each with its own nested linear solver (`primary_solver` /
  `schur_solver`).

**Matrix-free iterative** — never assemble $J$; apply $J\,v$ directly:

- [](solvers-GMRES) / [](solvers-BiCGStab) — Krylov solvers that reach the
  solution through a handful of Jacobian-vector products. For the large,
  well-conditioned systems of implicit backward-Euler updates, a few matvecs
  beat an $O(N^3)$ dense factorization, and the win grows with system size and
  on GPU. `GMRES` keeps a restart window (`restart`); both take `max_its` /
  `abs_tol` / `rel_tol`. The inner tolerance can be loose — the outer Newton
  re-solves each step, so an inexact inner solve still converges.

### Preconditioners

An iterative linear solver may take a preconditioner $M^{-1} \approx J^{-1}$
that clusters the spectrum and cuts the iteration count. Each is an authored
`[Solvers]` object referenced by `GMRES` / `BiCGStab`:
[](solvers-NoPreconditioner) (the default), [](solvers-JacobiPreconditioner),
[](solvers-BlockJacobiPreconditioner), and [](solvers-FullPreconditioner).
The `cache_strategy` field governs how often the preconditioner is rebuilt
across Newton iterations — `none` (every step), `chord` (build once, reuse), or
`max_its` (rebuild when a solve exceeds `cache_max_its` iterations).

## Derivative (sensitivity) solvers

Differentiating through the converged solve uses the implicit function theorem,
which needs its own linear solves: $\partial u/\partial g = -J^{-1}\,\partial
r/\partial g$ (inputs) and $\partial u/\partial \theta = -J^{-1}\,\partial
r/\partial \theta$ (parameters). [](models-ImplicitUpdate) exposes these as
`input_sensitivity_solver` and `param_sensitivity_solver`. Both default to a
direct solve — so derivatives stay exact regardless of the forward solver — and
either can be set to an iterative solver to trade exactness for speed on large
systems.

## Wiring it together

A solver is assembled from the objects it references. A `Newton` names its
`linear_solver`; an iterative linear solver may name a preconditioner; an
`ImplicitUpdate` names the `Newton` (and, optionally, distinct sensitivity
solvers):

```ini
[Solvers]
  [newton]
    type = Newton
    linear_solver = 'gmres'
  []
  [gmres]
    type = GMRES
    preconditioner = 'bjac'
    cache_strategy = 'chord'
  []
  [bjac]
    type = BlockJacobiPreconditioner
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
```

## See also

- The [`[Solvers]` catalog](solvers-syntax) — the exhaustive per-type option
  reference (fields, meanings, defaults).
- [](tutorials-models-implicit-model) — a worked implicit update, from residual
  to Newton solve to differentiating through it.
- [](model-compilation-pipeline) — how the direct and iterative solves are
  lowered when a model is compiled with `neml2-compile`.
