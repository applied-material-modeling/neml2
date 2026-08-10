(sub-batch-declaration)=
# Sub-batch axes

Most variables in a model are one value per batch member: a stress, a
temperature, a scalar hardening variable. Some are one value *per site within*
each batch member — a slip rate per slip system, a number density per
precipitate bin, a state per grain. That extra structure is the **sub-batch**
region, and a tensor carrying it decomposes as

```
(*dynamic_batch, *sub_batch, *base)
```

where `base` is fixed by the tensor type (`Scalar` has none, `SR2` has `(6,)`)
and `dynamic_batch` is whatever the caller passes at run time. `sub_batch` sits
between them, and unlike the other two it is neither derivable from the type
nor free to vary per call: 12 slip systems is a property of the material model,
fixed for the life of the run.

Every route agrees on this decomposition — see [](deployment-overview) for the
route list. What differs is only where the numbers come from.

## Declaring the extent

State the extent in `[Settings]`:

```
[Settings]
  [example_batch_shape]
    dislocation_density = '(2; 12)'
  []
[]
```

The `;` splits the spec into a dynamic region and a sub-batch region. Here the
variable `dislocation_density` has 12 sub-batch entries. The `2` on the left is
a **nominal** trace hint: it seeds Inductor's kernel autotuning during
`neml2-compile` and is ignored everywhere else, because at run time the leading
batch axis is whatever you pass. The part after the `;` is real and every route
reads it.

A uniform spec applies to every variable:

```
[Settings]
  example_batch_shape = '(1000,)'
[]
```

The two forms are mutually exclusive within one file.

A spec declares a sub-batch extent **only if it writes one** — only if it
carries a `;`. `'(1000,)'` pins a dynamic batch and says nothing about
sub-batch, which is why the uniform form above does not assert that every
variable in the file is un-sub-batched. To assert that explicitly, write the
region and leave it empty: `'(2; )'`.

A sub-batch axis belongs to the **variable**, not to the time step, so
declaring `dislocation_density` also covers its history lags
`dislocation_density~1`, `dislocation_density~2`, …. Declaring a lag
differently from its base variable is rejected when the file is read.

## What the declaration is for

Without it, the sub-batch extent has to be inferred from whichever tensor
happens to arrive first — and some never arrive at all:

- An implicit unknown with **no predictor** is not supplied by anything. The
  driver invents a zero initial guess for it, and at base shape that guess has
  one row where the residual has one per site; the linear solve then fails with
  a non-square matrix.
- An unknown with **no history input** (a rate, e.g. `slip_rates`) has no
  earlier value to copy a shape from.
- An **initial condition** ends up doing the declaring by accident: writing
  `Scalar(torch.full((nbatch, 12), 1e1), sub_batch_ndim=1)` exists mostly so
  that *something* in the file establishes the 12.

With the extent declared, that last one goes back to being a value:

```
[Tensors]
  [initial_dislocation_density]
    type = Python
    expr = 'Scalar(1.0e1)'
  []
[]
```

The seed broadcasts into the declared extent. `per_slip_hardening` and
`per_slip_hardening_declared` under
`tests/regression/solid_mechanics/crystal_plasticity/` are the same scenario
written both ways, and they agree exactly.

## Consistency

A declaration and a hand-shaped value are two statements about one variable, so
a file that makes them differently is rejected rather than silently resolved —
otherwise the driver would build one shape while `neml2-compile` traced the
other. A value carrying **no** sub-batch region is not a contradiction: it
broadcasts into whatever is declared.

Check what a file declares before running it:

```bash
neml2-inspect model.i model
```

Each input and output that carries a declared extent is printed with it. A
variable you expected to see without one usually means a mistyped name — a
declaration for a name no variable has is not applied.

## Where each route reads it

| Consumer | Uses the declaration to |
| --- | --- |
| `TransientDriver` | size the zero-fill for model inputs no force or initial condition supplies |
| `ImplicitUpdate` | lay out the unknown vector, and expand a seeded initial guess into the declared extent |
| `neml2-compile` | build the example inputs it traces, and seed each nested equation system's layout ([](model-compilation-pipeline)) |
| `NEML2PyzagModel` | check the state and force tensors it is handed |

All of them resolve it through the same reader, which is what keeps the routes
from drifting apart.

## Limits

The extent is a declaration, not an inference: it is not derived from the
crystal geometry, the bin count, or any other data object in the file. State it
explicitly.

`cpp-eager` does not support sub-batch models at all — it has no slot to
declare per-input sub-batch shapes at its boundary and rejects them. See
[](model-eager-cpp).
