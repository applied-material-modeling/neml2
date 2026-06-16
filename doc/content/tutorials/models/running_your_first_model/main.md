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

(tutorials-models-running-your-first-model)=
# Running your first model

You'll write a tiny input file, load it from Python, give it a strain,
and read back a stress. The model is intentionally trivial — linear
isotropic elasticity:

$$
  \boldsymbol{\sigma}
  = 3K\,\operatorname{vol}\boldsymbol{\varepsilon}
  + 2G\,\operatorname{dev}\boldsymbol{\varepsilon},
$$

where $K$ is the bulk modulus and $G$ is the shear modulus. The focus
of this tutorial is the workflow, not the material — the same
`load_model` + call pattern carries over to every NEML2 model, even
though the inputs and outputs will differ.

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The `type` line picks a model from NEML2's catalog (see
[](models-LinearIsotropicElasticity) for everything
`LinearIsotropicElasticity` accepts). The `coefficients` pair sets the
elastic moduli as Young's modulus and Poisson's ratio.

## Loading the model

`neml2.load_model(path, name)` parses the input file and returns the
named model as a Python object you can call like a function:

```{code-cell} ipython3
import neml2
model = neml2.load_model("input.i", "elasticity")
model
```

## Evaluating the model

The strain is a symmetric 3×3 tensor (`SR2`). Build one for 1% uniaxial
strain along the x-axis and evaluate:

```{code-cell} ipython3
import torch
from neml2.types import SR2

# Uniaxial strain along the x-axis: epsilon_xx = 1%
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
strain
```

```{code-cell} ipython3
stress = model(strain)
stress
```

That's it — the model returned the Cauchy stress. The first three
components are the diagonal entries; the last three are the
off-diagonals (stored in Mandel form, so they each carry a $\sqrt{2}$
factor).

## Where to go next

- The next tutorial, [](tutorials-models-parameters), walks through
  how to read and mutate the moduli we hard-coded in the input file.
- The model takes a single scalar-shaped input here, but the same call
  works just as well on batched inputs — see
  [](tutorials-models-vectorization).
- Real material models compose multiple ingredients (elasticity,
  hardening, flow rule, …). The mechanism for binding them together is
  the subject of [](tutorials-models-cross-referencing) and
  [](tutorials-models-composition).
