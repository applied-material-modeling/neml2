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

This tutorial walks through the end-to-end "hello world" of NEML2:
write an input file, load it from Python, evaluate the model on an
input, read the output. The model is intentionally trivial — linear
isotropic elasticity — so the focus stays on the workflow.

## The physics

Linear isotropic elasticity maps a symmetric strain tensor
$\boldsymbol{\varepsilon}$ to a symmetric stress tensor
$\boldsymbol{\sigma}$ via

$$
  \boldsymbol{\sigma}
  = 3K\,\operatorname{vol}\boldsymbol{\varepsilon}
  + 2G\,\operatorname{dev}\boldsymbol{\varepsilon},
$$

where $K$ is the bulk modulus and $G$ is the shear modulus. The model
takes the moduli as parameters in any one of several equivalent
parameterizations — here we feed it Young's modulus $E$ and Poisson's
ratio $\nu$ and let it derive $K$ and $G$ internally.

More abstractly, every NEML2 model is a map

$$
  y = f(x;\, p, b),
$$

where $x$ are the input variables, $y$ the output variables, $p$ the
parameters, and $b$ the buffers. Calling the model evaluates $f$ at
some $x$ with whatever $p, b$ were set up at load time.

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Two things are happening here:

1. The `type = LinearIsotropicElasticity` line selects a model class
   from the registry. Browse [](models-LinearIsotropicElasticity) for
   the full option list.
2. `coefficients` + `coefficient_types` define the elastic moduli. The
   pair `'YOUNGS_MODULUS POISSONS_RATIO'` tells the model to interpret
   the two numbers as Young's modulus and Poisson's ratio (other
   choices include `BULK_MODULUS`, `SHEAR_MODULUS`, `LAME_LAMBDA`,
   `P_WAVE_MODULUS`).

## Loading the model

`neml2.load_model(path, name)` parses the input file and returns the
named model as a Python object — a `Model` subclass (which is itself a
`torch.nn.Module`):

```{code-cell} ipython3
import neml2
model = neml2.load_model("input.i", "elasticity")
model
```

## Evaluating the model

A model is invoked like any callable. The input here is a symmetric
second-order tensor (`SR2`) representing the elastic strain. NEML2's
`SR2` wraps a `torch.Tensor` of base shape `(6,)`, storing the six
independent components in Mandel notation:

```{code-cell} ipython3
import torch
from neml2.types import SR2

# Uniaxial tension along the x-axis: epsilon_xx = 1%
strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
strain
```

Now evaluate:

```{code-cell} ipython3
stress = model(strain)
stress
```

The output is the symmetric Cauchy stress. The leading three slots are
the diagonal components; the trailing three are the off-diagonal
components in Mandel packing. For uniaxial strain with the moduli
above, the result matches the closed-form `K + 4G/3` along the loaded
axis and `K - 2G/3` along the lateral axes.

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
