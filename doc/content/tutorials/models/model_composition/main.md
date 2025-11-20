@insert-title:tutorials-models-model-composition

[TOC]

## Problem description

We have been working with the linear, isotropic elasticity model in the previous tutorials. We started with that example because it is arguably the simplest possible material model in the context of solid mechanics. It is simple not just because of the simplicity in the description of the material behavior, but also due to the fact that its mathematical formulation only involves one linear equation.

Much more complicated, nonlinear models can be created using NEML2.

Using a Perzyna-type viscoplasticity model as an example, it can be formulated as
\f{align*}
  \boldsymbol{\varepsilon}^e & = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \\
  \boldsymbol{\sigma} & = 3K\operatorname{vol}\boldsymbol{\varepsilon}^e + 2G\operatorname{dev}\boldsymbol{\varepsilon}^e, \\
  \bar{\sigma} & = J_2(\boldsymbol{\sigma}), \\
  f^p & = \bar{\sigma} - \sigma_y, \\
  \boldsymbol{N} & = \pdv{f^p}{\boldsymbol{\sigma}}, \\
  \dot{\gamma} & = \left( \dfrac{\left< f^p \right>}{\eta} \right)^n, \\
  \dot{\boldsymbol{\varepsilon}}^p & = \dot{\gamma} \boldsymbol{N}.
\f}
The above formulation makes a series of *constitutive choices*:
- The strain is small and can be additively decomposed into elastic and plastic strains.
- The elastic response is linear and isotropic.
- The plastic flow is isochoric.
- There is no isotropic hardening associated with plasticity.
- There is no kinematic hardening associated with plasticity.
- There is no back stress associated with plasticity.
- The plastic flow is associative.
- The plastic rate-sensitivity follows a power-law relation.

Any change in one of the constitutive choices will result in a new material model. Suppose there are a total of \f$ N \f$ constitutive choices, each having \f$ k \f$ variants, the total number of possible material models would be \f$ k^N \f$.

In other words, the number of possible material models grows *exponentially* with the number of constitutive choices, and implementing all combinations is practically infeasible.

To address such challenge, NEML2 introduces a *model composition* mechanism which allows multiple models to be "stitched" together in a flexible, modular manner.

This tutorial demonstrates model composition using a much simplified model (without loss of generality). The model can be written as
\f{align}
  \bar{a} & = I_1(\boldsymbol{a}), \label{1} \\
  \bar{b} & = J_2(\boldsymbol{b}), \label{2} \\
  \dot{b} & = \bar{b} \boldsymbol{a} + \bar{a} \boldsymbol{b}, \label{3}
\f}
where \f$ \boldsymbol{a} \f$ and \f$ \boldsymbol{b} \f$ are symmetric second order tensors.

## Writing the input file

Let us first search for available models describing this set of equations:
- \f$ \eqref{1} \& \eqref{2} \f$ correspond to [SR2Invariant](#sr2invariant);
- \f$ \eqref{3} \f$ corresponds to [LinearCombination](#sr2linearcombination).

The input file then looks like
@list-input:models/model_composition/input.i

## Evaluating the models: The hard way

Now that all three models are defined in the input file, we can load and evaluate them in sequence, with a bit of effort:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:models/model_composition/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:models/model_composition/ex2.py

  Output:
  @list-output:ex2

</div>

## Evaluating the models: The easy way

We were able to successfully calculate \f$ \dot{\boldsymbol{b}} \f$ by
1. calculating \f$ \bar{a} \f$ by evaluating \f$ \eqref{1} \f$,
2. calculating \f$ \bar{b} \f$ by evaluating \f$ \eqref{2} \f$,
3. setting the two coefficients of  \f$ \eqref{3} \f$ to be \f$ \bar{b} \f$ and \f$ \bar{a} \f$ respectively,
4. calculating \f$ \dot{\boldsymbol{b}} \f$ by evaluating \f$ \eqref{3} \f$.

However, that is not ideal because we had to
- Manually evaluate the equations and figure out the evaluation order, and
- Manually set the parameters in \f$ \eqref{3} \f$ as outputs from \f$ \eqref{1}\&\eqref{2} \f$.

This manual method is not scalable when the number of equations, variables, and parameters increase.

Using NEML2's model composition capability can address these issues without sacrificing modularity. [ComposedModel](#composedmodel) allows us to compose a new model from the three existing models:
@list-input:models/model_composition/input_composed.i

\note
The names of the other two models are used to specify the coefficients of \f$ \eqref{3} \f$, i.e. `coefficients = 'eq2 eq1'`. This syntax is different from what was covered in the [previous tutorial](#tutorials-models-model-parameters) on model parameters and will be explained in more details in the [next tutorial](#tutorials-models-model-parameters-revisited).

Let us first inspect the composed model and compare it against the three sub-models:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:models/model_composition/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:python:models/model_composition/ex4.py

  Output:
  @list-output:ex4

</div>

Note that the composed model "eq" automatically:
- Identified the input variables \f$ \boldsymbol{a} \f$ and \f$ \boldsymbol{b} \f$,
- Identified the output variable \f$ \dot{\boldsymbol{b}} \f$,
- Registered the parameters of \f$ \eqref{3} \f$ as input variables, and
- Sorted out the evaluation order.

The composed model can be evaluated in the same way as regular models:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:models/model_composition/ex5.cxx

  Output:
  @list-output:ex5
- <b class="tab-title">Python</b>
  @list:python:models/model_composition/ex6.py

  Output:
  @list-output:ex6

</div>

@insert-page-navigation
