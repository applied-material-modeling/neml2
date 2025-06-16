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
- The plastic flow associative.
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
```
[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'forces/a'
    invariant = 'state/a_bar'
    invariant_type = I1
  []
  [eq2]
    type = SR2Invariant
    tensor = 'state/b'
    invariant = 'state/b_bar'
    invariant_type = VONMISES
  []
  [eq3]
    type = SR2LinearCombination
    from_var = 'forces/a state/b'
    to_var = 'state/b_rate'
    coefficients = '1 1'
    coefficient_as_parameter = true
  []
[]
```

## Running the models: the hard way

Now that all three models are defined in the input file, we can load and evaluate them in sequence, with a bit of effort:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);
    auto factory = load_input("input.i");
    auto eq1 = factory->get_model("eq1");
    auto eq2 = factory->get_model("eq2");
    auto eq3 = factory->get_model("eq3");

    // Create the input variables
    auto a_name = VariableName("forces", "a");
    auto b_name = VariableName("state", "b");
    auto a = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);
    auto b = SR2::fill(100, 20, 10, 5, -30, -20);

    // Evaluate the first model to get a_bar
    auto a_bar_name = VariableName("state", "a_bar");
    auto a_bar = eq1->value({{a_name, a}})[a_bar_name];

    // Evaluate the second model to get b_bar
    auto b_bar_name = VariableName("state", "b_bar");
    auto b_bar = eq2->value({{b_name, b}})[b_bar_name];

    // Evaluate the third model to get b_rate
    eq3->set_parameter("c_0", b_bar);
    eq3->set_parameter("c_1", a_bar);
    auto b_rate_name = VariableName("state", "b_rate");
    auto b_rate = eq3->value({{a_name, a}, {b_name, b}})[b_rate_name];

    std::cout << "b_rate: \n" << b_rate << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src1
  ```
- <b class="tab-title">Python</b>
  @source:src2
  ```python
  import neml2
  from neml2.tensors import SR2
  import torch

  torch.set_default_dtype(torch.double)
  factory = neml2.load_input("input.i")
  eq1 = factory.get_model("eq1")
  eq2 = factory.get_model("eq2")
  eq3 = factory.get_model("eq3")

  # Create the input variables
  a = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
  b = SR2.fill(100, 20, 10, 5, -30, -20)

  # Evaluate the first model to get a_bar
  a_bar = eq1.value({"forces/a": a})["state/a_bar"]

  # Evaluate the second model to get b_bar
  b_bar = eq2.value({"state/b": b})["state/b_bar"]

  # Evaluate the third model to get b_rate
  eq3.c_0 = b_bar
  eq3.c_1 = a_bar
  b_rate = eq3.value({"forces/a": a, "state/b": b})["state/b_rate"]

  print("b_rate:")
  print(b_rate)
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

## Running the models: the easy way

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
```
[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'forces/a'
    invariant = 'state/a_bar'
    invariant_type = I1
  []
  [eq2]
    type = SR2Invariant
    tensor = 'state/b'
    invariant = 'state/b_bar'
    invariant_type = VONMISES
  []
  [eq3]
    type = SR2LinearCombination
    from_var = 'forces/a state/b'
    to_var = 'state/b_rate'
    coefficients = 'eq2 eq1'
    coefficient_as_parameter = true
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
```

\note
The names of the other two models are used to specify the coefficients of \f$ \eqref{3} \f$, i.e. `coefficients = 'eq2 eq1'`. This syntax is different from what was covered in the [previous tutorial](#tutorials-models-model-parameters) on model parameters and will be explained in more details in the [next tutorial](#tutorials-models-model-parameters-revisited).

Let us first inspect the composed model and compare it against the three sub-models:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/models/Model.h"

  int
  main()
  {
    using namespace neml2;

    auto factory = load_input("input_composed.i");
    auto eq1 = factory->get_model("eq1");
    auto eq2 = factory->get_model("eq2");
    auto eq3 = factory->get_model("eq3");
    auto eq = factory->get_model("eq");

    std::cout << "eq1:\n" << *eq1 << std::endl << std::endl;
    std::cout << "eq2:\n" << *eq2 << std::endl << std::endl;
    std::cout << "eq3:\n" << *eq3 << std::endl << std::endl;
    std::cout << "eq:\n" << *eq << std::endl << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src3
  ```
- <b class="tab-title">Python</b>
  @source:src4
  ```python
  import neml2
  from neml2.tensors import SR2

  factory = neml2.load_input("input_composed.i")
  eq1 = factory.get_model("eq1")
  eq2 = factory.get_model("eq2")
  eq3 = factory.get_model("eq3")
  eq = factory.get_model("eq")

  print("eq1:")
  print(eq1, "\n")
  print("eq2:")
  print(eq2, "\n")
  print("eq3:")
  print(eq3, "\n")
  print("eq:")
  print(eq, "\n")
  ```
  @endsource

  Output:
  ```
  @attach-output:src4
  ```

</div>

Note that the composed model "eq" automatically:
- Identified the input variables \f$ \boldsymbol{a} \f$ and \f$ \boldsymbol{b} \f$,
- Identified the output variable \f$ \dot{\boldsymbol{b}} \f$,
- Registered the parameters of \f$ \eqref{3} \f$ as input variables, and
- Sorted out the evaluation order.

The composed model can be evaluated in the same way as regular models:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src5
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);
    auto eq = load_model("input_composed.i", "eq");

    // Create the input variables
    auto a_name = VariableName("forces", "a");
    auto b_name = VariableName("state", "b");
    auto a = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);
    auto b = SR2::fill(100, 20, 10, 5, -30, -20);

    // Evaluate the composed model
    auto b_rate_name = VariableName("state", "b_rate");
    auto b_rate = eq->value({{a_name, a}, {b_name, b}})[b_rate_name];

    std::cout << "b_rate: \n" << b_rate << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src5
  ```
- <b class="tab-title">Python</b>
  @source:src6
  ```python
  import neml2
  from neml2.tensors import SR2
  import torch

  torch.set_default_dtype(torch.double)
  eq = neml2.load_model("input_composed.i", "eq")

  # Create the input variables
  a = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
  b = SR2.fill(100, 20, 10, 5, -30, -20)

  # Evaluate the composed model
  b_rate = eq.value({"forces/a": a, "state/b": b})["state/b_rate"]

  print("b_rate:")
  print(b_rate)
  ```
  @endsource

  Output:
  ```
  @attach-output:src6
  ```

</div>

@insert-page-navigation
