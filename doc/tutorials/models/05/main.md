# Tutorial 5: Model composition {#tutorials-models-05}

[TOC]

## Problem description

We have been working with the linear, isotropic elasticity model in the previosu tutorials. We started with that example because it is arguably the simplest possible material model in the context of solid mechanics. It is simple not just because of the simplicity in the description of the material behavior, but also due to the fact that its mathematical formulation only involves one equation.

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

  \dot{\bar{a}} & = c_{aa} \bar{a} + c_{ab} b, \label{2} \\
  \dot{\boldsymbol{a}} & =
  \dot{b} & = c_{ba} \bar{a} + c_{bb} b, \label{3}
\f}
where \f$ \boldsymbol{a} \f$ is a symmetric second order tensor, and \f$ b \f$ is a scalar.

## Writing the input file

Let us first search for available models describing this set of equations:
- \f$ \eqref{1} \f$ corresponds to [SR2Invariant](#sr2invariant);
- \f$ \eqref{2} \f$ and \f$ \eqref{3} \f$ both correspond to [LinearCombination](#scalarlinearcombination).

The input file then looks like
```
c_aa = 1
c_ab = 2
c_ba = 3
c_bb = 4

[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'state/a'
    invariant = 'state/a_bar'
    invariant_type = I1
  []
  [eq2]
    type = ScalarLinearCombination
    from_var = 'state/a_bar state/b'
    to_var = 'state/a_rate'
    coefficients = '${c_aa} ${c_ab}'
    coefficient_as_parameter = true
  []
  [eq3]
    type = ScalarLinearCombination
    from_var = 'state/a_bar state/b'
    to_var = 'state/b_rate'
    coefficients = '${c_ba} ${c_bb}'
    coefficient_as_parameter = true
  []
[]
```

## Running the models

Now that all three models are defined in the input file, we can use the following code to load each of them and evaluate them in sequence.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Scalar.h"
  #include "neml2/tensors/SR2.h"

  using namespace neml2;

  int
  main()
  {
    load_input("input.i");
    auto & model1 = get_model("eq1");
    auto & model2 = get_model("eq2");
    auto & model3 = get_model("eq3");

    // Create the input variables
    auto a_name = VariableName("state", "a");
    auto a = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);
    auto b_name = VariableName("state", "b");
    auto b = Scalar::full(1.1);

    // Evaluate the first model to get a_bar
    auto output1 = model1.value({{a_name, a}, {b_name, b}});
    auto a_bar_name = VariableName("state", "a_bar");
    auto & a_bar = output1[a_bar_name];

    // Evaluate the second model to get a_rate
    auto output2 = model2.value({{a_bar_name, a_bar}, {b_name, b}});
    auto a_rate_name = VariableName("state", "a_rate");
    auto & a_rate = output2[a_rate_name];

    // Evaluate the third model to get b_rate
    auto output3 = model3.value({{a_bar_name, a_bar}, {b_name, b}});
    auto b_rate_name = VariableName("state", "b_rate");
    auto & b_rate = output3[b_rate_name];

    std::cout << "a_rate: \n" << a_rate << std::endl;
    std::cout << "b_rate: \n" << b_rate << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach_output:src1
  ```
- <b class="tab-title">Python</b>
  @source:src2
  ```python
  import neml2
  from neml2.tensors import Scalar, SR2

  neml2.load_input("input.i")
  model1 = neml2.get_model("eq1")
  model2 = neml2.get_model("eq2")
  model3 = neml2.get_model("eq3")

  # Create the input variables
  a = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
  b = Scalar.full(1.1)

  # Evaluate the first model to get a_bar
  a_bar = model1.value({"state/a": a})["state/a_bar"]

  # Evaluate the second model to get a_rate
  a_rate = model2.value({"state/a_bar": a_bar, "state/b": b})["state/a_rate"]

  # Evaluate the third model to get b_rate
  b_rate = model3.value({"state/a_bar": a_bar, "state/b": b})["state/b_rate"]

  print("a_rate:")
  print(a_rate)
  print("b_rate:")
  print(b_rate)
  ```
  @endsource

  Output:
  ```
  @attach_output:src2
  ```

</div>

## Stitching the models together

While we are able to successfully calculate \f$ \dot{\bar{a}} \f$

<div class="section_buttons">

| Previous                           |                               Next |
| :--------------------------------- | ---------------------------------: |
| [Tutorial 4](#tutorials-models-04) | [Tutorial 6](#tutorials-models-06) |

</div>
