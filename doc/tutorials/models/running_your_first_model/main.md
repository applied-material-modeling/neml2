@insert-title:tutorials-models-running-your-first-model

[TOC]

## Problem description

Let us start with the simplest example for solid mechanics. Consider a solid material whose elastic behavior (mapping from strain \f$ \boldsymbol{\varepsilon} \f$ to stress \f$ \boldsymbol{\sigma} \f$, or vice versa) can be described as
\f[
  \boldsymbol{\sigma} = 3 K \operatorname{vol} \boldsymbol{\varepsilon} + 2 G \operatorname{dev} \boldsymbol{\varepsilon},
\f]
where \f$ K \f$ is the bulk modulus, and \f$ G \f$ is the shear modulus.

## Searching for available models

All available material models are listed in the [syntax documentation](@ref syntax-models). The documentation of each model provides a brief description, followed by a list of input file options. Each option has a short description right next to it, and can be expanded to show additional details.

There is an existing model that solves this exact problem: [LinearIsotropicElasticity](#linearisotropicelasticity). The syntax documentation lists the input file options associated with this model.

## Writing the input file

As explained in the syntax documentation for [LinearIsotropicElasticity](#linearisotropicelasticity), the option "strain" is used to specify the name of the variable for the elastic strain, and the option "stress" is used to specify the name of the variable for the stress. The options "coefficients" and "coefficient_types" are used to specify the values of the parameters, in this case \f$ K \f$ and \f$ G \f$.

Using these information, the input file for constructing this model can be composed as:
```
[Models]
  [my_model]
    type = LinearIsotropicElasticity
    strain = 'forces/E'
    stress = 'state/S'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e5 7.8e4'
  []
[]
```

## Choosing the frontend

There are three common ways of interacting with NEML2 input files:
- Calling the appropriate APIs in a C++ program
- Calling the appropriate APIs in a Python script
- Using the NEML2 Runner

These methods are discussed in the [getting started guide](#tutorials-getting-started). In this set of tutorials, the C++ example code and the Python script example code are shown side-by-side with tabs, and in most cases the C++ APIs and the Python APIs have a nice one-to-one correspondance.


## Loading a model from the input file

The following code parses the given input file named "input.i" and retrieves a Model named "my_model". Once retrieved, we can print out a summary of the model by streaming it to the console:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"

  int
  main()
  {
    using namespace neml2;

    auto & model = load_model("input.i", "my_model");
    std::cout << model << std::endl;
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

  model = neml2.load_model("input.i", "my_model")
  print(model)
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

The summary includes information about the model's name, primary floating point numeric type (denoted as "Dtype"), current device, input variables, output variables, parameters, and buffers (if any). Note that the variables and parameters are additionally marked with tensor types surrounded by square brackets, i.e., `[SR2]` and `[Scalar]`. These are NEML2's primitive tensor types which will be extensively discussed in another set of [tutorials](#tutorials-tensors).

## Model structure and forward operators

Before going over model evaluation, let us zoom out from this particular example and briefly discuss the structure of NEML2 models.

All NEML2 models, including this simple elasticity model under consideration, take the following general form
\f[
  y = f(x; p, b),
\f]
where \f$ x \f$ and \f$ y \f$ are respectively sets of input and output variables, \f$ p \f$ is the set of parameters, and \f$ b \f$ is the set of buffers. The utilities of parameters and buffers will be discussed in another [tutorial](#tutorials-models-model-parameters). The forward operator \f$ f \f$ is responsible for mapping from input variables \f$ x \f$ to \f$ y \f$. NEML2 provides three forward operators for all models:
- neml2::Model::value calculates the output variables, i.e., \f$ y = f(x; p, b) \f$.
- neml2::Model::dvalue calculates the derivatives of the output variables with respect to the input variables, i.e., \f$ \pdv{y}{x} = \pdv{f(x; p, b)}{x} \f$.
- neml2::Model::value_and_dvalue calculates both the output variables and their derivatives.

All three forward operators take a map/dictionary of variable values as input and return the requested output variables and/or their derivatives.

In addition to these standard forward operators, some models in NEML2 also support calculating second derivatives. Three additional forward operators are provided to request second derivatives:
- neml2::Model::d2value
- neml2::Model::dvalue_and_d2value
- neml2::Model::value_and_dvalue_and_d2value

## Evaluating the model

Model evaluation consists of two simple steps:
1. Specify the input variable values
2. Call the appropriate forward operator

In this example, the elasticity model can be evaluated using the following code:

\note
Note that `set_default_dtype(kFloat64)` is used to change the default precision to double precision.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  int
  main()
  {
    using namespace neml2;

    set_default_dtype(kFloat64);
    auto & model = load_model("input.i", "my_model");

    // Create the strain
    auto strain_name = VariableName("forces", "E");
    auto strain = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);

    // Evaluate the model
    auto output = model.value({{strain_name, strain}});

    // Get the stress
    auto stress_name = VariableName("state", "S");
    auto & stress = output[stress_name];

    std::cout << "strain: \n" << strain << std::endl;
    std::cout << "stress: \n" << stress << std::endl;
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
  import torch

  torch.set_default_dtype(torch.double)
  model = neml2.load_model("input.i", "my_model")

  # Create the strain
  strain = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)

  # Evaluate the model
  output = model.value({"forces/E": strain})

  # Get the stress
  stress = output["state/S"]

  print("strain:")
  print(strain)
  print("stress:")
  print(stress)
  ```
  @endsource

  Output:
  ```
  @attach-output:src4
  ```

</div>

@insert-page-navigation
