# Tutorial 1: Writing and running your first input file {#tutorials-input-01}

[TOC]

## Problem description

Let us start with the simplest example for solid mechanics. Consider a solid material whose elastic behavior (mapping from strain \f$ \boldsymbol{\varepsilon} \f$ to stress \f$ \boldsymbol{\sigma} \f$, or vice versa) can be described as
\f[
  \boldsymbol{\sigma} = 3 K \operatorname{vol} \boldsymbol{\varepsilon} + 2 G \operatorname{dev} \boldsymbol{\varepsilon}
\f]
where \f$ K \f$ is the bulk modulus, and \f$ G \f$ is the shear modulus.

## Searching among available models

All available material models are listed in the [syntax documentation](@ref syntax-models). The documentation of each model provides a brief description, followed by a list of input file options. Each option has a short description right next to it, and can be expanded to show additional details.

There is an existing model that solves this exact problem: [LinearIsotropicElasticity](#linearisotropicelasticity). The syntax documentation lists the input file options associated with this model.

## Writing the input file

As explained in the syntax documentation for [LinearIsotropicElasticity](#linearisotropicelasticity), the option "strain" is used to specify the name of the variable for the elastic strain, and the option "stress" is used to specify the name of the variable for the stress. The options "coefficients" and "coefficient_types" are used to specify the values of the parameters, in this case \f$ K \f$ and \f$ G \f$.

Related readings:
- [Input file syntax](#input-file)
- [Naming conventions](#naming-conventions)

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

These methods are discussed in the [user guide](#user-getting-started). In this set of tutorials, the C++ example code and the Python script example code are shown side-by-side with tabs, and in most cases the C++ APIs and the Python APIs have a nice one-to-one correspondance.


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
    auto & model = neml2::load_model("input.i", "my_model");
    std::cout << model << std::endl;
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

  model = neml2.load_model("input.i", "my_model")
  print(model)
  ```
  @endsource

  Output:
  ```
  @attach_output:src2
  ```

</div>

The summary includes information about the model's name, primary floating point numeric type (denoted as "Dtype"), current device, input variables, output variables, parameters, and buffers (if any). Note that the variables and parameters are additionally marked with tensor types surrounded by square brackets, i.e., `[SR2]` and `[Scalar]`. These are NEML2's primitive tensor types which will be extensively discussed in another set of [tutorials](#tutorials-tensor).

## Model structure

Before going over model evaluation, let us zoom out from this particular example and briefly discuss the structure of NEML2 models.

All NEML2 models, including this simple elasticity model under consideration, take the following general form
\f[
  y = f(x; p, b)
\f]


<div class="section_buttons">

| Previous                                     |                              Next |
| :------------------------------------------- | --------------------------------: |
| [Working with input files](#tutorials-input) | [Tutorial 2](#tutorials-input-02) |

</div>
