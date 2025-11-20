@insert-title:tutorials-models-model-parameters

[TOC]

## Problem description

In this tutorial, we will revisit the problem defined in the previous tutorial and demonstrate how to work with model parameters.

Recall that the linear elasticity material model can be mathematically written as
\f[
  \boldsymbol{\sigma} = 3 K \operatorname{vol} \boldsymbol{\varepsilon} + 2 G \operatorname{dev} \boldsymbol{\varepsilon}.
\f]
Also recall that all NEML2 models can be written in the following general form
\f[
  y = f(x; p, b).
\f]
Pattern matching suggests the following set definitions:
\f[
  x = \left\{ \boldsymbol{\varepsilon} \right\}, \quad y = \left\{ \boldsymbol{\sigma} \right\}, \quad p = \left\{ K, G \right\}, \quad b = \varnothing.
\f]

## Parameter vs buffer

Both \f$ K \f$ and \f$ G \f$ are here categorized as model parameters. The major differences between parameters and buffers are
- Parameters are "trainable", whereas buffers are not. NEML2 can use automatic differentiation to calculate the derivative of output variables with respect to the model parameters, but not for the model buffers.
- Parameters can be (recursively) defined by other models, whereas buffers cannot. This feature is discussed in a later tutorial (@ref tutorials-models-model-parameters-revisited).

In summary, parameter is a more powerful superset of buffer.  However, there is overhead cost associated with maintaining a parameter that buffers avoid.

\note
Some models allow users to choose whether to declare data as parameters or buffers.

## Retrieving the parameter value

All model parameters are associated with a unique name, either predefined by the model itself or chosen by the user. The following code iterates through all parameters in the model and print out their values:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:model_parameters/ex2.py

  Output:
  @list-output:ex2

</div>

neml2::Model::get_parameter can be used to retrieve a specific parameter given its name. In other words, the above code is equivalent to

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  In Python, model parameters can be more conveniently retrieved as attributes, i.e.

  @list:python:model_parameters/ex4.py

  Output:
  @list-output:ex4

</div>

## Updating the parameter value

Model parameters can always be changed by changing the input file. However, in certain cases (e.g., training and optimization), the parameter values should preferrably be updated at runtime (e.g., after each epoch or optimization iteration).

The neml2::Model::set_parameter and neml2::Model::set_parameters methods can be used for that purpose:


<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters/ex5.cxx

  Output:
  @list-output:ex5
- <b class="tab-title">Python</b>
  @list:python:model_parameters/ex6.py

  Output:
  @list-output:ex6

</div>

@insert-page-navigation
