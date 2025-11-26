@insert-title:tutorials-models-model-parameters-revisited

[TOC]

## Problem description

In a [previous tutorial](#tutorials-models-model-parameters), we briefly explained the usage of model parameters and their retrieval and update. In this tutorial, we will introduce other ways of specifying model parameters in the input file and how they facilitate model composition.

Consider the following equations describing the stress-strain relation accounting for a thermal eigenstrain:
\f{align}
  \boldsymbol{\varepsilon}^g & = \alpha (T - T_0) \boldsymbol{I}, \label{1} \\
  \boldsymbol{\varepsilon}^e & = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^g, \label{2} \\
  \boldsymbol{\sigma} & = 3 K \operatorname{vol} \boldsymbol{\varepsilon}^e + 2 G \operatorname{dev} \boldsymbol{\varepsilon}^e, \label{3}
\f}
where \f$ \alpha \f$ is the coefficient of thermal expansion, and \f$ T_0 \f$ is the reference temperature at which the thermal eigenstrain is zero.

## Parameter specification

In NEML2 input files, model parameters can be specified in one of three ways:
1. As plain numeric literals
2. Using the name of a tensor
3. Using a *variable specifier*

### Plain numeric literal

In this mode, parameter values are directly specified by plain numbers. The corresponding input file looks like
@list-input:model_parameters_revisited/input1.i

As usual, we can inspect the structure of the composed model using the following code.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters_revisited/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:model_parameters_revisited/ex2.py

  Output:
  @list-output:ex2

</div>

The composed model has two input variables (strain and temperature), one output variable (stress), and three parameters:
- `eq1_alpha`: Coefficient of thermal expansion defined in `eq1`.
- `eq3_G`: Shear modulus defined in `eq3`.
- `eq3_K`: Bulk modulus defined in `eq3`.

Note that the parameters of the composed model is defined by the *union* of the sub-model parameters, and that their names are prefixed by the name of the sub-model.

### Tensor name

While specifying parameter values as plain numeric literals is convenient and expressive, it is however not always possible to do so. For example, when the parameter value is *batched*, it is not possible to fully define it using one number. In such scenario, a tensor can be created under the `[Tensors]` section, and the name of that tensor can be used in place of the parameter value.

The following example input file shows how to specify the coefficient of thermal expansion as a batched scalar:
@list-input/model_parameters_revisited/input2.i

In this input file, a tensor named "alpha" with batch shape `(2, 2)` is created under the "[Tensors]" section. The coefficient of thermal expansion in model "eq1" then references that tensor using its name, i.e., "CTE = 'alpha'". Note that this approach does not alter the model structure, i.e., `eq1_alpha` is still a model parameter, albeit initialized with the tensor value.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters_revisited/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:python:model_parameters_revisited/ex4.py

  Output:
  @list-output:ex4

</div>

### Variable specifier

While specifying parameters using tensor names is significantly more flexible than using plain numeric literals, the parameter values still remain "static", i.e., the parameter values remain the same during model evaluation unless explicitly updated by the user. In practice, especially when dealing with multiple physics phenomena such as thermo-mechanical coupling, this limitation makes it difficult to specify temperature-dependent parameters.

To remove this limitation, NEML2 allows the use of *variable specifiers* to define model parameters. Suppose the original problem is modified to account for temperature-dependent parameters, e.g.,
\f{align*}
  \boldsymbol{\varepsilon}^g & = \textcolor{blue}{\alpha(T)} (T - T_0) \boldsymbol{I}, \\
  \boldsymbol{\varepsilon}^e & = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^g, \\
  \boldsymbol{\sigma} & = 3 \textcolor{blue}{K(T)} \operatorname{vol} \boldsymbol{\varepsilon}^e + 2 \textcolor{blue}{G(T)} \operatorname{dev} \boldsymbol{\varepsilon}^e,
\f}
where \f$ \alpha(T) \f$, \f$ K(T) \f$, and \f$ G(T) \f$ are linearly interpolated from given data. The input file can be modified accordingly:
@list-input:model_parameters_revisited/input3.i

Here, the three temperature-dependent model parameters are defined using [ScalarLinearInterpolation](#scalarlinearinterpolation), and their corresponding model names are used to specify model parameters in "eq1" and "eq3". It is important to understand that **specifying model parameters by variable coupling alters the model structure**. Mathematically, this corresponds to changing
\f[
  \boldsymbol{\sigma} = f(\boldsymbol{\varepsilon}, T; \alpha, K, G)
\f]
to
\f[
  \boldsymbol{\sigma} = f(\boldsymbol{\varepsilon}, T; \mathcal{P}_\alpha, \mathcal{P}_K, \mathcal{P}_G)
\f]
where \f$ \mathcal{P}_\alpha \f$, \f$ \mathcal{P}_K \f$, and \f$ \mathcal{P}_G \f$ denote the parametrizations of \f$ \alpha \f$, \f$ K \f$, and \f$ G \f$, respectively. In our example, these parametrizations are the abscissa and ordinate of the interpolant.

The composed model automatically reflects such restructuring:
<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters_revisited/ex5.cxx

  Output:
  @list-output:ex5
- <b class="tab-title">Python</b>
  @list:python:model_parameters_revisited/ex6.py

  Output:
  @list-output:ex6

</div>

When the referenced model has more than one output variable, the variable specification becomes ambiguous. A more precise variable specifier in the form of `<model-name>.<variable-name>` can be used to remove such ambiguity. For example, although unncessary (as [ScalarLinearInterpolation](#scalarlinearinterpolation) defines one and only one output variable), the above example input is equivalent to
```
[Models]
  # ...
  [eq1]
    type = ThermalEigenstrain
    reference_temperature = '300'
    CTE = 'alpha.parameters/alpha'
    eigenstrain = 'forces/Eg'
  []
  # ...
[]
```

Finally, a special form of variable specifier can be used to effectively transform a model parameter into an **input variable**. For example, the following input file converts the coefficient of thermal expansion into an input variable.
@list-input:model_parameters_revisited/input4.i

Again, NEML2 automatically reflects such change in model structure:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:model_parameters_revisited/ex7.cxx

  Output:
  @list-output:ex7
- <b class="tab-title">Python</b>
  @list:python:model_parameters_revisited/ex8.py

  Output:
  @list-output:ex8

</div>

@insert-page-navigation
