# Tutorial 1: Writing your first input file {#tutorials-input-01}

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
```python
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

<div class="section_buttons">

| Previous                                     |                              Next |
| :------------------------------------------- | --------------------------------: |
| [Working with input files](#tutorials-input) | [Tutorial 2](#tutorials-input-02) |

</div>
