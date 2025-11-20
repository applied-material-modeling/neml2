@insert-title:tutorials-optimization-parameter-calibration

[TOC]

## Problem definition

Suppose we are given the experimental data of one pair of strain \f$\boldsymbol{\varepsilon}^*\f$ and stress \f$\boldsymbol{\sigma}^*\f$ measurement of a specimen, and we are tasked to infer the bulk \f$K\f$ and shear \f$G\f$ modulus of the material.

Let us define the loss function as the *distance* between the stress prediction and the stress measurement, i.e.
\f[
  l(K, G) = \norm{\boldsymbol{\sigma}(\boldsymbol{\varepsilon}^*; K, G) - \boldsymbol{\sigma}^*}^2.
\f]
The material parameters, \f$K^*\f$ and \f$G^*\f$, can then be "inferred" by minimizing the loss function
\f[
  \left( K^*, G^* \right) = \mathop{\mathrm{argmin}}\limits_{K, G} \ l.
\f]

## Calibration

The following Python script uses the simple gradient descent algorithm to find \f$K^*\f$ and \f$G^*\f$:
\f{align*}
  K_{i+1} &= K_i - \gamma \pdv{l}{K_i}, \\
  G_{i+1} &= G_i - \gamma \pdv{l}{G_i},
\f}
which iterates until the loss function \f$l\f$ is sufficiently small. A constant learning rate \f$\gamma = 1\f$ is chosen in this example.

@list:python:parameter_calibration/ex1.py

![Material parameter calibration](tutorials/optimization/parameter_calibration/calibration.svg){html: width=85%}

@insert-page-navigation
