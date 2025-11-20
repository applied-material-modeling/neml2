@insert-title:tutorials-optimization-automatic-differentiation

[TOC]

## Model definition

For demonstration purposes, let us go back to an earlier example model -- linear isotropic elasticity:
@list-input:automatic_differentiation/input.i

Recall that both the bulk modulus and the shear modulus are treated as *trainable* parameters, which can be verified by inspecting the model summary:
@list:python:automatic_differentiation/ex1.py

Output:
@list-output:ex1

## Automatic differentiation

NEML2 tensors can be used interchangeably with PyTorch tensors in a seamless fashion. Function graph traced through NEML2 tensor operations can be back-propagated using PyTorch's autograd engine.

For example, the following Python script demonstrates the calculation of \f$\pdv{l}{p}\f$ using an arbitrary, scalar-valued loss function \f$l\f$ obtained by a combination of NEML2 operations and PyTorch operations.

@list:python:automatic_differentiation/ex2.py

Output:
@list-output:ex2

@insert-page-navigation
