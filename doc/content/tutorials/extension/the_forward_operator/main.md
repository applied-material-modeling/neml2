@insert-title:tutorials-extension-the-forward-operator

[TOC]

## Method signature

The base class neml2::Model declares a pure virtual method neml2::Model::set_value for derived classes to define the forward operator. The forward operator is responsible for calculating output variables given input variables, parameters, and buffers. Optionally, the neml2::Model::set_value method can define derivatives and second derivatives, if requested.

The signature of the method is
```cpp
virtual void set_value(bool, bool, bool) = 0;
```
The three boolean arguments denote whether the caller is requesting output variable values, their derivatives, or their second derivatives. Derived classes should therefore override this method, i.e.
```cpp
void set_value(bool, bool, bool) override;
```

## Implementation

By default, the model forward operator is responsible for calculating the output variable values and their first derivatives. Definition of second order derivatives is not required. Such default behavior can be changed by modifying the corresponding input file option in neml2::Model::expected_options, i.e.
```cpp
options.set<bool>("define_values") = true;
options.set<bool>("define_derivatives") = true;
options.set<bool>("define_second_derivatives") = false;
```
In other words, the default configuration guarantees the third boolean argument of neml2::Model::set_value to always be false, while at least one of the first two boolean arguments is true.

Recall that the equation for this model is
\f[
  \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}.
\f]
The forward operator can be implemented as
@list:cpp:the_forward_operator/ex1.cxx:forward

## Evaluation

The model definition is said to be *complete* once all components are properly defined. Custom models can be evaluated in the same way as existing models that come with NEML2.

@list:cpp:the_forward_operator/ex1.cxx:main

Output:
@list-output:ex1

@insert-page-navigation
