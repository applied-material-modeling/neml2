# Custom Model {#custom-model}

[TOC]

## Model description {#custom-model-model-description}

The following tutorials serve as a developer-facing step-by-step guide for creating and testing a custom material model. A simple linear isotropic hardening is used as the example in this tutorial. The model can be mathematically written as
\f{align*}
  h &= K \bar{\varepsilon}^p,
\f}
where \f$\bar{\varepsilon}^p\f$ is the equivalent plastic strain and \f$h\f$ is the isotropic hardening. The input variable for this model is
\f$\boldsymbol{\varepsilon}\f$, the output variable for this model is \f$k\f$, and the parameters of the model is \f$K\f$.

## Variable declaration {#custom-model-variable-declaration}

The development of every model begins with the declaration and registration of its input and output variables. Here, we first define an abstract base class that will be later used to define the linear isotropic hardening relation. The abstract base class defines the isotropic hardening relation
\f[
  h = f\left( \bar{\varepsilon}^p \right),
\f]
mapping the equivalent plastic strain to the isotropic hardening. The base class is named `IsotropicHardening` following the [naming conventions](@ref naming-conventions). The header file [IsotropicHardening.h](@ref neml2::IsotropicHardening) is displayed below
```cpp
#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class IsotropicHardening : public Model
{
public:
  static OptionSet expected_options();

  IsotropicHardening(const OptionSet & options);

protected:
  // Equivalent plastic strain
  const Variable<Scalar> & _ep;

  // Isotropic hardening
  Variable<Scalar> & _h;
};
} // namespace neml2
```
Since isotropic hardening _is a_ model, the class inherits from `Model`. The user-facing expected options are defined by the static method `expected_options`. NEML2 handles the parsing of user-specified options and pass them to the constructor (see [Input file syntax](@ref input-file-syntax) on how the input file works). The input variable of the model is the equivalent plastic strain, and the output variable of the model is the isotropic hardening. Their corresponding variable value references are stored as `_ep` and `_h`, respectively, again following the [naming conventions](@ref naming-conventions).

The expected options and the constructor are defined as
```cpp
#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
OptionSet
IsotropicHardening::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("equivalent_plastic_strain") = VariableName("state", "internal", "ep");
  options.set<VariableName>("isotropic_hardening") = VariableName("state", "internal", "k");
  return options;
}

IsotropicHardening::IsotropicHardening(const OptionSet & options)
  : Model(options),
    _ep(declare_input_variable<Scalar>("equivalent_plastic_strain")),
    _h(declare_output_variable<Scalar>("isotropic_hardening"))
{
}
} // namespace neml2
```
Recall that variable names on `LabeledAxis` are always fully qualified, the equivalent plastic strain and the isotropic hardening are denoted as "state/internal/ep" and "state/internal/k", respectively. An instance of the class is constructed by extracting user-specified options (of type `OptionSet`). Note how `declare_input_variable<Scalar>` and `declare_output_variable<Scalar>` are used to declare and register the input and output variables.

## Parameter declaration {#custom-model-parameter-declaration}

Now that the abstract base class `IsotropicHardening` has been implemented, we are ready to define our first concrete NEML2 model that describes a linear isotropic hardening relation
\f[
  h = K \bar{\varepsilon}^p.
\f]
Note that \f$K\f$ is a model parameter. Following the naming convention, the concrete class is named `LinearIsotropicHardening`. The header file is displayed below.
```cpp
#pragma once

#include "neml2/models/solid_mechanics/IsotropicHardening.h"

namespace neml2
{
// Simple linear map between equivalent strain and isotropic hardening
class LinearIsotropicHardening : public IsotropicHardening
{
public:
  static OptionSet expected_options();

  LinearIsotropicHardening(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // The linear isotropic hardening modulus
  const Scalar & _K;
};
} // namespace neml2
```
It derives from the abstract base class `IsotropicHardening` and implements the method `set_value` as the forward operator. The model parameter \f$K\f$ is stored as a protected member variable `_K`. The model implementation is shown below.
```cpp
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicHardening);

OptionSet
LinearIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.set<TensorName>("hardening_modulus");
  return options;
}

LinearIsotropicHardening::LinearIsotropicHardening(const OptionSet & options)
  : IsotropicHardening(options),
    _K(declare_parameter<Scalar>("K", "hardening_modulus"))
{
}

void
LinearIsotropicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _h = _K * _ep;

  if (dout_din)
    _h.d(_ep) = _K;

  if (d2out_din2)
  {
    // zero
  }
}
} // namespace neml2
```
Note that an additional option named "hardening_modulus" is requested from the user. The model parameter is registered using the API `declare_parameter<Scalar>`. In the `set_value` method, the current value of the input variable, equivalent plastic strain, is stored in the member `_ep`, and so the isotropic hardening can be computed as
```cpp
_K * _ep
```
The computed result is copied into the model output variable `_h` by
```cpp
_h = _K * _ep;
```
In addition, the first derivative of the forward operator is defined as
```cpp
_h.d(_ep) = _K;
```
Last but not the least, the model is registed in the NEML2 model factory using the macro
```cpp
register_NEML2_object(LinearIsotropicHardening);
```
so that an instance of the class can be created at runtime.

## In-code documentation {#custom-model-in-code-documentation}

In the examples above, we have added Doxygen-style in-code documentation for the class and member variables.
These documentation strings are automatically parsed and translated by Doxygen, and the resulting page can be found at neml2::LinearIsotropicHardening.

On the other hand, as the number of models grows, it becomes increasingly challenging to maintain a list of input file options so that users can refer to. For example, the input file syntax for specifying `LinearIsotropicHardening` is shown below
```python
[Models]
  [my_custom_model]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'state/internal/ep'
    isotropic_hardening = 'state/internal/h'
    hardening_modulus = 1000
  []
[]
```
Some natural questions are, how is the user supposed to know, without inspecting the source code,
- the name of the object (e.g., "LinearIsotropicHardening")
- the object's functionalities (e.g., the underlying mathematical description)
- the option names (e.g., "equivalent_plastic_strain", "isotropic_hardening", and "hardening_modulus")
- that some options are optional, while others are mandatary
- the default values of optional input file options

Doxygen-style in-code documentation cannot directly address the above issues. Therefore, NEML2 provides another form of in-code documentation to generate the so-called Syntax Documentation.

Again, using the `IsotropicHardening` as an example, the following in-code documentation can be added in the `expected_options` method:
```cpp
OptionSet
IsotropicHardening::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Map equivalent plastic strain to isotropic hardening";

  options.set<VariableName>("equivalent_plastic_strain") = VariableName("state", "internal", "ep");
  options.set("equivalent_plastic_strain").doc() = "Equivalent plastic strain";

  options.set<VariableName>("isotropic_hardening") = VariableName("state", "internal", "k");
  options.set("isotropic_hardening").doc() = "Isotropic hardening";

  return options;
}
```

```cpp
OptionSet
LinearIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.doc() += " following a linear relationship, i.e., \\f$ h = K \\varepsilon_p \\f$ where "
                   "\\f$ K \\f$ is the hardening modulus.";

  options.set<TensorName>("hardening_modulus");
  options.set("hardening_modulus").doc() = "Hardening modulus";

  return options;
}
```
The object description is added using neml2::OptionSet::doc, and each expected option is accompanied with a brief explanation. NEML2 can then automatically extract and build the [syntax documentation](#linearisotropichardening).
