# C++ Backend {#cpp-backend}

[TOC]

## Loading a model from an input file

The following input file defines a linear isotropic elasticity material model:

```python
[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/E'
    stress = 'state/S'
  []
[]
```

The input file defines two parameters: Young's modulus of 100 and Poisson's ratio of 0.3. While optional, the input file also sets the variable names of strain and stress to be "forces/E" and "state/S", respectively (refer to the [documentation](@ref naming-conventions)).

Assuming the above input file is named "input_file.i", the C++ code snippet below parses the input file and loads the material model (see [external project integration](@ref external-project-integration) on how to integrate NEML2 into another C++ project).

```cpp
#include "neml2/base/Factory.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/batch_stack.h"

int main() {
  auto & model = neml2::load_model("input.i", "model");

  // ...

  return 0;
}
```

## Inspecting model information

Once the model is loaded, it can be streamed to provide a high-level summary of the model, i.e.,
```cpp
  std::cout << model << std::endl;
```
The example model produces the following summary
```
Name:       model
Dtype:      double
Device:     cpu
Input:      forces/E [SR2]
Output:     state/S [SR2]
Parameters: E [Scalar]
            nu [Scalar]
```

## Evaluate the model

Suppose we want to perform 3 material updates simultaneously, the input variables shall have a batch size of 3 (refer to the [tensor system documentation](@ref system-tensors) for more detailed explanation on the term "batch"). The following code constructs the 3 input strains and performs 3 material updates _simultaneously_.

```cpp
  auto strain_name = neml2::VariableName("forces", "E");
  auto stress_name = neml2::VariableName("state", "S");

  auto strain1 = neml2::SR2::fill(0.1, 0.2, 0.3, -0.1, -0.1, 0.2);
  auto strain2 = neml2::SR2::fill(0.2, 0.2, 0.1, -0.1, -0.2, -0.5);
  auto strain3 = neml2::SR2::fill(0.3, -0.2, 0.05, -0.1, -0.3, 0.1);
  auto strain = neml2::batch_stack({strain1, strain2, strain3});

  auto output = model.value({{strain_name, strain}});
  auto stress = output.at(stress_name)
```

The forward operator is invoked using the neml2::Model::value method which takes a map of neml2::Tensor with keys being neml2::VariableName. Other forward operator APIs are available to additionally calculate the first and second derivatives of the output variables with respect to the input variables. There exist a total of 6 variants of the forward operator:

| Method                                     | Output variable values | 1st order derivatives | 2nd order derivatives |
| :----------------------------------------- | :--------------------: | :-------------------: | :-------------------: |
| neml2::Model::value                        |    \f$\checkmark\f$    |                       |                       |
| neml2::Model::dvalue                       |                        |   \f$\checkmark\f$    |                       |
| neml2::Model::d2value                      |                        |                       |   \f$\checkmark\f$    |
| neml2::Model::value_and_dvalue             |    \f$\checkmark\f$    |   \f$\checkmark\f$    |                       |
| neml2::Model::dvalue_and_d2value           |                        |   \f$\checkmark\f$    |   \f$\checkmark\f$    |
| neml2::Model::value_and_dvalue_and_d2value |    \f$\checkmark\f$    |   \f$\checkmark\f$    |   \f$\checkmark\f$    |


## Model parameters

Each model contains zero or more parameters. The parameters are initialized with values specified in the input file. In the above example, the elasticity model contains two parameters: `E` and `nu`. Model parameters can be retrieved using the neml2::Model::get_parameter method, the parameter value can be updated using the assignment operator, and automatic differentiation can be used to track the derivatives w.r.t. a parameter:
```cpp
  auto & E = model.get_parameter("E");
  E = neml2::Scalar::full(200.0);
  E.requires_grad_();
```
Alternatively, the parameter value can be directly updated using the neml2::Model::set_parameter or neml2::Model::set_parameters method:
```cpp
  model.set_parameter("E", neml2::Scalar::full(200.0));
```
```cpp
  model.set_parameters({{"E", neml2::Scalar::full(200.0)},
                        {"nu", neml2::Scalar::full(0.25)}});
```
