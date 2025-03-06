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
- Parameters can be (recursively) defined by other models, whereas buffers cannot.

In summary, a parameter is a more powerful superset of a buffer.  However, there is overhead cost associated with maintaining a parameter that buffers avoid.

\note
Some models allow users to choose whether to declare data as parameters or buffers.

## Retrieving the parameter value

All model parameters are associated with a unique name, either predefined by the model itself or chosen by the user. The following code iterates through all parameters in the model and print out their values:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Tensor.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "my_model");

    for (auto && [pname, pval] : model.named_parameters())
      std::cout << pname << ":\n" << Tensor(*pval) << std::endl;
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

  for pname, pval in model.named_parameters().items():
    print("{}:".format(pname))
    print(pval.tensor())
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

neml2::Model::get_parameter can be used to retrieve a specific parameter given its name. In other words, the above code is equivalent to

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Tensor.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "my_model");

    auto & G = model.get_parameter("G");
    auto & K = model.get_parameter("K");

    std::cout << "G:\n" << Tensor(G) << std::endl;
    std::cout << "K:\n" << Tensor(K) << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src3
  ```
- <b class="tab-title">Python</b>
  In Python, model parameters can be more conveniently retrieved as attributes, i.e.

  @source:src4
  ```python
  import neml2

  model = neml2.load_model("input.i", "my_model")

  G = model.G
  K = model.K

  print("G:")
  print(G.tensor())
  print("K:")
  print(K.tensor())
  ```
  @endsource

  Output:
  ```
  @attach-output:src4
  ```

</div>

## Updating the parameter value

Model parameters can always be changed by changing the input file. However, in certain cases (e.g., training and optimization), the parameter values should preferrably be updated at runtime (e.g., after each epoch or optimization iteration).

The neml2::Model::set_parameter and neml2::Model::set_parameters methods can be used for that purpose:


<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src5
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Tensor.h"
  #include "neml2/tensors/Scalar.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "my_model");

    // Before modification
    auto & G = model.get_parameter("G");
    std::cout << "G (before modification):\n" << Tensor(G) << std::endl;

    // After modification
    G = Scalar::full(59000);
    std::cout << "G (after modification):\n" << Tensor(G) << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src5
  ```
- <b class="tab-title">Python</b>
  @source:src6
  ```python
  import neml2
  from neml2.tensors import Scalar

  model = neml2.load_model("input.i", "my_model")

  # Before modification
  print("G (before modification):")
  print(model.G.tensor())

  # After modification
  model.G = Scalar.full(59000)
  print("G (after modification):")
  print(model.G.tensor())
  ```
  @endsource

  Output:
  ```
  @attach-output:src6
  ```

</div>

@insert-page-navigation
