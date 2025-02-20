# Loading the input file {#tutorials-input-02}

[TOC]

## Choosing the frontend

There are three common ways of interacting with NEML2 input files:
- Calling the appropriate APIs in a C++ program
- Calling the appropriate APIs in a Python script
- Using the NEML2 Runner

These methods are discussed in the [user guide](#user-getting-started). In the following tutorials, the C++ example code and the Python script example code are shown side-by-side with tabs, and in most cases the C++ APIs and the Python APIs have a nice one-to-one correspondance.

## The input file

Suppose the input file we created from the previous tutorial is stored in the working directory with name "input.i". Recall that the input file has the following content:
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

## Loading the input file and retrieve the model

The following code parses the given input file named "input.i" and retrieves a Model named "my_model".

<div class="tabbed">

- <b class="tab-title">C++</b>
  ```cpp
  #include "neml2/models/Model.h"

  int
  main()
  {
    auto & model = neml2::load_model("input.i", "my_model");
  }
  ```
- <b class="tab-title">Python</b>
  ```python
  import neml2

  model = neml2.load_model("input.i", "my_model")
  ```

</div>

## Inspecting the model

Once retrieved, we can print out a summary of the model using the following code:

<div class="tabbed">

- <b class="tab-title">C++</b>
  ```cpp
  #include "neml2/models/Model.h"

  int
  main()
  {
    auto & model = neml2::load_model("input.i", "my_model");
    std::cout << model << std::endl;
  }
  ```

  Output:
  ```
  @attach_output@
  ```
- <b class="tab-title">Python</b>
  ```python
  import neml2

  model = neml2.load_model("input.i", "my_model")
  print(model)
  ```

  Output:
  ```
  @attach_output@
  ```
</div>

The summary includes information about the model's name, primary floating point numeric type (denoted as "Dtype"), current device, input variables, output variables, parameters, and buffers (if any). Note that the variables and parameters are additionally marked with tensor types surrounded by square brackets, i.e., `[SR2]` and `[Scalar]`. These are NEML2's primitive tensor types which will be extensively discussed in another set of [tutorials](#tutorials-tensor).


<div class="section_buttons">

| Previous                          |                              Next |
| :-------------------------------- | --------------------------------: |
| [Tutorial 1](#tutorials-input-01) | [Tutorial 3](#tutorials-input-03) |

</div>
