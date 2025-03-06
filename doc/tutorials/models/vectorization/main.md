@insert-title:tutorials-models-vectorization

[TOC]

## What is vectorization?

Vectorization refers to *array programming*.  Both SIMD (single instruction, multiple data) operation and SIMT (single instruction, multiple threads) operation are implementations of array programming. These terms all refer to the methods that apply operations to an entire set of values at once.

While the benefit of vectorization depends on the instruction support, instruction throughput, and register size, among other factors, it is generally expected that vectorization can provide a considerable amount of speedup to many operations.  For example, given a device's instruction set with 1024-bit registers. These can be used to operate on 16 operands of 64-bits apiece. More specifically, if we are dealing with double-precision floating point numbers, a single instruction can do 16 operations at once. In other words, ideally, \f$ N \f$ operations can be completed using \f$ N/16 \f$ instructions. And again, ideally, this leads to 16 times faster execution than operations without vectorization.

Although the implementation varies, both CPU and CUDA supports vectorization. NEML2 supports vectorization of model evaluation on both CPU and CUDA.  All NEML2 models ensure that the data being processed, e.g., variables, parameters, buffers, are contiguously allocated, and that the same set of operations are applied to those contiguous chunks of data.

## For loop versus vectorization: A numerical experiment

Given the material model we have been working with in the previous tutorials, how much (wall) time would it take to perform 100,000 forward evaluations?

There are two ways of completing this task:
- using a for loop, or
- using vectorization

### For loop

The following code snippet shows how to use a for loop to perform the \f$ N \f$ evaluations. For convenience, the \f$ N \f$ strains are evenly spaced.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include <torch/cuda.h>
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  using namespace neml2;

  int
  main()
  {
    // Preparation
    Size N = 10;
    auto device = kCPU;
    auto & model = load_model("input.i", "my_model");
    model.to(device);

    // Create the strain on the device
    auto strain_name = VariableName("forces", "E");
    auto strain_min = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device);
    auto strain_max = SR2::fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device);
    auto strain = SR2::linspace(strain_min, strain_max, N);

    // Evaluate the model N times
    for (Size i = 0; i < N; i++)
      auto output = model.value({{strain_name, strain.batch_index({i})}});
  }
  ```
  @endsource
- <b class="tab-title">Python</b>
  @source:src2
  ```python
  import torch
  import neml2
  from neml2.tensors import SR2

  # Preparation
  N = 10
  device = torch.device("cpu")
  model = neml2.load_model("input.i", "my_model")
  model.to(device=device)

  # Create the strain on the device
  strain_min = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device=device)
  strain_max = SR2.fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device=device)
  strain = SR2.linspace(strain_min, strain_max, N)

  # Evaluate the model N times
  for i in range(N):
    output = model.value({"forces/E": strain.batch[i]})
  ```
  @endsource

</div>

### Vectorization

The following code snippet shows how to rely on NEML2 internal vectorization to perform the \f$ N \f$ evaluations.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include <torch/cuda.h>
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  using namespace neml2;

  int
  main()
  {
    // Preparation
    Size N = 10;
    auto device = kCPU;
    auto & model = load_model("input.i", "my_model");
    model.to(device);

    // Create the strain on the device
    auto strain_name = VariableName("forces", "E");
    auto strain_min = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device);
    auto strain_max = SR2::fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device);
    auto strain = SR2::linspace(strain_min, strain_max, N);

    // Evaluate the model N times
    auto output = model.value({{strain_name, strain}});
  }
  ```
  @endsource
- <b class="tab-title">Python</b>
  @source:src4
  ```python
  import torch
  import neml2
  from neml2.tensors import SR2

  # Preparation
  N = 10
  device = torch.device("cpu")
  model = neml2.load_model("input.i", "my_model")
  model.to(device=device)

  # Create the strain on the device
  strain_min = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device=device)
  strain_max = SR2.fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device=device)
  strain = SR2.linspace(strain_min, strain_max, N)

  # Evaluate the model N times
  output = model.value({"forces/E": strain})
  ```
  @endsource

</div>

For a fair comparison, the input variables are allocated prior to model evaluation, and only the model evaluation part is timed. The timing study was performed with \f$ N = \{10, 1000, 100000\} \f$ on both CPU and CUDA. Results are summarized in the following table.

| Device | N      | For loop | Vectorization |
| :----- | :----- | :------- | :------------ |
| CPU    | 10     | 0.061    | 0.003         |
|        | 1000   | 0.232    | 0.004         |
|        | 100000 | 16.782   | 0.041         |
| CUDA   | 10     | 0.145    | 0.004         |
|        | 1000   | 0.337    | 0.004         |
|        | 100000 | 26.176   | 0.005         |

## Remarks

While there are many other factors contributing to the performance difference between the for loop approach and the vectorization approach, we hope this simple experiment serves as a reasonable introduction to the benefit of vectorization.

It is worth noting that, in the above code snippets,
- The pre-allocated strain tensor has a shape of `(N; 6)`. The leading dimension with size N is referred to as a *batch dimension* in NEML2.
- The strain tensor with shape `(N; 6)` can operate with scalar-valued model parameters \f$ K \f$ and \f$ G \f$. This behavior relies on the concept of *broadcasting*.

*Batching* and *broadcasting* are the fundamental mechanisms that allow NEML2 to efficiently vectorize model evaluation while providing users with a unified API for both CPU and CUDA. More details on these mechanisms are discussed in the [tutorials](#tutorials-tensors) on NEML2 tensors.

@insert-page-navigation
