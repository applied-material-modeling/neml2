@insert-title:tutorials-tensors-tensor-creation

[TOC]

Related reading: [PyTorch tensor creation API](https://pytorch.org/cppdocs/notes/tensor_creation.html)

## Wrapping a torch tensor

A neml2::Tensor can be created from a `torch::Tensor` by marking its batch dimension:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include <torch/torch.h>
  #include "neml2/tensors/Tensor.h"

  int
  main()
  {
    // A raw torch tensor
    auto A = torch::rand({5, 3, 2, 8});
    std::cout << "Shape of A: " << A.sizes() << '\n' << std::endl;

    // Mark it with batch dimension of 2
    auto B = neml2::Tensor(A, 2);
    std::cout << "      Shape of B: " << B.sizes() << std::endl;
    std::cout << "Batch shape of B: " << B.batch_sizes() << std::endl;
    std::cout << " Base shape of B: " << B.base_sizes() << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach_output:src1
  ```
- <b class="tab-title">Python</b>
  @source:src2
  ```python
  import torch
  from neml2.tensors import Tensor

  # A raw torch tensor
  A = torch.rand(5, 3, 2, 8)
  print("Shape of A:", A.shape, "\n")

  # Mark it with batch dimension of 2
  B = Tensor(A, 2)
  print("      Shape of B:", B.shape)
  print("Batch shape of B:", B.batch.shape)
  print(" Base shape of B:", B.base.shape)
  ```
  @endsource

  Output:
  ```
  @attach_output:src2
  ```

</div>

## Factory methods

A factory tensor creation function produces a new tensor. All factory functions adhere to the same schema:

<div class="tabbed">
- <b class="tab-title">C++</b>
  ```
  <TensorType>::<function_name>(<function-specific-options>, const neml2::TensorOptions & options);
  ```
- <b class="tab-title">Python</b>
  ```
  <TensorType>.<function_name>(<function-specific-options>, *, dtype, device, requires_grad)
  ```
</div>

where `<TensorType>` is the class name of the primitive tensor type listed [here](#tutorials-tensors-tensor-types), and `<function-name>` is the name of the factory function which produces the new tensor. `<function-specific-options>` are positional arguments a particular factory function accepts. Refer to each tensor type's class documentation for the concrete signature. The last argument `const TensorOptions & options` configures the data type, device, and other "meta" properties of the produced tensor. The commonly used meta properties are
- `dtype`: the data type of the elements stored in the tensor. Available options are `kInt8`, `kInt16`, `kInt32`, `kInt64`, `kFloat32`, and `kFloat64`. Support for unsigned integer types were added in recent versions of PyTorch.
- `device`: the compute device where the tensor will be allocated. Available options are `kCPU` and `kCUDA`. On MacOS, the device type `torch::kMPS` could be used but is not officially supported by NEML2.
- `requires_grad`: whether the tensor is part of a function graph used by automatic differentiation to track functional relationship. Available options are `true` and `false`.

For example, the following code creates a statically (base) shaped, dense, single precision tensor of type `SR2` filled with zeros, with batch shape \f$(5, 3)\f$, allocated on the CPU.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/tensors/SR2.h"

  using namespace neml2;

  int
  main()
  {
    auto A = SR2::zeros({5, 3}, TensorOptions().dtype(kFloat32).device(kCPU));
  }
  ```
  @endsource
- <b class="tab-title">Python</b>
  @source:src4
  ```python
  import torch
  from neml2.tensors import SR2

  A = SR2.zeros((5, 3), dtype=torch.float32, device=torch.device("cpu"))
  ```
  @endsource

</div>

\note
All factory methods are listed [here](@ref tensor_creation).

@insert-page-navigation
