@insert-title:tutorials-tensors-tensor-creation

[TOC]

Related reading: [PyTorch tensor creation API](https://pytorch.org/cppdocs/notes/tensor_creation.html)

## Wrapping a torch tensor

A neml2::Tensor can be created from a `torch::Tensor` by marking its batch dimension:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:tensors/tensor_creation/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:tensors/tensor_creation/ex2.py

  Output:
  @list-output:ex2

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
  @list:cpp:tensors/tensor_creation/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:python:tensors/tensor_creation/ex4.py

  Output:
  @list-output:ex4

</div>

\note
All the factory methods are listed [here](@ref tensor_creation).

@insert-page-navigation
