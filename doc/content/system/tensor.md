# Tensor {#system-tensors}

[TOC]

Refer to [Syntax Documentation](@ref syntax-tensors) for the list of available objects.

## Tensor types {#tensor-types}

Currently, PyTorch is the only supported tensor backend in NEML2. Therefore, all tensor types in NEML2 directly inherit from `ATensor`. In the future, support for other tensor backends may be added, but the public-facing interfaces will remain largely the same.

### Dynamically shaped tensor {#dynamically-shaped-tensor}

neml2::Tensor is a general-purpose *dynamically shaped* tensor type for batched tensors. With a view towards vectorization, the same set of operations can be "simultaneously" applied to a "batch" of tensors. To provide a unified user interface for dealing with such batched operation, NEML2 assumes that the *first* \f$N\f$ dimensions of a tensor are batched dimensions, and the following dimensions are the base dimensions.

> Unlike PyTorch, NEML2 explicitly distinguishes between batch dimensions and base dimensions.

A `Tensor` can be created from a `ATensor` and a batch dimension:
```cpp
Tensor A(at::rand({1, 1, 5, 2}), 2);
```
The batch sizes of `A` is `(1, 1)`:
```cpp
auto batch_sz = A.batch_sizes();
neml2_assert(batch_sz == TensorShape{1, 1});
```
and the base sizes of `A` is `(5, 2)`:
```cpp
auto base_sz = A.base_sizes();
neml2_assert(batch_sz == TensorShape{5, 2});
```

### Statically shaped tensors {#statically-shaped-tensor}

neml2::PrimitiveTensor is the parent class for all tensor types with a *fixed* base shape. It is templated on the base shape of the tensor. NEML2 offers a rich collection of primitive tensor types inherited from `PrimitiveTensor`. Currently implemented primitive tensor types are summarized in the following table.

| Tensor type                            | Base shape        | Description                                                      |
| :------------------------------------- | :---------------- | :--------------------------------------------------------------- |
| [Scalar](@ref neml2::Scalar)           | \f$()\f$          | Rank-0 tensor, i.e. scalar                                       |
| [Vec](@ref neml2::Vec)                 | \f$(3)\f$         | Rank-1 tensor, i.e. vector                                       |
| [R2](@ref neml2::R2)                   | \f$(3,3)\f$       | Rank-2 tensor                                                    |
| [SR2](@ref neml2::SR2)                 | \f$(6)\f$         | Symmetric rank-2 tensor                                          |
| [WR2](@ref neml2::WR2)                 | \f$(3)\f$         | Skew-symmetric rank-2 tensor                                     |
| [R3](@ref neml2::R3)                   | \f$(3,3,3)\f$     | Rank-3 tensor                                                    |
| [SFR3](@ref neml2::SFR3)               | \f$(6,3)\f$       | Rank-3 tensor with symmetry on base dimensions 0 and 1           |
| [R4](@ref neml2::R4)                   | \f$(3,3,3,3)\f$   | Rank-4 tensor                                                    |
| [SFR4](@ref neml2::SFR4)               | \f$(6,3,3\f$)     | Rank-4 tensor with symmetry on base dimensions 0 and 1           |
| [WFR4](@ref neml2::WFR4)               | \f$(3,3,3\f$)     | Rank-4 tensor with skew symmetry on base dimensions 0 and 1      |
| [SSR4](@ref neml2::SSR4)               | \f$(6,6)\f$       | Rank-4 tensor with minor symmetry                                |
| [R5](@ref neml2::R5)                   | \f$(3,3,3,3,3)\f$ | Rank-5 tensor                                                    |
| [SSFR5](@ref neml2::SSFR5)             | \f$(6,6,3)\f$     | Rank-5 tensor with minor symmetry on base dimensions 0-3         |
| [Rot](@ref neml2::Rot)                 | \f$(3)\f$         | Rotation tensor represented in the Rodrigues form                |
| [Quaternion](@ref neml2::Quaternion)   | \f$(4)\f$         | Quaternion                                                       |
| [MillerIndex](@ref neml2::MillerIndex) | \f$(3)\f$         | Crystal direction or lattice plane represented as Miller indices |

Furthermore, all primitive tensor types can be "registered" as variables on a `LabeledAxis`.

## Working with tensors {#working-with-tensors}

### Tensor creation {#tensor-creation}

A factory tensor creation function produces a new tensor. All factory functions adhere to the same schema:
```cpp
<TensorType>::<function_name>(<function-specific-options>, const TensorOptions & options);
```
where `<TensorType>` is the class name of the primitive tensor type listed above, and `<function-name>` is the name of the factory function which produces the new tensor. `<function-specific-options>` are any required or optional arguments a particular factory function accepts. Refer to each tensor type's class documentation for the concrete signature. The last argument `const TensorOptions & options` configures the data type, device, layout and other "meta" properties of the produced tensor. The commonly used meta properties are
- `dtype`: the data type of the elements stored in the tensor. Available options are `kInt8`, `kInt16`, `kInt32`, `kInt64`, `kFloat32`, and `kFloat64`.
- `device`: the compute device where the tensor will be allocated. Available options are `kCPU` and `kCUDA`.
- `requires_grad`: whether the tensor is part of a function graph used by automatic differentiation to track functional relationship. Available options are `true` and `false`.

For example, the following code
```cpp
auto a = SR2::zeros({5, 3},
                    TensorOptions()
                      .device(kCPU)
                      .dtype(kFloat32));
```
creates a statically (base) shaped, dense, single precision tensor of type `SR2` filled with zeros, with batch shape \f$(5, 3)\f$, allocated on the CPU.

### Tensor broadcasting {#tensor-broadcasting}

Quoting Numpy's definition of broadcasting:

> The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.

NEML2's broadcasting semantics is largely the same as those of Numpy and PyTorch. However, since NEML2 explicitly distinguishes between batch and base dimensions, the broadcasting semantics must also be extended. Two NEML2 tensors are said to be _batch-broadcastable_ if iterating backward from the last batch dimension, one of the following is satisfied:
1. Both tensors have the same size on the dimension;
2. One tensor has size 1 on the dimension;
3. The dimension does not exist in one tensor.

_Base-broadcastable_ follows a similar definition. Most binary operators on dynamically shaped tensors, i.e., those of type `Tensor`, require the operands to be both batch- _and_ base-broadcastable. On the other hand, most binary operators on statically base shaped tensors, i.e., those of pritimitive tensor types, only require the operands to be batch-broadcastable.

### Tensor indexing {#tensor-indexing}

In defining the forward operator of a material model, many different tensors representing inputs, outputs, residuals, and Jacobians have to be created, copied, and destroyed on the fly. These operations occupy a significant amount of computing time, especially on GPUs.

To address this challenge, NEML2 creates *views*, instead of copies, of tensors whenever possible. As its name suggests, the view of a tensor is a possibly different interpretation of the underlying data. Quoting the PyTorch documentation:

> For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension, or only span across original dimensions \f$d, d+1, ..., d+k\f$ that satisfy the following contiguity-like condition that \f$\forall i = d,...,d+k-1\f$,
> \f[
> \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]
> \f]
> Otherwise, it will not be possible to view self tensor as shape without copying it.

In NEML2, use neml2::TensorBase::base_index for indexing the base dimensions and neml2::TensorBase::batch_index for indexing the batch dimensions:
```cpp
Tensor A::create({{2, 3, 4}, {-1, -2, 3}, {6, 9, 7}}, 1);
// A = [[  2  3  4]
//      [ -1 -2  3]
//      [  6  9  7]]
Tensor B = A.batch_index({indexing::Slice(0, 2)});
// B = [[  2  3  4]
//      [ -1 -2  3]]
Tensor C = A.base_index({indexing::Slice(1, 3)});
// C = [[  3  4]
//      [ -2  3]
//      [  9  7]]
```
To modify the content of a tensor, use neml2::TensorBase::base_index_put_ or neml2::TensorBase::batch_index_put_:
```cpp
A.base_index_put_({Slice(1, 3)}, at::ones({3, 2}));
// A = [[  2  1  1]
//      [ -1  1  1]
//      [  6  1  1]]
A.batch_index_put_({Slice(0, 2)}, at::zeros({2, 3}));
// A = [[  0  0  0]
//      [  0  0  0]
//      [  6  1  1]]
```
A detailed explanation on tensor indexing APIs is available as part of the official [PyTorch documentation](https://pytorch.org/cppdocs/notes/tensor_indexing.html).

