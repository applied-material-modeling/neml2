@insert-title:tutorials-tensors-tensor-types

[TOC]

NEML2 tensors are extremely similar to the tensor type provided by its default backend ATen. In fact, all NEML2 tensors inherit from `at::Tensor`, *the* tensor type from ATen. What's different from ATen is that the NEML2 tensor library provides the following enhancements:
- Explicit distinction between *batch* and *base* dimensions
- **Primitive** tensor types commonly used in traditional scientific computing
- Commonly used math operators and functions for primitive tensor types

## Batching

Given a general, \f$n\f$-dimensional tensor (or array), its shape is usually denoted using a tuple consisting of sizes of each of its dimension, i.e.
\f[
  (d_0, d_1, d_2, ..., d_i, ..., d_{n-2}, d_{n-1}), \quad i = 0, 1, 2, ..., n-1,
\f]
where \f$d_i\f$ is the number of components along dimension \f$i\f$.

In NEML2, we explicitly introduce the concept of batching such that the shape of a \f$n\f$-dimensional tensor may now be denoted (with a slight abuse of notation) as
\f{align*}
  (c_0, c_1, c_2, ..., c_i, ..., c_{n_b-2}, c_{n_b-1}\textcolor{blue}{;} d_0, d_1, d_2, ..., d_j, ..., d_{n-n_b-2}, d_{n-n_b-1}), \\
  i = 0, 1, 2, ..., n_b-1, \\
  j = 0, 1, 2, ..., n-n_b-1,
\f}
where \f$n_b \in [0, n]\f$ is the number of batch dimensions, \f$c_i\f$ is the number of components along each batch dimension, and \f$d_j\f$ is the number of components along each base dimension. In NEML2's notation, a separator \f$;\f$ is used to delimit batch and base sizes. Every batched tensor must have one and only one \f$;\f$ delimiter.

For example, given a tensor of shape
\f[
  (3, 100, 5, 13, 2),
\f]
we could assign it a batch dimension of 3, and the resulting shape of the batched tensor becomes
\f[
  (3, 100, 5; 13, 2).
\f]
Similarly, if the batch dimension is 0 or 5, the resulting shape would become
\f{align*}
  (; 3, 100, 5, 13, 2), \\
  (3, 100, 5, 13, 2;).
\f}
It is valid for the \f$;\f$ delimiter to appear at the beginning or the end of the shape tuple.  If it appears at the beginning it means that tensor has no batch dimensions.  If it appears at the end it means the tensor is a batched scalar.

## Why batching?

The definition of batching is fairly straightforward. However, fair question to ask is "why another layer of abstraction on top of the regular tensor shape definition"? Let's briefly consider the high-level motivation behind this design choice.

**Shape ambiguity**: Consider a tensor of shape \f$ (55, 100, 3, 3, 3, 3) \f$, it can be interpreted as
- a scalar with batch shape \f$ (55, 100, 3, 3, 3, 3) \f$,
- a 3-vector with batch shape \f$ (55, 100, 3, 3, 3) \f$,
- a second order tensor with batch shape \f$ (55, 100, 3, 3) \f$,
- a third order tensor with batch shape \f$ (55, 100, 3) \f$, or
- a fourth order tensor with batch shape \f$ (55, 100) \f$.

Such ambiguity can be avoided if the user consistently keeps track of batching throughout the lifetime of all related tensor operations, which is manageable in simple models but difficult to scale if the model grows in complexity. Therefore, in NEML2, we incorporate the management and propagation of this information directly in the tensor library, removing such overhead from user implementation.

**Scalar broadcasting**: With more details to be covered in a [later tutorial](#tutorials-tensors-broadcasting) on broadcasting, explicit batching allows general broadcasting rules between scalar and other primitive tensor types.

**Shape generalization**: As mentioned in the [tutorial](#tutorials-models-just-in-time-compilation) on JIT compilation, NEML2 supports tracing tensor operations into graphs. The PyTorch tracing engine does not support tracing operations on tensor shapes, and so users are left with two options:
- Trace the graph with example input variables having the same shape as real input variables to be used later on.
- Re-trace the graph whenever any input variable has a different shape.

The first option implies a potential memory bound at the stage of JIT compilation, and is difficult to guarantee in practice. The second option imposes a significant runtime overhead whenever re-tracing is necessary.

With explicit batching, NEML2 is able to generate *batch-generalizable* graphs. As long as the batch dimensions of the input variables remain unchanged, the same traced graph can be reused without re-tracing.

## Tensor types

neml2::Tensor is the general-purpose tensor type whose base shape can be modified at runtime.

In addition, NEML2 offers a rich collection of primitive tensor types whose base shapes are static, i.e., remain fixed at runtime. Currently implemented tensor types are summarized in the following table.

| Tensor type                        | Base shape              | Description                                                      |
| :--------------------------------- | :---------------------- | :--------------------------------------------------------------- |
| [Tensor](#neml2::Tensor)           | Dynamic                 | General-purpose tensor with dynamic base shape                   |
| [Scalar](#neml2::Scalar)           | \f$()\f$                | Rank-0 tensor, i.e. scalar                                       |
| [Vec](#neml2::Vec)                 | \f$(3)\f$               | Rank-1 tensor, i.e. vector                                       |
| [R2](#neml2::R2)                   | \f$(3,3)\f$             | Rank-2 tensor                                                    |
| [SR2](#neml2::SR2)                 | \f$(6)\f$               | Symmetric rank-2 tensor                                          |
| [WR2](#neml2::WR2)                 | \f$(3)\f$               | Skew-symmetric rank-2 tensor                                     |
| [R3](#neml2::R3)                   | \f$(3,3,3)\f$           | Rank-3 tensor                                                    |
| [SFR3](#neml2::SFR3)               | \f$(6,3)\f$             | Rank-3 tensor with symmetry on base dimensions 0 and 1           |
| [R4](#neml2::R4)                   | \f$(3,3,3,3)\f$         | Rank-4 tensor                                                    |
| [SFR4](#neml2::SFR4)               | \f$(6,3,3\f$)           | Rank-4 tensor with symmetry on base dimensions 0 and 1           |
| [WFR4](#neml2::WFR4)               | \f$(3,3,3\f$)           | Rank-4 tensor with skew symmetry on base dimensions 0 and 1      |
| [SSR4](#neml2::SSR4)               | \f$(6,6)\f$             | Rank-4 tensor with minor symmetry                                |
| [SWR4](#neml2::SWR4)               | \f$(6,3)\f$             | Rank-4 tensor with minor symmetry then skew symmetry             |
| [WSR4](#neml2::WSR4)               | \f$(3,6)\f$             | Rank-4 tensor with skew symmetry then minor symmetry             |
| [WWR4](#neml2::WWR4)               | \f$(3,3)\f$             | Rank-4 tensor with skew symmetry                                 |
| [R5](#neml2::R5)                   | \f$(3,3,3,3,3)\f$       | Rank-5 tensor                                                    |
| [SSFR5](#neml2::SSFR5)             | \f$(6,6,3)\f$           | Rank-5 tensor with minor symmetry on base dimensions 0-3         |
| [R8](#neml2::R8)                   | \f$(3,3,3,3,3,3,3,3)\f$ | Rank-8 tensor                                                    |
| [SSSSR8](#neml2::SSSSR8)           | \f$(6,6,6,6)\f$         | Rank-8 tensor with minor symmetry                                |
| [Rot](#neml2::Rot)                 | \f$(3)\f$               | Rotation tensor represented in the Rodrigues form                |
| [Quaternion](#neml2::Quaternion)   | \f$(4)\f$               | Quaternion                                                       |
| [MillerIndex](#neml2::MillerIndex) | \f$(3)\f$               | Crystal direction or lattice plane represented as Miller indices |

All primitive tensor types can be declared as variables, parameters, and buffers in a model.

@insert-page-navigation
