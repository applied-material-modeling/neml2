@insert-title:tutorials-tensors-broadcasting

[TOC]

Related reading: NumPy manual on [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## NEML2's broadcasting rules

NEML's broadcasting rules are built on top of NumPy's general broadcasting rules.

When operating on two tensors, NEML2 compares each dimension's size pair one-by-one, from right to left. Before the comparison, the shapes of two tensors are aligned at the final batch dimension. The comparison starts with the trailing (i.e., rightmost) batch dimension and works its way left. Two batch dimensions are compatible (broadcastable) when
- Their sizes are equal,
- One of them has size one, or
- One of them does not exist.

If all dimensions are compatible, the two tensors are *batch-broadcastable*. Many operations in NEML2 require operands to be batch-broadcastable, and if not, an exception will be thrown (typically only in Debug mode).

The operands do not need to have the same batch dimension (per the third rule) in order to be batch-broadcastable. After broadcasting, each dimension of the operands are effectively expanded to have the same size as larger size of the two.

Note that we have not yet mentioned anything about the broadcasting rules for base dimensions, because the requirement for operands' base shapes largely depend on the operation itself. And for all primitive tensor types, since the base shapes are determined statically, no shape check is needed at runtime.

## Example

The utility of broadcasting is best demonstrated with practical examples.

Suppose we are given two samples made of different materials, and for each sample, we are given a number of strain measurements at different locations. Assuming both materials are homogeneous, we could utilize NEML2's broadcasting rules to perform the constitutive update extremely efficiently.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include <torch/torch.h>
  #include "neml2/tensors/SSR4.h"
  #include "neml2/tensors/SR2.h"
  #include "neml2/tensors/Scalar.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);

    // Number of samples
    Size ns = 2;
    // Number of strain measurements
    Size nm = 1000;

    // Elasticity tensor of the two materials
    auto youngs_modulus = Scalar::create({1e5, 2e5});
    auto poissons_ratio = Scalar::create({0.1, 0.2});
    auto C = SSR4::isotropic_E_nu(youngs_modulus, poissons_ratio);

    // (Fake) strain measurements
    auto strain = SR2(torch::rand({nm, ns, 6}, kFloat64));

    // Perform the constitutive update
    auto stress = C * strain;

    // Do the shapes make sense?
    std::cout << "     Shape of C:" << C.batch_sizes() << C.base_sizes() << std::endl;
    std::cout << "Shape of strain: " << strain.batch_sizes() << strain.base_sizes() << std::endl;
    std::cout << "Shape of stress: " << stress.batch_sizes() << stress.base_sizes() << std::endl;
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
  import torch
  from neml2.tensors import Scalar, SR2, SSR4

  # Number of samples
  ns = 2
  # Number of strain measurements
  nm = 1000

  # Elasticity tensor of the two materials
  youngs_modulus = Scalar(torch.tensor([1e5, 2e5], dtype=torch.float64))
  poissons_ratio = Scalar(torch.tensor([0.1, 0.2], dtype=torch.float64))
  C = SSR4.isotropic_E_nu(youngs_modulus, poissons_ratio)

  # (Fake) strain measurements
  strain = SR2(torch.rand(nm, ns, 6, dtype=torch.float64))

  # Perform the constitutive update
  stress = C * strain

  # Do the shapes make sense?
  print("     Shape of C:", C.batch.shape, C.base.shape)
  print("Shape of strain:", strain.batch.shape, strain.base.shape)
  print("Shape of stress:", stress.batch.shape, stress.base.shape)
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

To better understand broadcasting, let us try to apply the broadcasting rules manually:
1. Align the shapes at the final batch dimension (i.e., align the semicolons):
   ```
   Shape of C:            (2; 6, 6)
   Shape of strain: (1000, 2; 6)
   ```
2. Examine batch sizes from right to left:
   ```
   Shape of C:            (2; 6, 6)
   Shape of strain: (1000, 2; 6)
                           ▲
   ```
   The sizes are equal -- the two operands are compatible at the final batch dimension.
3. Continue to the next batch dimension:
   ```
   Shape of C:            (2; 6, 6)
   Shape of strain: (1000, 2; 6)
                        ▲
   ```
   One of the size does not exist -- the two operands are compatible at the second last batch dimension.
4. All batch dimensions are compatible, therefore the two operands are *batch-broadcastable*.
5. The resulting batch shapes are effectively the "expand" of the two:
   ```
   Shape of C:      (1000, 2; 6, 6)
                        ▲
                        expanded (without copying)
   Shape of strain: (1000, 2; 6)
   ```

@insert-page-navigation
