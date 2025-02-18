@insert-title:tutorials-tensors-tensor-view

[TOC]

## View v.s. copy

NEML2 allows a tensor to be a view of an existing tensor. As the name suggests, a tensor view shares the same underlying data with the tensor it is viewing into. In other words, supporting tensor view avoids data copy. Moreover, tensor views optionally (and often) reinterpret the shape and/or striding of the original data, allowing for fast and memory efficient reshaping, slicing, and element-wise operations.

\remark
Tensor view is the enabling technique behind efficient assembly of implicit systems.

In fact, all indexing mechanisms covered in the [previous tutorial](#tutorials-tensors-indexing) are creating tensor views, i.e., zero copy, negligible allocation. In addition to those indexing API, NEML2 also provides flexible tensor reshaping API (documented in neml2::TensorBase).

## Who touched my data?

Tensor view avoids explicit data copy, which means
- Modification in the original data will be reflected by the tensor view
- Modifying data viewed by the tensor view will alter the original data

It is therefore important to understand the difference between view and copy, and when to declare ownership (copy, or clone) of the original data.

The following example demonstrates the two bullet points, i.e., "change in original data" <-> "change in data viewed by the tensor view".

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/tensors/Tensor.h"
  using namespace neml2;
  using namespace indexing;

  int
  main()
  {
    // Create a tensor with shape (; 4, 3) filled with zeros
    auto a = Tensor::zeros({4, 3});
    std::cout << "a =\n" << a << std::endl;

    // b is a view into the first row and the third row of a
    auto b = a.base_index({Slice(None, None, 2)});
    std::cout << "b =\n" << b << std::endl;

    // Modification in a is reflected in b
    a += 1.0;
    std::cout << "\nAfter first modification" << std::endl;
    std::cout << "a =\n" << a << std::endl;
    std::cout << "b =\n" << b << std::endl;

    // Modification in data viewed by b is reflected in a
    b += 1.0;
    std::cout << "\nAfter second modification" << std::endl;
    std::cout << "a =\n" << a << std::endl;
    std::cout << "b =\n" << b << std::endl;
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
  from neml2.tensors import Tensor

  # Create a tensor with shape (; 4, 3) filled with zeros
  a = Tensor.zeros((4, 3))
  print("a =")
  print(a)

  # b is a view into the first row and the third row of a
  b = a.base[::2]
  print("b =")
  print(b)

  # Modification in a is reflected in b
  a += 1.0
  print("\nAfter first modification")
  print("a =")
  print(a)
  print("b =")
  print(b)

  # Modification in data viewed by b is reflected in a
  b += 1.0
  print("\nAfter second modification")
  print("a =")
  print(a)
  print("b =")
  print(b)
  ```
  @endsource

  Output:
  ```
  @attach_output:src2
  ```

</div>

\note
The same statements/rules still hold when multiple tensor views are viewing into the same underlying data, even if they are views of different regions.

@insert-page-navigation
