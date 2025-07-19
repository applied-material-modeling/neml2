@insert-title:tutorials-tensors-indexing

[TOC]

Related reading:
- NumPy manual [Indexing on ndarrays](https://numpy.org/doc/2.2/user/basics.indexing.html)
- PyTorch documentation [Tensor Indexing API](https://pytorch.org/cppdocs/notes/tensor_indexing.html)

## Batch versus base indexing

NEML2 tensor indexing works much in the same way as in NumPy and PyTorch. We also offer a simple one-to-one translation between C++ and Python tensor indexing API. The major difference between indexing (batched) NEML2 tensor and indexing tensors in PyTorch or N-D arrays in NumPy is that **most indexing APIs have two variants -- one for indexing batch dimensions and the other for indexing base dimensions**.

## Single element indexing

Single element indexing works exactly like that for indexing flat containers such as vectors, arrays, lists, etc. The indexing is 0-based and accepts negative integers (for reverse indexing).

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/tensors/Tensor.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);

    // Create a tensor with shape (; 5)
    auto a = Tensor::create({3, 4, 5, 6, 7}, 0);
    std::cout << "a.base[2] = " << a.base_index({2}).item<double>() << std::endl;
    std::cout << "a.base[-1] = " << a.base_index({-1}).item<double>() << std::endl;
    std::cout << "a.base[-2] = " << a.base_index({-2}).item<double>() << '\n' << std::endl;

    // Create a tensor with shape (4;)
    auto b = Tensor::create({7, 6, 5, 4}, 1);
    std::cout << "b.batch[2] = " << b.batch_index({2}).item<double>() << std::endl;
    std::cout << "b.batch[-1] = " << b.batch_index({-1}).item<double>() << std::endl;
    std::cout << "b.batch[-2] = " << b.batch_index({-2}).item<double>() << std::endl;
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
  from neml2.tensors import Tensor

  torch.set_default_dtype(torch.double)

  # Create a tensor with shape (; 5)
  a = Tensor(torch.tensor([3, 4, 5, 6, 7]), 0)
  print("a.base[2] =", a.base[2].item())
  print("a.base[-1] =", a.base[-1].item())
  print("a.base[-2] =", a.base[-2].item())
  print()

  # reate a tensor with shape (4;)
  b = Tensor(torch.tensor([7, 6, 5, 4]), 1)
  print("b.batch[2] =", b.batch[2].item())
  print("b.batch[-1] =", b.batch[-1].item())
  print("b.batch[-2] =", b.batch[-2].item())
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

Single element indexing can be used to index multidimensional tensors, in which case each integer corresponds to one dimension, counting from the leading (leftmost) dimension onward. If the number of indices is smaller than the number of dimensions, a view of the subdimensional tensor is returned.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/tensors/Tensor.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);

    // Create a tensor with shape (2, 2 ; 3, 1)
    auto a = Tensor::create({{{{1}, {2}, {3}}, {{4}, {5}, {6}}},
                             {{{-1}, {-2}, {-3}}, {{-4}, {-5}, {-6}}}}, 2);

    // Single element indexing along batch dimensions
    std::cout << "a.batch[1, 0] = \n" << a.batch_index({1, 0}) << '\n' << std::endl;
    std::cout << "a.batch[0] = \n" << a.batch_index({0}) << '\n' << std::endl;

    // Single element indexing along base dimensions
    std::cout << "a.base[2, 0] = \n" << a.base_index({2, 0}) << '\n' << std::endl;
    std::cout << "a.base[1] = \n" << a.base_index({1}) << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src3
  ```
- <b class="tab-title">Python</b>
  @source:src4
  ```python
  import torch
  from neml2.tensors import Tensor

  torch.set_default_dtype(torch.double)

  # Create a tensor with shape (2, 2 ; 3, 1)
  a = Tensor(torch.tensor([[[[1], [2], [3]], [[4], [5], [6]]],
                           [[[-1], [-2], [-3]], [[-4], [-5], [-6]]]]), 2)

  # Single element indexing along batch dimensions
  print("a.batch[1, 0] =")
  print(a.batch[1, 0], "\n")
  print("a.batch[0] =")
  print(a.batch[0], "\n")

  # Single element indexing along base dimensions
  print("a.base[2, 0] =")
  print(a.base[2, 0], "\n")
  print("a.base[1] =")
  print(a.base[1])
  ```
  @endsource

  Output:
  ```
  @attach-output:src4
  ```

</div>

## Slicing

NEML2 supports the same slicing rules as in NumPy and PyTorch. A slice takes the form of
```
start:stop:step
```
where `start`, `stop`, and `step` are integers representing the starting index, ending index, and the striding of the sliced tensor view. All of these integers are optional when constructing a slice: `start` default to 0, `stop` default to \f$\infty\f$ (i.e., `std::numeric_limits<Size>::max()`), and `step` default to 1. Note that `step` must be positive.

It is best to learn slicing from examples. Below are equivalent C++ and Python codes applying the same set of slicing operations on the same tensor.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src5
  ```cpp
  #include "neml2/tensors/Scalar.h"


  int
  main()
  {
    using namespace neml2;
    using namespace indexing;
    set_default_dtype(kFloat64);

    // Create a tensor with shape (20;)
    auto a0 = Scalar::full(0.0);
    auto a1 = Scalar::full(19.0);
    auto a = Scalar::linspace(a0, a1, 20);

    std::cout << "Basic syntax: start:stop:step" << std::endl;
    std::cout << "a.batch[5:17:2] =\n" << a.batch_index({Slice(5, 17, 2)}) << '\n' << std::endl;

    std::cout << "Negative start and stop are counted backward" << std::endl;
    std::cout << "a.batch[-15:-3:2] =\n" << a.batch_index({Slice(-15, -3, 2)}) << '\n' << std::endl;

    std::cout << "start default to 0" << std::endl;
    std::cout << "a.batch[:17:3] =\n" << a.batch_index({Slice(None, 17, 3)}) << '\n' << std::endl;

    std::cout << "stop default to 'consuming all remaining elements'" << std::endl;
    std::cout << "a.batch[12::2] =\n" << a.batch_index({Slice(12, None, 2)}) << '\n' << std::endl;

    std::cout << "step default to 1" << std::endl;
    std::cout << "a.batch[3:6:] =\n" << a.batch_index({Slice(3, 6, None)}) << '\n' << std::endl;

    std::cout << "Trailing colon(s) can be omitted" << std::endl;
    std::cout << "a.batch[3:6] =\n" << a.batch_index({Slice(3, 6)}) << '\n' << std::endl;
    std::cout << "a.batch[17:] =\n" << a.batch_index({Slice(17)}) << '\n' << std::endl;

    std::cout << "The default is therefore equivalent to slicing the entire dimension" << std::endl;
    std::cout << "a.batch[:] =\n" << a.batch_index({Slice()}) << std::endl;
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
  from neml2.tensors import Scalar
  import torch

  torch.set_default_dtype(torch.double)

  # Create a tensor with shape (20;)
  a0 = Scalar.full(0.0)
  a1 = Scalar.full(19.0)
  a = Scalar.linspace(a0, a1, 20)

  print("Basic syntax: start:stop:step")
  print("a.batch[5:17:2] =")
  print(a.batch[5:17:2], "\n")

  print("Negative start and stop are counted backward")
  print("a.batch[-15:-3:2] =")
  print(a.batch[-15:-3:2], "\n")

  print("start default to 0")
  print("a.batch[:17:3] =")
  print(a.batch[:17:3], "\n")

  print("stop default to 'consuming all remaining elements'")
  print("a.batch[12::2] =")
  print(a.batch[12::2], "\n")

  print("step default to 1")
  print("a.batch[3:6:] =")
  print(a.batch[3:6:], "\n")

  print("Trailing colon(s) can be omitted")
  print("a.batch[3:6] =")
  print(a.batch[3:6], "\n")
  print("a.batch[17:] =")
  print(a.batch[17:], "\n")

  print("The default is therefore equivalent to slicing the entire dimension")
  print("a.batch[:] =")
  print(a.batch[:])
  ```
  @endsource

  Output:
  ```
  @attach-output:src6
  ```

</div>

Similar to single element indexing, slicing can also be used to index multidimensional tensors. When the number of slices is smaller than the number of dimensions, a view of the subdimensional tensor is returned.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src7
  ```cpp
  #include "neml2/tensors/Tensor.h"

  int
  main()
  {
    using namespace neml2;
    using namespace indexing;
    set_default_dtype(kFloat64);

    // Create a tensor with shape (3, 4; 2)
    auto a0 = Tensor::create({{0, 1}, {2, 3}, {4, 5}}, 1);
    auto a1 = Tensor::create({{3, 4}, {5, 6}, {7, 8}}, 1);
    auto a = Tensor::linspace(a0, a1, 4, 1);

    std::cout << "a.batch[:2, -3:4] =\n" << a.batch_index({Slice(None, 2), Slice(-3, 4)}) << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src7
  ```
- <b class="tab-title">Python</b>
  @source:src8
  ```python
  import torch
  from neml2.tensors import Tensor

  torch.set_default_dtype(torch.double)

  a0 = Tensor(torch.tensor([[0, 1], [2, 3], [4, 5]]), 1)
  a1 = Tensor(torch.tensor([[3, 4], [5, 6], [7, 8]]), 1)
  a = Tensor.linspace(a0, a1, 4, 1)

  print("a.batch[:2, -3:4] =")
  print(a.batch[:2, -3:4])
  ```
  @endsource

  Output:
  ```
  @attach-output:src8
  ```

</div>

## Dimensional indexing tools

When indexing multidimensional tensors, having to specify the element index or slicing for each dimension would be cumbersome. To simplify multidimensional indexing, some special syntax and notations are reserved for inferring tensor shapes.

NEML2 currently supports two special symbols:
- `Ellipsis` or `...` (only available in Python) is equivalent to one or multiple `Slice()` or `:` expanding the rest of the dimensions.
- `None` (when used as a index) is equivalent to NumPy's `newaxis` which unsqueezes a unit-length dimension at the specified place.

Again, the use of these special symbols are best illustrated by examples.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src9
  ```cpp
  #include <torch/torch.h>
  #include "neml2/tensors/Tensor.h"

  int
  main()
  {
    using namespace neml2;
    using namespace indexing;
    set_default_dtype(kFloat64);

    // Create a tensor with shape (5, 3, 1, 3; 1, 7, 8)
    auto a = Tensor(torch::rand({5, 3, 1, 3, 1, 7, 8}), 4);

    // Batch indexing with ellipsis
    std::cout << "a.batch[2:, ..., :].shape = " << a.batch_index({Slice(2), Ellipsis, Slice()}).sizes() << std::endl;

    // Batch indexing with none
    std::cout << "a.batch[2:, :, :, None, :].shape = " << a.batch_index({Slice(2), Slice(), Slice(), None, Slice()}).sizes() << std::endl;

    // Batch indexing with both ellipsis and none
    std::cout << "a.batch[..., None, :].shape = " << a.batch_index({Ellipsis, None, Slice()}).sizes() << std::endl;

    // Same rules apply to base indexing
    std::cout << "a.base[..., None, :].shape = " << a.base_index({Ellipsis, None, Slice()}).sizes() << std::endl;
  }
  ```
  @endsource

  Output:
  ```
  @attach-output:src9
  ```
- <b class="tab-title">Python</b>
  @source:src10
  ```python
  import torch
  from neml2.tensors import Tensor

  torch.set_default_dtype(torch.double)

  # Create a tensor with shape (5, 3, 1, 3; 1, 7, 8)
  a = Tensor(torch.rand(5, 3, 1, 3, 1, 7, 8), 4)

  # Batch indexing with ellipsis
  print("a.batch[2:, ..., :].shape =", a.batch[2:, ..., :].shape)

  # Batch indexing with none
  print("a.batch[2:, :, :, None, :].shape =", a.batch[2:, :, :, None, :].shape)

  # Batch indexing with both ellipsis and none
  print("a.batch[..., None, :].shape =", a.batch[..., None, :].shape)

  # Same rules apply to base indexing
  print("a.base[..., None, :].shape =", a.base[..., None, :].shape)
  ```
  @endsource

  Output:
  ```
  @attach-output:src10
  ```

</div>


@insert-page-navigation
