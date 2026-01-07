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
  @list:cpp:tensors/indexing/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:tensors/indexing/ex2.py

  Output:
  @list-output:ex2

</div>

Single element indexing can be used to index multidimensional tensors, in which case each integer corresponds to one dimension, counting from the leading (leftmost) dimension onward. If the number of indices is smaller than the number of dimensions, a view of the subdimensional tensor is returned.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:tensors/indexing/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:python:tensors/indexing/ex4.py

  Output:
  @list-output:ex4

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
  @list:cpp:tensors/indexing/ex5.cxx

  Output:
  @list-output:ex5
- <b class="tab-title">Python</b>
  @list:python:tensors/indexing/ex6.py

  Output:
  @list-output:ex6

</div>

Similar to single element indexing, slicing can also be used to index multidimensional tensors. When the number of slices is smaller than the number of dimensions, a view of the subdimensional tensor is returned.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:tensors/indexing/ex7.cxx

  Output:
  @list-output:ex7
- <b class="tab-title">Python</b>
  @list:python:tensors/indexing/ex8.py

  Output:
  @list-output:ex8

</div>

## Dimensional indexing tools

When indexing multidimensional tensors, having to specify the element index or slicing for each dimension would be cumbersome. To simplify multidimensional indexing, some special syntax and notations are reserved for inferring tensor shapes.

NEML2 currently supports two special symbols:
- `Ellipsis` or `...` (only available in Python) is equivalent to one or multiple `Slice()` or `:` expanding the rest of the dimensions.
- `None` (when used as a index) is equivalent to NumPy's `newaxis` which unsqueezes a unit-length dimension at the specified place.

Again, the use of these special symbols are best illustrated by examples.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:tensors/indexing/ex9.cxx

  Output:
  @list-output:ex9
- <b class="tab-title">Python</b>
  @list:python:tensors/indexing/ex10.py

  Output:
  @list-output:ex10

</div>


@insert-page-navigation
