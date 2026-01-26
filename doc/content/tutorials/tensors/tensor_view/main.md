@insert-title:tutorials-tensors-tensor-view

[TOC]

## View versus copy

NEML2 allows a tensor to be a view of an existing tensor. As the name suggests, a tensor view shares the same underlying data with the tensor it is viewing into. In other words, supporting tensor view avoids data copy. Moreover, tensor views optionally (and often) reinterpret the shape and/or striding of the original data, allowing for fast and memory efficient reshaping, slicing, and element-wise operations.

\remark
Tensor views enable efficient assembly of implicit systems.

In fact, all indexing mechanisms covered in the [previous tutorial](#tutorials-tensors-indexing) are creating tensor views, i.e., zero copy, negligible allocation. In addition to those indexing API, NEML2 also provides flexible tensor reshaping API (documented in neml2::TensorBase).

## Who touched my data?

Tensor views avoid explicit data copying, which means:
- Modification in the original data will be reflected by the tensor view
- Modifying data viewed by the tensor view will alter the original data

It is therefore important to understand the difference between view and copy, and when to declare ownership (copy, or clone) of the original data.

The following example demonstrates the two bullet points, i.e., "change in original data" <-> "change in data viewed by the tensor view".

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:tensors/tensor_view/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:tensors/tensor_view/ex2.py

  Output:
  @list-output:ex2

</div>

\note
The same statements/rules still hold when multiple tensor views are viewing into the same underlying data, even if they are views of different regions.

@insert-page-navigation
