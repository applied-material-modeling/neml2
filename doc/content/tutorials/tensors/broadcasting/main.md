@insert-title:tutorials-tensors-broadcasting

[TOC]

Related reading: NumPy manual on [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## NEML2's broadcasting rules

NEML2's broadcasting rules are built on top of NumPy's general broadcasting rules.

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
  @list:cpp:tensors/broadcasting/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:tensors/broadcasting/ex2.py

  Output:
  @list-output:ex2

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
