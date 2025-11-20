@insert-title:tutorials-models-just-in-time-compilation

[TOC]

## Problem description

In NEML2, all tensor operations are traceable. The trace of an operation records the operator type, a stack of arguments and outputs, together with additional context. Multiple operations performed in sequence can be traced together into a *graph* representing the flow of data through operators. Such graph representation is primarily used for two purposes:
- Just-in-time (JIT) compilation and optimization of the operations
- Backward automatic-differentiation (AD)

This tutorial illustrates the utility of JIT compilation of NEML2 models, and a [later tutorial](#tutorials-optimization-automatic-differentiation) demonstrates the use of backward AD to calculate parameter derivatives.

In this tutorial, let us consider the following problem
\f{align}
  \dot{a} & = \dfrac{a - a_n}{t - t_n}, \label{1} \\
  \dot{b} & = \dfrac{b - b_n}{t - t_n}, \label{2} \\
  \dot{c} & = \dfrac{c - c_n}{t - t_n}, \label{3}
\f}
where the subscript \f$ n \f$ represents the variable value from the previous time step.

## Model structure

All three equations can be translated to [ScalarVariableRate](#scalarvariablerate). The input file looks like

@list-input:just_in_time_compilation/input.i

And the composed model correctly defines \f$ a \f$, \f$ a_n \f$, \f$ b \f$, \f$ b_n \f$, \f$ c \f$, \f$ c_n \f$, \f$ t \f$, \f$ t_n \f$ as input variables and \f$ \dot{a} \f$, \f$ \dot{b} \f$, \f$ \dot{c} \f$ as output variables.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:just_in_time_compilation/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:just_in_time_compilation/ex2.py

  Output:
  @list-output:ex2

</div>

## Tracing

NEML2 enables tracing of tensor operations lazily. No tracing is performed when the model is first loaded from the input file. Tracing takes place when the model is being evaluated for the first time. The following code can be used to view the traced graph in text format.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:just_in_time_compilation/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:python:just_in_time_compilation/ex4.py

  Output:
  @list-output:ex4

</div>

Note that the above graph is called a *profiling* graph. While it is not the most human-friendly to read, let us highlight some lines of the text output to try to associated it with the equations.

The following lines
```
  %18 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%eq::state/a)
  %19 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%eq::old_state/a)
  %10 : Tensor = aten::sub(%18, %19, %8)
  %20 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%eq::forces/t)
  %21 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%eq::old_forces/t)
  %11 : Tensor = aten::sub(%20, %21, %8)
  %22 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%10)
  %23 : Tensor = prim::profile[profiled_type=Double(requires_grad=0, device=cpu), seen_none=0](%11)
  %12 : Tensor = aten::div(%22, %23)
```
cover three tensor operations: two `aten::sub`s and one `aten::div`, which correspond to \f$ \eqref{1} \f$. Note that each variable is wrapped inside a profiling node denoted as `prim::profile`. These wrappers allow the graph executor to record and analyze the runtime statistics of tensor operations, in order to identify hot spots to optimize.

## JIT optimization

With the profiling graph, further execution of the same traced graph automatically identifies opportunities for optimization. In summary, the following types of optimizations are enabled in NEML2 by default:
- Inlining
- Constant pooling
- Removing expands
- Canonicalization
- Dead code elimination
- Constant propagation
- Input shape propagation
- Common subexpression extraction
- Peephole optimization
- Loop unrolling

See the [PyTorch JIT](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md) design document for detailed explanation on each of the optimization pass.

The code below shows that, after a few forward evaluations, the traced graph can be substantially optimized.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:just_in_time_compilation/ex5.cxx

  Output:
  @list-output:ex5
- <b class="tab-title">Python</b>
  @list:python:just_in_time_compilation/ex6.py

  Output:
  @list-output:ex6

</div>

Note how the optimized graph successfully identifies the common subexpression \f$ t - t_n \f$ and reuses it in all three equations.

## Limitations

JIT optimization and compilation isn't the holy grail for improving performance of all models. For tensor operations that branch based on variable data, the traced graph cannot capture such data dependency and would potentially produce wrong results. NEML2 is unable to generate traced graphs for models that include derivatives of other models in the forward evaluation when those derivatives are defined with automatic differentiation.

Due to these limitations, certain models disable the use of JIT compilation. The most notable case is [ImplicitUpdate](#implicitupdate) due to its use of Newton-Raphson solvers which are in general data dependent.  However, the portions of the complete model defining the implicit function to solve can often benefit from JIT compilation.

When multiple models are composed together, a single function graph is by default traced through all sub-models. However, if one of the sub-model does not allow JIT, e.g., is of type `ImplicitUpdate`, then the composed model falls back to trace each individual sub-model except for those explicitly disabling JIT. Therefore, it is generally recommended to compose JIT-enabled sub-models separate from those JIT-disabled ones, allowing for more optimization opportunities.

@insert-page-navigation
