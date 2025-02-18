@insert-title:tutorials-models-just-in-time-compilation

[TOC]

## Problem description

In NEML2, all tensor operations are traceable. The trace of an operation records the operator type, a stack of arguments and outputs, together with additional context. Multiple operations performed in sequence can be traced together into a *graph* representing the flow of data through operators. Such graph representation is primarily used for two purposes:
- just-in-time (JIT) compilation and optimization of the operations
- backward automatic-differentiation (AD)

This tutorial illustrates the utility of JIT compilation of NEML2 models, and the a [later tutorial](#tutorials-optimization-automatic-differentiation) demonstrates the use of backward AD to calculate parameter derivatives.

In this tutorial, let us consider the following problem
\f{align}
  \dot{a} & = \dfrac{a - a_n}{t - t_n}, \label{1} \\
  \dot{b} & = \dfrac{b - b_n}{t - t_n}, \label{2} \\
  \dot{c} & = \dfrac{c - c_n}{t - t_n}, \label{3}
\f}
where the subscript \f$ n \f$ represents the variable value from the previous time step.

## Model structure

All three equations can be translated to [ScalarVariableValue](#scalarvariablerate). The input file looks like
```
[Models]
  [eq1]
    type = ScalarVariableRate
    variable = 'state/a'
  []
  [eq2]
    type = ScalarVariableRate
    variable = 'state/b'
  []
  [eq3]
    type = ScalarVariableRate
    variable = 'state/c'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
```
And the composed model correctly defines \f$ a \f$, \f$ a_n \f$, \f$ b \f$, \f$ b_n \f$, \f$ c \f$, \f$ c_n \f$, \f$ t \f$, \f$ t_n \f$ as input variables and \f$ \dot{a} \f$, \f$ \dot{b} \f$, \f$ \dot{c} \f$ as output variables.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "eq");
    std::cout << model << std::endl;
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
  import neml2

  model = neml2.load_model("input.i", "eq")
  print(model)
  ```
  @endsource

  Output:
  ```
  @attach_output:src2
  ```

</div>

## Tracing

NEML2 enables tracing of tensor operations lazily. No tracing is performed when the model is first loaded from the input file. Tracing takes place when the model is being evaluated for the first time. The following code can be used to view the traced graph in text format.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Scalar.h"
  #include "neml2/jit/utils.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "eq");

    // Create example input variables for tracing
    auto a = Scalar::full(1.0);
    auto b = Scalar::full(2.0);
    auto c = Scalar::full(3.0);
    auto t = Scalar::full(0.1);
    auto a_n = Scalar::full(0.0);
    auto b_n = Scalar::full(1.0);
    auto c_n = Scalar::full(2.0);
    auto t_n = Scalar::full(0.0);

    // Evaluate the model for the first time
    // This is when tracing takes place
    model.value({{VariableName("state", "a"), a},
                 {VariableName("state", "b"), b},
                 {VariableName("state", "c"), c},
                 {VariableName("forces", "t"), t},
                 {VariableName("old_state", "a"), a_n},
                 {VariableName("old_state", "b"), b_n},
                 {VariableName("old_state", "c"), c_n},
                 {VariableName("old_forces", "t"), t_n}});

    utils::last_executed_optimized_graph()->dump();
  }
  ```
  @endsource

  Output:
  ```
  @attach_output:src3
  ```
- <b class="tab-title">Python</b>
  @source:src4
  ```python
  import neml2
  from neml2.tensors import Scalar
  import torch

  model = neml2.load_model("input.i", "eq")

  # Create example input variables for tracing
  a = Scalar.full(1.0)
  b = Scalar.full(2.0)
  c = Scalar.full(3.0)
  t = Scalar.full(0.1)
  a_n = Scalar.full(0.0)
  b_n = Scalar.full(1.0)
  c_n = Scalar.full(2.0)
  t_n = Scalar.full(0.0)

  # Evaluate the model for the first time
  # This is when tracing takes place
  model.value({"state/a": a,
               "state/b": b,
               "state/c": c,
               "forces/t": t,
               "old_state/a": a_n,
               "old_state/b": b,
               "old_state/c": c,
               "old_forces/t": t})

  print(torch.jit.last_executed_optimized_graph())
  ```
  @endsource

  Output:
  ```
  @attach_output:src4
  ```

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
  @source:src5
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Scalar.h"
  #include "neml2/jit/utils.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input.i", "eq");

    // Create example input variables for tracing
    auto a = Scalar::full(1.0);
    auto b = Scalar::full(2.0);
    auto c = Scalar::full(3.0);
    auto t = Scalar::full(0.1);
    auto a_n = Scalar::full(0.0);
    auto b_n = Scalar::full(1.0);
    auto c_n = Scalar::full(2.0);
    auto t_n = Scalar::full(0.0);

    // Evaluate the model for multiple times
    auto inputs = ValueMap({{VariableName("state", "a"), a},
                            {VariableName("state", "b"), b},
                            {VariableName("state", "c"), c},
                            {VariableName("forces", "t"), t},
                            {VariableName("old_state", "a"), a_n},
                            {VariableName("old_state", "b"), b_n},
                            {VariableName("old_state", "c"), c_n},
                            {VariableName("old_forces", "t"), t_n}});
    for (int i = 0; i < 10; i++)
      model.value(inputs);

    utils::last_executed_optimized_graph()->dump();
  }
  ```
  @endsource

  Output:
  ```
  @attach_output:src5
  ```
- <b class="tab-title">Python</b>
  @source:src6
  ```python
  import neml2
  from neml2.tensors import Scalar
  import torch

  model = neml2.load_model("input.i", "eq")

  # Create example input variables for tracing
  a = Scalar.full(1.0)
  b = Scalar.full(2.0)
  c = Scalar.full(3.0)
  t = Scalar.full(0.1)
  a_n = Scalar.full(0.0)
  b_n = Scalar.full(1.0)
  c_n = Scalar.full(2.0)
  t_n = Scalar.full(0.0)

  # Evaluate the model for the first time
  # This is when tracing takes place
  inputs = {"state/a": a,
           "state/b": b,
           "state/c": c,
           "forces/t": t,
           "old_state/a": a_n,
           "old_state/b": b,
           "old_state/c": c,
           "old_forces/t": t}
  for i in range(10):
    model.value(inputs)

  print(torch.jit.last_executed_optimized_graph())
  ```
  @endsource

  Output:
  ```
  @attach_output:src6
  ```

</div>

Note how the optimized graph successfully identifies the common subexpression \f$ t - t_n \f$ and reuses it in all three equations.

## Limitations

JIT optimization and compilation isn't the holy grail for improving performance of all models. For tensor operations that branch based on variable data, the traced graph cannot capture such data dependency and would potentially produce wrong results. For models defining the derivatives relying on backward automatic differentiation, NEML2 is not able to generate efficient traced graphs.

Due to these limitations, certain models disable the use of JIT compilation. The most notable case is [ImplicitUpdate](#implicitupdate) due to its use of Newton-Raphson solvers which are in general data dependent.

When multiple models are composed together, a single function graph is by default traced through all sub-models. However, if one of the sub-model does not allow JIT, e.g., is of type `ImplicitUpdate`, then the composed model falls back to trace each individual sub-model except for those explicit disabling JIT. Therefore, it is generally recommended to compose JIT-enabled sub-models separate from those JIT-disabled ones, allowing for more optimization opportunities.

@insert-page-navigation
