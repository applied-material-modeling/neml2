@insert-title:tutorials-models-implicit-model

[TOC]

## Problem description

One of the most notable differences between constitutive models and feed-forward neural networks is that implicitly updating certain stiff systems is much more computationally efficient than explicit algorithms.

A generally nonlinear, recursive, implicit system of equations take the following form
\f{align*}
  \mathbf{r}(\mathbf{s}) & = f(\mathbf{s}, \mathbf{f}, \mathbf{s}_n, \mathbf{f}_n; \mathbf{p}), \\
  \mathbf{s} &= \mathop{\mathrm{root}}\limits_{\mathbf{s}} (\mathbf{r}).
\f}
Here \f$ \mathbf{r} \f$ represents the residual (for root-fiding) of the system of equations, and \f$ \mathbf{s} \f$, \f$ \mathbf{f} \f$, \f$ \mathbf{s}_n \f$, \f$ \mathbf{f}_n \f$ are defined by four reserved sub-axes in NEML2:
- The `state` sub-axis hosts input variables in set \f$ \mathbf{s} \f$. The variables on this sub-axis are the primary unknowns to be solved for.
- The `forces` sub-axis hosts *prescribed* input variables in set \f$ \mathbf{f} \f$. These variables are prescribed and, by definition, do not change while the system is being solved.
- The `old_state` and `old_forces` sub-axes respectively correspond to \f$ \mathbf{s}_n \f$ and \f$ \mathbf{f}_n \f$. These variables correspond to the *previous* solution to the system to facilitate the recursive definition of internal variables in history-dependent models. The equivalent plastic strain in plasticity models is a well-known example.

The Perzyna viscoplasticity model mentioned in a [previous tutorial](#tutorials-models-model-composition) takes this form:
\f{align}
  \boldsymbol{\varepsilon}^e & = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \label{1} \\
  \boldsymbol{\sigma} & = 3K\operatorname{vol}\boldsymbol{\varepsilon}^e + 2G\operatorname{dev}\boldsymbol{\varepsilon}^e, \label{2} \\
  \bar{\sigma} & = J_2(\boldsymbol{\sigma}), \label{3} \\
  f^p & = \bar{\sigma} - \sigma_y, \label{4} \\
  \boldsymbol{N} & = \pdv{f^p}{\boldsymbol{\sigma}}, \label{5} \\
  \dot{\gamma} & = \left( \dfrac{\left< f^p \right>}{\eta} \right)^n, \label{6} \\
  \dot{\boldsymbol{\varepsilon}}^p & = \dot{\gamma} \boldsymbol{N}. \label{7}
\f}
Its residual can be defined using backward-Euler time integration as
\f{align}
  \mathbf{r}\left( \boldsymbol{\varepsilon}^p \right) = \boldsymbol{\varepsilon}^p - \boldsymbol{\varepsilon}^p_n - \left( t - t_n \right) \dot{\boldsymbol{\varepsilon}}^p, \label{8}
\f}
the solution procedure of which is often referred to as the *return-mapping algorithm* in the solid mechanics community.

This tutorial illustrates the use of [ImplicitUpdate](#implicitupdate) in conjunction with a Newton-Raphson solver to perform the implicit update.

## Defining the system of equations

After searching among existing models provided by NEML2, we are able to translate each of these equations into a NEML2 model. The input file looks like
```
[Models]
  [eq1]
    type = SR2LinearCombination
    from_var = 'forces/E state/Ep'
    to_var = 'state/Ee'
    coefficients = '1 -1'
  []
  [eq2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [eq3]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [eq4]
    type = YieldFunction
    yield_stress = 5
    yield_function = 'state/fp'
    effective_stress = 'state/s'
  []
  [surface]
    type = ComposedModel
    models = 'vonmises yield'
  []
  [eq5]
    type = Normality
    model = 'surface'
    function = 'state/fp'
    from = 'state/S'
    to = 'state/N'
  []
  [eq6]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
    yield_function = 'state/fp'
    flow_rate = 'state/gamma_rate'
  []
  [eq7]
    type = AssociativePlasticFlow
    flow_rate = 'state/gamma_rate'
    flow_direction = 'state/N'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [eq8]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/Ep'
  []
  [system]
    type = ComposedModel
    models = 'eq1 eq2 surface eq5 eq6 eq7 eq8'
  []
[]
```
Note the one-to-one correspondance between the models and the equations.

The structure of the system of equations can be summarized using the code below.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include "neml2/models/Model.h"

  using namespace neml2;

  int
  main()
  {
    auto & system = load_model("input1.i", "system");
    std::cout << system << std::endl;
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
  import neml2

  system = neml2.load_model("input1.i", "system")
  print(system)
  ```
  @endsource

  Output:
  ```
  @attach-output:src2
  ```

</div>

## Solving the system of equations

Once the system of equations are properly defined, we can use the [ImplicitUpdate](#implicitupdate) to solve the system of equations. The [ImplicitUpdate](#implicitupdate) is responsible for the following:
- (Re)declare the solution to the system of equations as output variables.
- Validate the shape of input variables and residual variables to make sure the system is square.
- Assemble the residual vector and Jacobian matrix of the underlying linear system.
- Invoke a *solver* to solve the system of equations.
- Apply the implicit function theorem to calculate exact derivatives (up to machine precision).

At the time of writing, NEML2 offers three fully vectorized Newton solvers to be used in conjunction with [ImplicitUpdate](#implicitupdate):
- [Newton](#newton), the vanilla version of the Newton-Raphson algorithm which always takes the "full" step.
- [NewtonWithLineSearch](#newtonwithlinesearch), similar to Newton but offers several commonly used (again fully vectorized) line search strategies.
- [NewtonWithTrustRegion](#newtonwithtrustregion), the trust-region variant of the Newton-Raphson algorithm, where a scalar-valued, fully vectorized quadratic sub-problem is solved to identify the search direction in each update.

In addition, the assembly routines as well as the application of the implicit function theorem are also implemented in a vectorized fashion.

The resulting input file looks like
```
[Models]
  # ...
  # Other models defining the system of equations
  # ...
  [system]
    type = ComposedModel
    models = 'eq1 eq2 surface eq5 eq6 eq7 eq8'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'system'
    solver = 'newton'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    verbose = true
  []
[]
```

The [ImplicitUpdate](#implicitupdate) model can then be invoked in the same way as regular models.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src3
  ```cpp
  #include "neml2/models/Model.h"
  #include "neml2/tensors/Scalar.h"
  #include "neml2/tensors/SR2.h"

  using namespace neml2;

  int
  main()
  {
    auto & model = load_model("input2.i", "model");

    // Create input variables
    // Unspecified variables are assumed to be zero
    auto E = SR2::fill(0.01, 0.005, -0.001);
    auto t = Scalar::full(1);

    // Solve the implicit model
    auto outputs = model.value({{VariableName("forces", "E"), E},
                                {VariableName("forces", "t"), t}});

    // Get the solution
    std::cout << "\nPlastic strain:\n" << outputs[VariableName("state", "Ep")] << std::endl;
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
  import neml2
  from neml2.tensors import Scalar, SR2

  model = neml2.load_model("input2.i", "model")

  # Create input variables
  # Unspecified variables are assumed to be zero
  E = SR2.fill(0.01, 0.005, -0.001)
  t = Scalar.full(1)

  # Solve the implicit model
  outputs = model.value({"forces/E": E, "forces/t": t})

  # Get the solution
  print("\nPlastic strain:")
  print(outputs["state/Ep"])
  ```
  @endsource

  Output:
  ```
  @attach-output:src4
  ```

</div>

## Remarks on the implicit function theorem

Unlike other regular models, declaring variables on the *correct* sub-axes is important because NEML2 relies on the reserved sub-axes (`state`, `forces`, etc.)
- To determine whether the variable value and derivatives should be calculated during the assembly of residual and Jacobian. For example, the derivatives with respect to all variables on the `forces` sub-axis are skipped because they are not required in the assembly of the linear system.
- To efficiently reuse the factorization of the system Jacobian when applying the implicit function theorem.

As long as models are defined using the *correct* sub-axis definitions and satisfy some mild continuity requirements, **NEML2 guarantees the correctness of the outcoming variable derivatives** after one or more implicit updates, up to machine precision. The same guarantee also applies to user-defined custom models.

This is a significant advantage compared to some of the alternative constitutive model libraries, especially in the context of coupling with external PDE solvers. For example, in the context of finite element method, thermodynamic forces (e.g. strain) are calculated at each quadrature point, and the constitutive library (e.g. NEML2) is responsible for updating the thermodynamic state variables (e.g. stress, plastic strain, etc.), which are then used in the residual definition of the discretized PDE. Therefore, the exact derivatives of the state variables w.r.t. the forces are the key to the assembly of the exact Jacobian of the descretized PDE, which is in turn the fundamental requirement for optimal convergence for many nonlinear solvers.

@insert-page-navigation
