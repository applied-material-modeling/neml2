# implicit_sub_batch_2d_forward: ``implicit_sub_batch_forward`` with a
# TWO-axis sub-batch, ``x`` of shape ``(B, 2, 3)`` and ``sub_batch_ndim = 2``.
#
# A one-axis sub-batch can pass by accident: right-aligned broadcasting
# lines a lone bin axis up with the right operand often enough to hide a
# lost or miscounted ``sub_batch_ndim``. Here the solved unknown crosses
# the export boundary into a forward segment that
#
# * broadcasts it against the PLAIN-batch input ``T`` (``z = T x``), which
#   only lines up if both sub-batch axes are kept out of the batch region;
# * reduces over the sub-batch twice, innermost first: ``vf_inner`` sums
#   the trailing axis to ``(B, 2)`` with ``sub_batch_ndim = 1``, and ``vf``
#   sums the remaining axis to ``(B,)``. ``vf_inner`` is exposed as an
#   output so the intermediate shape and its values are compared against
#   eager too, not just the final scalar.

[Settings]
  [example_batch_shape]
    x      = '(2; 2, 3)'
    x_rate = '(2; 2, 3)'
    t      = '(2,)'
    T      = '(2,)'
  []
[]

[Tensors]
  [R_inner]
    type = Python
    expr = 'Scalar(torch.linspace(0.1, 1.0, 6, dtype=torch.float64).reshape(2, 3))'
  []
  [R_outer]
    type = Python
    expr = 'Scalar(torch.linspace(0.5, 1.5, 2, dtype=torch.float64))'
  []
[]

[Models]
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [z]
    type = ScalarMultiplication
    from = 'T x'
    to = 'z'
  []
  [vf_inner]
    type = PrecipitateVolumeFraction
    number_density = 'z'
    volume_fraction = 'vf_inner'
    radius = 'R_inner'
  []
  [vf]
    type = PrecipitateVolumeFraction
    number_density = 'vf_inner'
    volume_fraction = 'vf'
    radius = 'R_outer'
  []
  [model]
    type = ComposedModel
    models = 'update z vf_inner vf'
    additional_outputs = 'x vf_inner'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'integrate'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 25
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
