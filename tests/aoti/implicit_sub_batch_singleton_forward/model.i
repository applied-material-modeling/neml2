# implicit_sub_batch_singleton_forward: ``implicit_sub_batch_2d_forward``
# with a singleton leading sub-batch axis, ``x`` of shape ``(B, 1, 3)`` and
# ``sub_batch_ndim = 2``.
#
# A size-1 axis is the one ``torch.export`` is most tempted to specialize
# or broadcast away, and a size-1 sub-batch axis is indistinguishable from
# a broadcast plain-batch axis by shape alone. The two reductions must
# still collapse ``(B, 1, 3)`` to ``(B, 1)`` and then ``(B,)``, and the
# plain-batch ``T`` must still broadcast against the solved unknown.

[Settings]
  [example_batch_shape]
    x      = '(2; 1, 3)'
    x_rate = '(2; 1, 3)'
    t      = '(2,)'
    T      = '(2,)'
  []
[]

[Tensors]
  [R_inner]
    type = Python
    expr = 'Scalar(torch.linspace(0.1, 1.0, 3, dtype=torch.float64).reshape(1, 3))'
  []
  [R_outer]
    type = Python
    expr = 'Scalar(torch.linspace(0.5, 1.5, 1, dtype=torch.float64))'
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
