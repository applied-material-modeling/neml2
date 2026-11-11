# implicit_sub_batch_forward: an ImplicitUpdate over a sub-batched unknown
# followed by a forward segment that CONSUMES the solved unknown.
#
# ``implicit_sub_batch`` covers the solve on its own, so it never exercises
# the hand-off: the solved unknown's sub-batch axis has to survive into the
# next segment's trace. It did not (#479). The typed wrappers drop their
# metadata through ``torch.export``'s pytree round-trip, so the forward
# segment traced with ``sub_batch_ndim = 0`` and read the per-bin axis as
# plain batch.
#
# The forward segment here is built to catch both faces of that:
#
# * ``vf`` reduces OVER the sub-batch (sum_i 4/3 pi R_i^3 z_i). Read as
#   plain batch the reduction is skipped and ``vf`` comes back ``(2, 4)``
#   instead of ``(2,)`` -- a silent wrong answer, caught here by the
#   eager-vs-AOTI comparison in ``test_aoti_export_reload_matches_eager``.
# * ``z`` multiplies the sub-batched unknown by the PLAIN-batch input ``T``.
#   Read as plain batch the ``(2, 4)`` bin axis no longer broadcasts against
#   ``T``'s ``(2,)`` and the export fails outright ("The size of tensor a
#   (s74) must match the size of tensor b (4)"), so this scenario also
#   guards the compile itself.

[Settings]
  [example_batch_shape]
    x      = '(2; 4)'
    x_rate = '(2; 4)'
    t      = '(2,)'
    T      = '(2,)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'logspace(Scalar(-1.0).sub_batch, Scalar(0.0).sub_batch, 4)'
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
  [vf]
    type = PrecipitateVolumeFraction
    number_density = 'z'
    volume_fraction = 'vf'
    radius = 'R'
  []
  [model]
    type = ComposedModel
    models = 'update z vf'
    additional_outputs = 'x'
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
