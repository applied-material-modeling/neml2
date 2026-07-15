# implicit_substep_ramp: a NONLINEAR scalar implicit whose stiffness is driven by
# a LONE total-form force ``driving_force`` (not time), used to validate ramped
# sub-incrementation of a force via ``incremental_variables``.
#
# The Perzyna yield is scaled by the force, y = x * driving_force, so the rate is
# x_rate = (max(x*driving_force, 0) / eta)^n and the backward-Euler residual
#   r(x) = x - x~1 - (t - t~1) * (max(x*driving_force,0)/eta)^n
# is nonlinear in x (n > 1) and stiff for a large ``driving_force``. Applying the
# full force in one shot overshoots the cubic rate and the single Newton solve
# fails; listing ``driving_force`` in ``incremental_variables`` makes the compiled
# substep driver ramp it from its previous-step value (0 here) to the current
# value across bisected sub-steps, so each sub-step is easy and the solve recovers.
# The framework auto-pairs the phantom ``driving_force~1`` old value.

[Models]
  [scaled]
    type = ScalarMultiplication
    from = 'x driving_force'
    to = 'scaled_yield'
  []
  [rate]
    type = PerzynaPlasticFlowRate
    yield_function = 'scaled_yield'
    flow_rate = 'x_rate'
    reference_stress = 1.0
    exponent = 3
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [residual_model]
    type = ComposedModel
    models = 'scaled rate integrate'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual_model'
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

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'x'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
    max_substepping_level = 8
    incremental_variables = 'driving_force'
  []
[]
