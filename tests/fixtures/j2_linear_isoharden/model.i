# neml2
# J2 + LinearIsotropicHardening residual surface used as the C++ reference in
# `tests/aoti/bench_J2Newton.cxx`. Mirrors the structure of
# `tests/regression/solid_mechanics/rate_independent_plasticity/isoharden/model.i`
# but swaps `VoceIsotropicHardening` for `LinearIsotropicHardening` and
# strips the [Tensors] / [Drivers] sections. The benchmark loads `surface`
# directly and calls `Model::value_and_dvalue(...)`; the implicit return-map
# solve is deferred to the next migration phase.
#
# Material parameters must match `tests/aoti/build_test_artifact.py` so the
# benchmark compares the same J2 problem in both contenders:
#   E   = 1e5
#   nu  = 0.3
#   K   = 1000   (LinearIsotropicHardening hardening_modulus)
#   sy  = 1000   (YieldFunction yield_stress)
#

[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'total_strain plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
  []
  # Block name avoids the bare `yield` keyword: AOTI export rejects Python
  # reserved keywords as block names (see neml2.factory._check_python_attr_name).
  [yield_surface]
    type = YieldFunction
    yield_stress = 1000
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [consistency]
    type = FBComplementarity
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'flow_rate'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden elastic_strain elasticity
              mandel_stress vonmises
              yield_surface normality eprate Eprate
              consistency integrate_ep integrate_Ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain flow_rate'
    residuals = 'plastic_strain_residual equivalent_plastic_strain_residual complementarity'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
    abs_tol = 1e-12
    rel_tol = 1e-12
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain flow_rate'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  # Outer composition: implicit return-map + post-solve stress. Phase 7 exports
  # this as an AOTIModel split at the return_map breakpoint (seg0 = implicit,
  # seg1 = forward).  Named `j2_model` (not `model`) so its AOTI
  # cache metadata file doesn't collide with the unrelated `model` in
  # `LinearIsotropicElasticity_production.i` (both HITs share the same
  # `tests/aoti/.neml2_aoti/` cache directory in the test suite).
  [j2_model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain equivalent_plastic_strain'
  []
[]
