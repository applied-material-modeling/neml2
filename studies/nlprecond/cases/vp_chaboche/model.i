# neml2
# nlprecond testbed case: vp_chaboche
# Derived from tests/regression/solid_mechanics/viscoplasticity/chaboche/model.i
#
# Differences from the parent regression scenario:
#   * [Drivers/regression] removed (no gold reference; this is a solver study)
#   * knobs exposed as HIT ${...} substitutions, supplied by studies.nlprecond.harness:
#       nbatch    -- dynamic batch members
#       npoint    -- time points; the driver takes npoint-1 steps
#       tfrac     -- fraction of the parent's full load history to cover. The
#                    harness sets it so the PER-STEP INCREMENT stays fixed as
#                    npoint shrinks -- fewer steps, not bigger ones. Increment
#                    size is a separate knob (dt_scale), which scales tfrac.
#       flow_n    -- PerzynaPlasticFlowRate exponent (rate sensitivity)
#       ls_iters  -- max_linesearch_iterations; 1 == full Newton step (no line search)
#       max_its   -- Newton iteration cap
#   * [Solvers/newton] is always NewtonWithLineSearch so ls_iters spans both arms
#   * the predictor wiring is left intact; the harness strips it for the
#     "nopred" arms by deleting the `predictor = '...'` line(s)
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, ${nbatch}, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, ${tfrac}, ${npoint}, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_strain]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(${nbatch})'
  []
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, ${tfrac}, ${npoint}, dtype=torch.float64).reshape(${npoint}, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2'
    to = 'back_stress'
    weights = '1 1'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    to = 'overstress'
    from = 'mandel_stress back_stress'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'overstress'
    invariant = 'effective_stress'
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = '${flow_n}'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'E'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'E_rate plastic_strain_rate'
    to = 'strain_rate'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain X1 X2'
    residuals = 'stress_residual equivalent_plastic_strain_residual X1_residual X2_residual'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = '${ls_iters}'
    max_its = '${max_its}'
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress X1 X2'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
