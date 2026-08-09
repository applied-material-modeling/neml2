# neml2
# nlprecond testbed case: vp_isoharden_inverted
# Derived from studies/nlprecond/cases/vp_isoharden/model.i -- SAME PHYSICS,
# different residual FORM. The Perzyna flow rule is stated implicitly in
# inverted form (eta*gdot^(1/n) - <f> = 0, PerzynaPlasticFlowRateResidual)
# with flow_rate carried as an unknown, instead of explicitly
# (gdot = (<f>/eta)^n, PerzynaPlasticFlowRate) and substituted in.
#
# Same root; what changes is what Newton sees. Substituting the explicit map
# makes every residual degree-n in the stress, and Newton on a degree-n
# monomial converges linearly at (1-1/n)^n. Carrying gdot as an unknown
# relocates that to a single 1/n power and leaves the stress / eq-plastic-
# strain residuals affine in gdot.
#
# Extra knob: gdot_cutoff -- flow rate below which the fractional power is
# replaced by its tangent line (bounds the derivative at the origin, where a
# cold start sits and where the elastic-branch root lies).
#
# Originally from tests/regression/solid_mechanics/viscoplasticity/isoharden/model.i
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
  # Strictly positive seed for the flow-rate unknown. At step 1 there is no
  # previous solve to extrapolate from, so this is what the predictor hands
  # Newton; the inverted residual is regularized at zero but a positive start
  # still saves a couple of iterations.
  [gdot_ic]
    type = Python
    expr = 'Scalar(torch.full((${nbatch},), ${gdot_seed}, dtype=torch.float64))'
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
    ic_Scalar_names = 'flow_rate'
    ic_Scalar_values = 'gdot_ic'
  []
[]

[Models]
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
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
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
  [flow_rate_resid]
    type = PerzynaPlasticFlowRateResidual
    reference_stress = 100
    exponent = '${flow_n}'
    cutoff = '${gdot_cutoff}'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
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
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises isoharden yield_surface normality flow_rate_resid Eprate eprate Erate Eerate elasticity integrate_stress integrate_ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain flow_rate'
    residuals = 'stress_residual equivalent_plastic_strain_residual flow_rate_residual'
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
    unknowns_SR2 = 'stress'
    unknowns_Scalar = 'equivalent_plastic_strain flow_rate'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
