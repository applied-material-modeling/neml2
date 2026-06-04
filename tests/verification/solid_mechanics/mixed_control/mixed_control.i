# neml2
# Native port of tests/verification/solid_mechanics/mixed_control/mixed_control.i.
# Chaboche + Voce + Perzyna under mixed strain/stress control.
# Component 3 (Mandel slot for yz / shear yz) is stress-controlled; all other
# components are strain-controlled. Reference time / strain / stress
# trajectories come from mixed_control.csv (converted from the original
# .vtest by scripts/vtest_to_csv.py).
[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'mixed_control.csv'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'mixed_control.csv'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'mixed_control.csv'
    variable = 'stress'
  []
  # Per-step (100,) FillSR2(one, one, one, zero, one, one). SR2.fill 6-arg
  # scales slot 3 (zero) by sqrt(2) (stays zero) and slots 4/5 (one) by
  # sqrt(2). Slot 3 is the only stress-controlled component.
  [control]
    type = Python
    expr = 'SR2(torch.tensor([1.0, 1.0, 1.0, 0.0, 2.0 ** 0.5, 2.0 ** 0.5], dtype=torch.float64).reshape(1, 6).expand(100, 6).contiguous())'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'stresses_force control'
    force_SR2_values = 'stresses control'
    save_as = 'result.pt'
  []
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.stress output.E'
    SR2_values = 'stresses strains'
    atol = 1e-5
    rtol = 1e-5
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 50
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2'
    to = 'X'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    to = 'O'
    from = 'mandel_stress X'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 10
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
    from = 'mandel_stress isotropic_hardening X'
    to = 'flow_direction isotropic_hardening_direction kinematic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 155.22903539478642
    # 200 * (2/3)^(5/8)
    exponent = 4
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 5000
    g = 8.246615467370033
    # 10.1 * sqrt(2/3)
    A = 1.224744871391589e-06
    # 1.0e-6 * sqrt(3/2)
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 4.245782220824175
    # 5.2 * sqrt(2/3)
    A = 1.224744871391589e-10
    # 1.0e-10 * sqrt(3/2)
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
  [mixed]
    type = MixedControlSetup
    x_above = 'stress'
    x_below = 'E'
  []
  [y_constraint]
    type = SR2LinearCombination
    from = 'y stresses_force'
    to = 'y_residual'
    weights = '1 -1'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2 mixed y_constraint'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress E equivalent_plastic_strain X1 X2'
    residuals = 'stress_residual y_residual equivalent_plastic_strain_residual X1_residual X2_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress E X1 X2'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
