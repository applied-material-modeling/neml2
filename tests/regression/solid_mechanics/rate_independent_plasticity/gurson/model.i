# neml2
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_strain]
    type = Python
    # Mandel storage: shear components carry a sqrt(2) factor. C++ FillSR2 with
    # (exx, eyy, ezz, eyz, exz, exy) = (0.1, -0.03, -0.01, -0.01, 0.02, 0.015)
    # serializes as [0.1, -0.03, -0.01, sqrt(2)*-0.01, sqrt(2)*0.02, sqrt(2)*0.015].
    expr = 'SR2(torch.tensor([0.1, -0.03, -0.01, -0.01 * 2**0.5, 0.02 * 2**0.5, 0.015 * 2**0.5], dtype=torch.float64).unsqueeze(0).expand(20, 6).contiguous())'
  []
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
  [f0]
    type = Python
    expr = 'Scalar.full(20, fill_value=0.01)'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
    ic_Scalar_names = 'void_fraction'
    ic_Scalar_values = 'f0'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'flow_invariant'
  []
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'mandel_stress'
    invariant = 'poro_invariant'
  []
  [yield_surface]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 1.25
    q2 = 1.0
    q3 = 1.57
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'j2 i1 yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [consistency]
    type = FBComplementarity
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'flow_rate'
  []
  [voidrate]
    type = GursonCavitation
  []
  [integrate_voidrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'void_fraction'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden elastic_strain elasticity
              mandel_stress j2 i1
              yield_surface normality Eprate voidrate
              consistency integrate_Ep integrate_voidrate eprate integrate_ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain void_fraction flow_rate'
    residuals = 'plastic_strain_residual equivalent_plastic_strain_residual void_fraction_residual complementarity'
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
    unknowns_SR2 = 'plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain void_fraction flow_rate'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain equivalent_plastic_strain void_fraction'
  []
[]
