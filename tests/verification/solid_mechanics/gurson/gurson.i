[Tensors]
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'time'
  []
  [strains]
    type = SR2VTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'strain'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'gurson.vtest'
    variable = 'stress'
  []
  [f0]
    type = Scalar
    values = '0.002'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    ic_Scalar_names = 'void_fraction'
    ic_Scalar_values = 'f0'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.stress'
    SR2_values = 'stresses'
    rtol = 1e-2
    atol = 1e-5
  []
[]

[Models]
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
    invariant = 'se'
  []
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'mandel_stress'
    invariant = 'sp'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 1.25
    q2 = 1.0
    q3 = 1.57
    flow_invariant = 'se'
    poro_invariant = 'sp'
  []
  [flow]
    type = ComposedModel
    models = 'j2 i1 yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
  []
  [Eprate]
    type = AssociativePlasticFlow
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
  [voidrate]
    type = GursonCavitation
  []
  [integrate_voidrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'void_fraction'
  []
  [surface]
    type = ComposedModel
    models = "elastic_strain elasticity
              mandel_stress j2 i1
              yield normality Eprate voidrate
              consistency integrate_Ep integrate_voidrate"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain void_fraction flow_rate'
    residuals = 'plastic_strain_residual void_fraction_residual complementarity'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain'
  []
[]
