ntime = 200
D = 1e-6
omega_Si = 12.0
omega_C = 5.3
omega_SiC = 12.5
chem_ratio = 1.0

oCm1 = 0.18867924528 # 1/Omega_C
oSiCm1 = 0.08 # 1/Omega_SiC

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1e4
    nstep = ${ntime}
  []
  [alpha]
    type = FullScalar
    batch_shape = '(200)'
    value = 0.01
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_Scalar_names = 'alpha'
    force_Scalar_values = 'alpha'
    ic_Scalar_names = 'phi_P phi_S alpha_P alpha_S'
    ic_Scalar_values = '0 0.3 0 0.05660377358'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [liquid_volume_fraction]
    type = ScalarLinearCombination
    from = 'alpha'
    to = 'phi_L'
    weights = '${omega_Si}'
  []
  [outer_radius]
    type = CylindricalChannelGeometry
    solid_fraction = 'phi_S'
    product_fraction = 'phi_P'
    inner_radius = 'ri'
    outer_radius = 'ro'
  []
  [liquid_reactivity]
    type = HermiteSmoothStep
    argument = 'phi_L'
    value = 'R_L'
    lower_bound = 0
    upper_bound = 0.1
  []
  [solid_reactivity]
    type = HermiteSmoothStep
    argument = 'phi_S'
    value = 'R_S'
    lower_bound = 0
    upper_bound = 0.1
  []
  [reaction_rate]
    type = DiffusionLimitedReaction
    diffusion_coefficient = '${D}'
    molar_volume = '${omega_Si}'
    product_inner_radius = 'ri'
    solid_inner_radius = 'ro'
    liquid_reactivity = 'R_L'
    solid_reactivity = 'R_S'
    reaction_rate = 'react'
  []
  [substance_product]
    type = ScalarLinearCombination
    from = 'phi_P'
    to = 'alpha_P'
    weights = '${oSiCm1}'
  []
  [product_rate]
    type = ScalarVariableRate
    variable = 'alpha_P'
  []
  [substance_solid]
    type = ScalarLinearCombination
    from = 'phi_S'
    to = 'alpha_S'
    weights = '${oCm1}'
  []
  [solid_rate]
    type = ScalarVariableRate
    variable = 'alpha_S'
  []
  ##############################################
  ### IVP
  ##############################################
  [residual_phiP]
    type = ScalarLinearCombination
    from = 'alpha_P_rate react'
    to = 'phi_P_residual'
    weights = '1.0 -1.0'
  []
  [residual_phiL]
    type = ScalarLinearCombination
    from = 'alpha_P_rate alpha_S_rate'
    to = 'phi_S_residual'
    weights = '1.0 ${chem_ratio}'
  []
  [model_residual]
    type = ComposedModel
    models = "residual_phiP residual_phiL
              liquid_volume_fraction outer_radius liquid_reactivity solid_reactivity
              reaction_rate substance_product product_rate substance_solid  solid_rate"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'model_residual'
    unknowns = 'phi_P phi_S'
    residuals = 'phi_P_residual phi_S_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    verbose = false
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model_update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [substance_product_new]
    type = ScalarLinearCombination
    from = 'phi_P'
    to = 'alpha_P'
    weights = '${oSiCm1}'
  []
  [substance_solid_new]
    type = ScalarLinearCombination
    from = 'phi_S'
    to = 'alpha_S'
    weights = '${oCm1}'
  []
  [model]
    type = ComposedModel
    models = 'model_update liquid_volume_fraction substance_solid_new substance_product_new'
    additional_outputs = 'phi_P phi_S'
  []
[]
