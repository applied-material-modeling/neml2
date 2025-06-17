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
    batch_shape = '(${ntime})'
    value = 0.01
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    time = 'forces/t'
    force_Scalar_names = 'forces/alpha'
    force_Scalar_values = 'alpha'
    ic_Scalar_names = 'state/phi_P state/phi_S state/alpha_P state/alpha_S'
    ic_Scalar_values = '0 0.3 0 0.05660377358'
    save_as = 'result.pt'
    verbose = false
    show_input_axis = false
    show_output_axis = false
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    verbose = false
  []
[]

[Models]
  [liquid_volume_fraction]
    type = ScalarLinearCombination
    from_var = 'forces/alpha'
    to_var = 'state/phi_L'
    coefficients = '${omega_Si}'
  []
  [outer_radius]
    type = CylindricalChannelGeometry
    solid_fraction = 'state/phi_S'
    product_fraction = 'state/phi_P'
    inner_radius = 'state/ri'
    outer_radius = 'state/ro'
  []
  [liquid_reactivity]
    type = HermiteSmoothStep
    argument = 'state/phi_L'
    value = 'state/R_L'
    lower_bound = 0
    upper_bound = 0.1
  []
  [solid_reactivity]
    type = HermiteSmoothStep
    argument = 'state/phi_S'
    value = 'state/R_S'
    lower_bound = 0
    upper_bound = 0.1
  []
  [reaction_rate]
    type = DiffusionLimitedReaction
    diffusion_coefficient = '${D}'
    molar_volume = '${omega_Si}'
    product_inner_radius = 'state/ri'
    solid_inner_radius = 'state/ro'
    liquid_reactivity = 'state/R_L'
    solid_reactivity = 'state/R_S'
    reaction_rate = 'state/react'
  []
  [substance_product]
    type = ScalarLinearCombination
    from_var = 'state/phi_P'
    to_var = 'state/alpha_P'
    coefficients = '${oSiCm1}'
  []
  [product_rate]
    type = ScalarVariableRate
    variable = 'state/alpha_P'
    rate = 'state/adot_P'
    time = 'forces/t'
  []
  [substance_solid]
    type = ScalarLinearCombination
    from_var = 'state/phi_S'
    to_var = 'state/alpha_S'
    coefficients = '${oCm1}'
  []
  [solid_rate]
    type = ScalarVariableRate
    variable = 'state/alpha_S'
    rate = 'state/adot_S'
    time = 'forces/t'
  []
  ##############################################
  ### IVP
  ##############################################
  [residual_phiP]
    type = ScalarLinearCombination
    from_var = 'state/adot_P state/react'
    to_var = 'residual/phi_P'
    coefficients = '1.0 -1.0'
  []
  [residual_phiL]
    type = ScalarLinearCombination
    from_var = 'state/adot_P state/adot_S'
    to_var = 'residual/phi_S'
    coefficients = '1.0 ${chem_ratio}'
  []
  [model_residual]
    type = ComposedModel
    models = "residual_phiP residual_phiL
              liquid_volume_fraction outer_radius liquid_reactivity solid_reactivity
              reaction_rate substance_product product_rate substance_solid  solid_rate"
  []
  [model_update]
    type = ImplicitUpdate
    implicit_model = 'model_residual'
    solver = 'newton'
  []
  [substance_product_new]
    type = ScalarLinearCombination
    from_var = 'state/phi_P'
    to_var = 'state/alpha_P'
    coefficients = '${oSiCm1}'
  []
  [substance_solid_new]
    type = ScalarLinearCombination
    from_var = 'state/phi_S'
    to_var = 'state/alpha_S'
    coefficients = '${oCm1}'
  []
  [model]
    type = ComposedModel
    models = 'model_update liquid_volume_fraction substance_solid_new substance_product_new'
    additional_outputs = 'state/phi_P state/phi_S'
  []
[]
