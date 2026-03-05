[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = ''
    input_Scalar_values = ''
    output_Scalar_names = 'state/barrier state/R_crit'
    output_Scalar_values = 'barrier R_crit'
    check_AD_parameter_derivatives = true
  []
[]

[Tensors]
  [gamma]
    type = Scalar
    values = '0.2'
  []
  [dg_total]
    type = Scalar
    values = '4.0'
  []
  [barrier]
    type = Scalar
    values = '0.03351032163829113'
  []
  [R_crit]
    type = Scalar
    values = '0.2'
  []
[]

[Models]
  [model]
    type = NucleationBarrierandCriticalRadius
    surface_energy = 'gamma'
    total_gibbs_free_energy_difference = 'dg_total'
    molar_volume = 2.0
    nucleation_barrier = 'state/barrier'
    critical_radius = 'state/R_crit'
  []
[]
