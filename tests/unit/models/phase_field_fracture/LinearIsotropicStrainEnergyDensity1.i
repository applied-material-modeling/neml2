[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'energy'
    input_SR2_names = 'Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'psie_active psie_inactive'
    output_Scalar_values = '2443.2 0.0'
    derivative_abs_tol = 0
    derivative_rel_tol = 1e-3
    check_second_derivatives = true
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '0.1 0.05 -0.03 0.02 0.06 0.03'
  []
[]

[Models]
  [energy]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'Ee'
    active_strain_energy_density = 'psie_active'
    inactive_strain_energy_density = 'psie_inactive'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e5 7.8e4'
    decomposition = 'NONE'
  []
[]
