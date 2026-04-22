[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'energy'
    input_SR2_names = 'Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'psie_active psie_inactive'
    output_Scalar_values = '0.076440 0.0'
    derivative_abs_tol = 1e-4
    check_second_derivatives = true
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '1e-6 1e-6 1e-6 0.02 0.06 0.03'
  []
[]

[Models]
  [energy]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'Ee'
    active_strain_energy_density = 'psie_active'
    inactive_strain_energy_density = 'psie_inactive'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e1 7.8'
    decomposition = 'VOLDEV'
  []
[]
