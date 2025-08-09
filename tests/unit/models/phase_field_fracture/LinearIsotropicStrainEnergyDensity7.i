[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'energy'
    input_SR2_names = 'state/internal/Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'state/psie_active state/psie_inactive'
    output_Scalar_values = '1711.17 577.627'
    derivative_abs_tol = 0
    derivative_rel_tol = 1e-3
    second_derivative_abs_tol = 0
    second_derivative_rel_tol = 1e-3
    check_second_derivatives = true
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '0.1 -0.08 0.03 0.02 0.06 0.03'
  []
[]

[Models]
  [energy]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'state/internal/Ee'
    strain_energy_density_active = 'state/psie_active'
    strain_energy_density_inactive = 'state/psie_inactive'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e5 7.8e4'
    decomposition = 'SPECTRAL'
  []
[]
