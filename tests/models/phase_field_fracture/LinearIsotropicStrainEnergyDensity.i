# Translated from tests/unit/models/phase_field_fracture/LinearIsotropicStrainEnergyDensity1.i
# (the NONE-decomposition fixture). The Python-native port currently covers
# NONE and VOLDEV; SPECTRAL still requires typed linalg.eigh / ieigh primitives.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'energy'
    input_SR2_names = 'Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'psie_active psie_inactive'
    output_Scalar_values = '2443.2 0.0'
    derivative_abs_tol = 1e-4
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [Ee]
    type = Python
    expr = "SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)"
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
