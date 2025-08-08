[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'energy'
    input_SR2_names = 'state/internal/Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'state/psie_active state/psie_inactive'
    output_Scalar_values = 'psie_active psie_inactive'
    value_rel_tol = 1e-4
    derivative_abs_tol = 1e-4
    second_derivative_abs_tol = 1e-3
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [Ee1]
    type = FillSR2
    values = '1e-5 1e-5 1e-5 0.02 0.06 0.03'
  []
  [Ee2]
    type = FillSR2
    values = '1e-5 1e-5 1e-5 -0.02 -0.06 0.03'
  []
  [Ee]
    type = LinspaceSR2
    start = 'Ee1'
    end = 'Ee2'
    nstep = 5
  []
  [psie_active]
    type = Scalar
    values = '0.045580 0.017770 0.007025 0.017770 0.045580'
    batch_shape = '(5)'
  []
  [psie_inactive]
    type = Scalar
    values = ' 0.030860 0.011870 0.007015 0.011870 0.030860'
    batch_shape = '(5)'
  []
[]

[Models]
  [energy]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'state/internal/Ee'
    strain_energy_density_active = 'state/psie_active'
    strain_energy_density_inactive = 'state/psie_inactive'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e1 7.8'
    decomposition = 'SPECTRAL'
  []
[]
