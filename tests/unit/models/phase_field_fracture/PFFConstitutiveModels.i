## Checks against residual_almost (using the value of AT2 instead of derivatives and ignoring non-local terms)
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/d'
    input_Scalar_values = '0.3 0.787 0.982'
    input_SR2_names = 'state/internal/Ee'
    input_SR2_values = 'Ee'
    output_Scalar_names = 'state/residual_almost'
    output_Scalar_values = '1197.4148 111.4794279 1.75602448'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [p]
    type = Scalar
    values = 2
  []
  [Ee]
    type = FillSR2
    values = '0.1 0.05 -0.03 0.02 0.06 0.03'
  []
  [Gc]
    type = Scalar
    values = 1
  []
  [l]
    type = Scalar
    values = 1
  []
  [Co]
    type = Scalar
    values = 1
  []
[]

[Models]
  [degrade]
    type = PowerDegradationFunction
    damage = 'state/d'
    degradation = 'state/g'
    power = 'p'
  []
  [cracked]
    type = CrackGeometricFunctionAT2
    damage = 'state/d'
    crack = 'state/alpha'
  []
  [energy]
    type = ElasticStrainEnergyDensity
    strain = 'forces/E'
    elastic_strain_energy = 'state/psie'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e5 7.8e4'
  []
  [degraded_energy]
    type = ScalarMultiplication
    from_var = 'state/g state/psie'
    to_var = 'state/dgraded'
  []
  [model]
    type = ScalarLinearCombination
    from_var = 'state/alpha state/dgraded'
    to_var = 'state/residual_almost'
    coefficients = 'Gc/l/Co 1'
  []
[]