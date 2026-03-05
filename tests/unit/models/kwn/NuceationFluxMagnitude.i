[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/Z state/beta state/dg'
    input_Scalar_values = 'Z beta dg'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
    check_AD_parameter_derivatives = true
  []
[]

[Tensors]
  [Z]
    type = Scalar
    values = '2.0'
  []
  [beta]
    type = Scalar
    values = '3.0'
  []
  [dg]
    type = Scalar
    values = '4.0'
  []
  [J]
    type = Scalar
    values = '18.195919791379'
  []
[]

[Models]
  [model]
    type = NuceationFluxMagnitude
    zeldovich_factor = 'state/Z'
    kinetic_factor = 'state/beta'
    nucleation_barrier = 'state/dg'
    temperature = 4.0
    nucleation_site_density = 5.0
    boltzmann_constant = 2.0
    nucleation_flux_magnitude = 'state/J'
  []
[]
