# Combined slip strength for a RAFM-steel style model.
#
# Demonstrates a model where the slip strength is the sum of five
# contributions:
#
#   tau = tau_a + tau_SS + tau_sg + tau_mx + tau_voce
#
#   tau_a     athermal constant strength -- the ScalarLinearCombination offset
#   tau_SS    SolidSolutionStrength,  alpha*mu*eps^(3/2)*c^(1/2)
#   tau_sg    SubgrainStrength,       alpha*mu*b/lambda
#   tau_mx    PrecipitateStrength,    alpha*mu*b/(2*R*N)
#   tau_voce  the evolving `slip_hardening` state, whose rate is given by the
#             existing VoceSingleSlipHardeningRule
#
# Reference value, computed by hand from the equations above:
#   tau_a    = 50
#   tau_voce = 120                              (the input state)
#   tau_sg   = 0.25 * 80000 * 2 / 4     = 10000
#   tau_mx   = 0.4 * 100 * 2 / (2*2*4)  =     5
#   tau_SS   = 0.5 * 100 * 4^1.5 * 0.25^0.5 = 200
#   total                               = 10375
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'subgrain_size precipitate_radius precipitate_density solute_concentration slip_hardening'
    input_Scalar_values = '4.0 2.0 4.0 0.25 120.0'
    output_Scalar_names = 'slip_strength'
    output_Scalar_values = '10375.0'
  []
[]

[Models]
  [subgrain]
    type = SubgrainStrength
    alpha = 0.25
    shear_modulus = 80000.0
    burgers_vector = 2.0
  []
  [precipitate]
    type = PrecipitateStrength
    alpha = 0.4
    shear_modulus = 100.0
    burgers_vector = 2.0
  []
  [solid_solution]
    type = SolidSolutionStrength
    alpha = 0.5
    shear_modulus = 100.0
    misfit_strain = 4.0
  []
  [total]
    type = ScalarLinearCombination
    from = 'slip_hardening subgrain_strength precipitate_strength solid_solution_strength'
    to = 'slip_strength'
    weights = '1.0 1.0 1.0 1.0'
    offset = 50.0
  []
  [model]
    type = ComposedModel
    models = 'subgrain precipitate solid_solution total'
  []
[]
