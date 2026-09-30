# neml2
# The *blended* exponential viscosity branch of the Salehani-Irani graph: nonzero damage
# history and a nonzero viscosity, so `damage~1` is load-bearing in the pinned
# value (a bug that ignored it outright would pass) and `alpha` is pinned away
# from 1. The rate-independent branch is pinned separately by
# SalehaniIraniTractionRateIndependent.i; the cap / blend / partial identities
# themselves are pinned in tests/unit/test_viscous_damage.py.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_separation tangential_separation_1 tangential_separation_2 damage~1 t t~1'
    # x = 0.5 + 0.5^2/2 + 0.5^2/2 = 0.75
    # d_trial = 1 - exp(-0.75) = 0.5276334473
    # eta = 1, t - t~1 = 1  ->  alpha = 1 - exp(-1) = 0.6321205588
    # d = 0.2 + alpha * (0.5276334473 - 0.2) = 0.4071038378
    input_Scalar_values = '0.5 0.5 0.5 0.2 1.0 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.4071038378'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    # T_n  = e * 0.5 * (1 - d)   = 0.8058294320
    # T_s1 = sqrt(2e) * sqrt(2)/4 * (1 - d) = 0.4887602570
    # T_s2 = same
    expr = 'Vec(torch.tensor([0.8058294320, 0.4887602570, 0.4887602570]))'
  []
[]

[Models]
  [envelope]
    type = SalehaniIraniDamage
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
  []
  [update]
    type = ExponentialViscousDamage
    viscosity = 1.0
  []
  [traction]
    type = SalehaniIraniTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    normal_strength = 1.0
    shear_strength = 1.0
  []
  [model]
    type = ComposedModel
    models = 'envelope update traction'
    additional_outputs = 'damage'
  []
[]
