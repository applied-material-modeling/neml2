# neml2
# The *blended* viscosity branch of SalehaniIraniTraction: nonzero damage
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
    # eta = 1, t - t~1 = 1  ->  alpha = 1 / (1 + 1) = 0.5
    # d = 0.2 + 0.5 * (0.5276334473 - 0.2) = 0.3638167236
    input_Scalar_values = '0.5 0.5 0.5 0.2 1.0 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.3638167236'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    # T_n  = e * 0.5 * (1 - d)   = 0.8646627199
    # T_s1 = sqrt(2e) * sqrt(2)/4 * (1 - d) = 0.5244444499
    # T_s2 = same
    expr = 'Vec(torch.tensor([0.8646627199, 0.5244444499, 0.5244444499]))'
  []
[]

[Models]
  [model]
    type = SalehaniIraniTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    normal_strength = 1.0
    shear_strength = 1.0
    viscosity = 1.0
  []
[]
