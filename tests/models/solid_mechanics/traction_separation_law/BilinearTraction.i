# neml2
# The *blended* viscosity branch of BilinearTraction: nonzero damage history and
# a nonzero viscosity, so `damage~1` is load-bearing in the pinned value (a bug
# that ignored it outright would pass) and `alpha` is pinned away from 1.
# The rate-independent branch is pinned separately by
# BilinearTractionRateIndependent.i; the cap / blend / partial identities
# themselves are pinned in tests/unit/test_viscous_damage.py.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'effective_separation critical_separation full_separation
                          normal_separation normal_penetration
                          tangential_separation_1 tangential_separation_2 damage~1 t t~1'
    # delta_m = 0.5, delta_c = 0.1, delta_f = 1.0, damage~1 = 0.4
    # d_trial = 1.0 * (0.5 - 0.1) / (0.5 * 0.9) = 0.8888888889
    # eta = 1, t - t~1 = 1  ->  alpha = 1 / (1 + 1) = 0.5
    # d = 0.4 + 0.5 * (0.8888888889 - 0.4) = 0.6444444444
    # Loading: dn = 0.02, dn_pen = 0, ds1 = 0.01, ds2 = -0.01
    # T_n  = K(1-d) dn + K dn_pen = 7.1111111111
    # T_s1 = K(1-d) ds1            = 3.5555555556
    # T_s2 = K(1-d) ds2            = -3.5555555556
    input_Scalar_values = '0.5 0.1 1.0 0.02 0.0 0.01 -0.01 0.4 1.0 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.6444444444'
    value_abs_tol = 1e-6
    # Damage chain has stiff slope; FD truncation against the bilinear interior dominates.
    derivative_abs_tol = 1e-3
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [T_expected]
    type = Python
    expr = 'Vec(torch.tensor([7.1111111111, 3.5555555556, -3.5555555556]))'
  []
[]

[Models]
  [model]
    type = BilinearTraction
    critical_separation = 'critical_separation'
    full_separation = 'full_separation'
    normal_penetration = 'normal_penetration'
    penalty_stiffness = 1000.0
    viscosity = 1.0
  []
[]
