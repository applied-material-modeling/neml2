# Translated from tests/unit/models/solid_mechanics/traction_separation_law/BilinearTraction.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'effective_separation critical_separation full_separation
                          normal_separation normal_penetration
                          tangential_separation_1 tangential_separation_2 damage~1'
    # delta_m = 0.5, delta_c = 0.1, delta_f = 1.0, d_old = 0.0
    # d_trial = 1.0 * (0.5 - 0.1) / (0.5 * 0.9) = 0.8888888889
    # damage = max(d_trial, d_old) = 0.8888888889
    # Loading: dn = 0.02, dn_pen = 0, ds1 = 0.01, ds2 = -0.01
    # T_n  = K(1-d) dn^+ + K dn^- = 1000*0.1111111111*0.02 + 0 = 2.222222222
    # T_s1 = K(1-d) ds1 = 1000*0.1111111111*0.01 = 1.111111111
    # T_s2 = K(1-d) ds2 = 1000*0.1111111111*(-0.01) = -1.111111111
    input_Scalar_values = '0.5 0.1 1.0 0.02 0.0 0.01 -0.01 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.8888888889'
    value_abs_tol = 1e-6
    # Damage chain has stiff slope; FD truncation against the bilinear interior dominates.
    derivative_abs_tol = 1e-3
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [T_expected]
    type = Python
    expr = 'Vec(torch.tensor([2.222222222, 1.111111111, -1.111111111]))'
  []
[]

[Models]
  [model]
    type = BilinearTraction
    critical_separation = 'critical_separation'
    full_separation = 'full_separation'
    normal_penetration = 'normal_penetration'
    penalty_stiffness = 1000.0
  []
[]
