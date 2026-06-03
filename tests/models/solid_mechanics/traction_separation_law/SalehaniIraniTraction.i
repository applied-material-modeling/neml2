# Translated from tests/unit/models/solid_mechanics/traction_separation_law/SalehaniIraniTraction.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_separation tangential_separation_1 tangential_separation_2 damage~1'
    input_Scalar_values = '0.5 0.5 0.5 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    # x = 0.5 + 0.125 + 0.125 = 0.75
    # d = 1 - exp(-0.75) = 0.5276334473
    output_Scalar_values = '0.5276334473'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    # Same numerical values as the original (no-state) law because (1 - damage) == exp(-x)
    # in the advancing branch (d_old = 0 here).
    # T_n  = e        * 0.5       * exp(-0.75) ~ 0.6420127083
    # T_s1 = sqrt(2e) * sqrt(2)/4 * exp(-0.75) ~ 0.3894003915
    # T_s2 = same
    expr = 'Vec(torch.tensor([0.6420127083, 0.3894003915, 0.3894003915]))'
  []
[]

[Models]
  [model]
    type = SalehaniIraniTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    normal_strength = 1.0
    shear_strength = 1.0
  []
[]
