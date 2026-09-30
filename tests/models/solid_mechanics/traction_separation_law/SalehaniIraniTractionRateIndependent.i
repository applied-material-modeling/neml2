# neml2
# The rate-independent Salehani-Irani damage update (viscosity omitted), on the
# same loading as SalehaniIraniTraction.i -- where d_trial = 0.5276334473 already
# sits below damage~1 = 0.6, so the irreversibility cap freezes and d must come
# back as exactly 0.6. Raising damage~1 above d_trial is the whole point: on the
# advancing branch d is independent of d_old, so only a regressing trial makes
# `damage~1` load-bearing in the *value*.
# This is also the backward-compatibility guarantee -- eta = 0 has to reproduce
# the pre-viscosity law exactly.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_separation tangential_separation_1 tangential_separation_2 damage~1 t t~1'
    # x = 0.75 as above, but damage~1 = 0.6 > d_trial = 0.5276334473 -> frozen
    # alpha = 1 (inviscid branch), so d = max(d_trial, damage~1) = 0.6
    # t / t~1 are present but must not be read on this branch.
    input_Scalar_values = '0.5 0.5 0.5 0.6 1.0 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.6'
    value_abs_tol = 1e-10
    # The cap zeroes d(d)/d(separation) on this branch, so the JVP here pins the
    # *severing* of the damage-separation coupling, not its slope.
    derivative_abs_tol = 1e-6
    derivative_rel_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    # T_n  = e * 0.5 * (1 - 0.6)   = 0.5436563657
    # T_s1 = sqrt(2e) * sqrt(2)/4 * (1 - 0.6) = 0.3297442541
    # T_s2 = same
    expr = 'Vec(torch.tensor([0.5436563657, 0.3297442541, 0.3297442541]))'
  []
[]

[Models]
  [envelope]
    type = SalehaniIraniDamage
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
  []
  [update]
    type = BackwardEulerViscousDamage
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
