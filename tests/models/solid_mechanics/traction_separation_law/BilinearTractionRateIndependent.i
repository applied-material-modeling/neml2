# neml2
# The rate-independent bilinear damage update (viscosity omitted), on a
# *regressing* trial damage: d_trial = 0.5555555556 sits below damage~1 = 0.6, so
# the irreversibility cap freezes and d must come back as exactly 0.6.
# This is the backward-compatibility guarantee -- eta = 0 has to reproduce the
# pre-viscosity law bit for bit -- and it is the scenario that makes
# `damage~1` load-bearing in the *value*, which the blended scenario cannot do
# (on the advancing branch d is independent of d_old at all).
# Loading is otherwise identical to BilinearTraction.i, so the two files differ
# only in the branch under test: here
#   T_n  = 1000 (1 - 0.6) 0.02 = 8.0
#   T_s1 = 1000 (1 - 0.6) 0.01 = 4.0
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'effective_separation normal_separation
                          tangential_separation_1 tangential_separation_2 damage~1 t t~1'
    # delta_m = 0.2, delta_c = 0.1, delta_f = 1.0, damage~1 = 0.6, viscosity = 0
    # d_trial = 1.0 * (0.2 - 0.1) / (0.2 * 0.9) = 0.5555555556  <  0.6  -> frozen
    # alpha = 1 (inviscid branch), so d = max(d_trial, damage~1) = 0.6
    # t / t~1 are present but must not be read on this branch.
    input_Scalar_values = '0.2 0.02 0.01 -0.01 0.6 1.0 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.6'
    value_abs_tol = 1e-10
    # The cap zeroes d(d)/d(effective_separation) on this branch, so the JVP here
    # pins the *severing* of the damage-separation coupling, not its slope.
    derivative_abs_tol = 1e-6
    derivative_rel_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    expr = 'Vec(torch.tensor([8.0, 4.0, -4.0]))'
  []
[]

[Models]
  [envelope]
    type = BilinearDamage
    critical_separation = 0.1
    full_separation = 1.0
  []
  [update]
    type = BackwardEulerViscousDamage
  []
  [traction]
    type = BilinearTraction
    penalty_stiffness = 1000.0
  []
  [model]
    type = ComposedModel
    models = 'envelope update traction'
    additional_outputs = 'damage'
  []
[]
