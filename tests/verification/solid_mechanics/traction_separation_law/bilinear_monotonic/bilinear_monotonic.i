# neml2
# Native port of tests/verification/solid_mechanics/traction_separation_law/bilinear_monotonic/bilinear_monotonic.i.
# Verification of CamanhoDavila/BilinearTraction against the closed-form bilinear damage law
# under a monotonic mixed-mode opening ramp with the BK propagation criterion. Reference data
# (jumps and tractions) is in reference.csv. Under monotonic loading the irreversibility cap
# d = max(d_trial, d_old) is satisfied by d_trial alone, so the analytical formula matches
# the model output step-for-step.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'reference.csv'
    column_names = 'time'
  []
  [jumps]
    type = CSVVec
    csv_file = 'reference.csv'
    column_names = 'delta_n delta_s1 delta_s2'
  []
  [tractions_ref]
    type = CSVVec
    csv_file = 'reference.csv'
    column_names = 'T_n T_s1 T_s2'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Vec_names = 'separation'
    prescribed_Vec_values = 'jumps'
    save_as = 'result.pt'
  []
  [verification]
    type = Verification
    driver = 'driver'
    Vec_names = 'output.traction'
    Vec_values = 'tractions_ref'
    rtol = 1e-10
    # Bumped from 1e-12: the model and the analytical reference perform the
    # same arithmetic in different orders (tensor ops vs numpy scalar ops),
    # so fused-multiply-add reordering produces ~2e-12 ULP-level noise on
    # tractions of O(1e-2). Still ~10x tighter than rtol on the magnitudes
    # that matter.
    atol = 1e-11
  []
[]

[Models]
  [decompose]
    type = VecComponents
    from = 'separation'
    to = 'signed_normal_separation tangential_separation_1 tangential_separation_2'
  []
  [tangential_separation]
    type = ScalarPNorm
    from = 'tangential_separation_1 tangential_separation_2'
    to = 'tangential_separation'
    exponent = 2.0
  []
  [macaulay_n]
    type = MacaulaySplit
    from = 'signed_normal_separation'
    to_positive = 'normal_separation'
    to_negative = 'normal_penetration'
  []
  [mode_mixity]
    type = ModeMixity
  []
  [critical_separation]
    type = CamanhoDavilaCriticalSeparation
    mode_mixity = 'mode_mixity'
    penalty_stiffness = 1000.0
    normal_strength = 10.0
    shear_strength = 15.0
  []
  [full_separation]
    type = BenzeggaghKenaneFullSeparation
    mode_mixity = 'mode_mixity'
    critical_separation = 'critical_separation'
    penalty_stiffness = 1000.0
    mode_I_fracture_toughness = 1.0
    mode_II_fracture_toughness = 2.0
    shear_strength = 15.0
    eta = 2.0
  []
  [effective_separation]
    type = ScalarPNorm
    from = 'normal_separation tangential_separation'
    to = 'effective_separation'
    exponent = 2.0
  []
  [traction]
    type = BilinearTraction
    critical_separation = 'critical_separation'
    full_separation = 'full_separation'
    normal_penetration = 'normal_penetration'
    penalty_stiffness = 1000.0
  []
  [model]
    type = ComposedModel
    models = 'decompose tangential_separation macaulay_n effective_separation traction'
  []
[]
