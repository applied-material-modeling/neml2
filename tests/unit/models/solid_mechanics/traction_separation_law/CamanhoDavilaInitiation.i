[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'beta dn_pos'
    # beta = sqrt(2), dn_pos = 0.5 (any positive value triggers the opening branch)
    input_Scalar_values = 'beta_val 0.5'
    output_Scalar_names = 'delta_init'
    # K=1000, N=10, S=10 -> δ_n0 = δ_s0 = 0.01.
    # β² = 2 -> δ_mixed = sqrt(0.0003) = 0.0173205081
    # δ_init = 0.01 * 0.01 * sqrt(3) / 0.0173205081 = 0.0100000000
    output_Scalar_values = '0.01'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [beta_val]
    type = Scalar
    values = '1.41421356237'
  []
[]

[Models]
  [model]
    type = CamanhoDavilaInitiation
    mixity = 'beta'
    normal = 'dn_pos'
    to = 'delta_init'
    penalty_stiffness = 1000.0
    normal_strength = 10.0
    shear_strength = 10.0
  []
[]
