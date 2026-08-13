# ModelUnitTest for MazarsEquivalentStrain:  eps_tilde = sqrt(sum_i <eps_i>_+^2)
#
# Pins the per-class forward + chain-rule output. The biaxial mixed state
# exercises an eigendecomposition with all three principal strains distinct
# (no degeneracy) and a mix of positive / negative principals (so the
# Macaulay bracket actually filters). The expected value was computed and
# verified against v2 to bit-identical precision in Phase H.

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain'
    input_SR2_values = 'E'
    output_Scalar_names = 'equivalent_strain'
    output_Scalar_values = 'eq_expected'
  []
[]

[Tensors]
  # Biaxial mixed strain — all 3 principal directions distinct, all 3 in [-3e-4, +5e-4]
  # Mandel storage: (xx, yy, zz, sqrt2*yz, sqrt2*xz, sqrt2*xy)
  [E]
    type = Python
    expr = 'SR2(torch.tensor([+5e-4, -3e-4, +1e-4, 2e-5, -1e-5, 3e-5], dtype=torch.float64))'
  []
  # Expected equivalent strain — computed by torch.linalg.eigh + macaulay + norm,
  # verified bit-identical to v2 in Phase H Layer 1.
  [eq_expected]
    type = Python
    expr = 'Scalar(torch.tensor(5.106413411461524e-04, dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = MazarsEquivalentStrain
  []
[]
