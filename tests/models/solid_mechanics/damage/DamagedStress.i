# ModelUnitTest for DamagedStress:  sigma = (1 - D) * sigma_tilde
#
# Pins the per-class forward + chain-rule output for the simplest leaf of the
# Mazars CDM family. A regression here would catch a sign flip, an off-by-one
# in the Macaulay clamp (none here, just multiplication), or a wrap_like /
# typed-arithmetic break.

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'damage'
    input_Scalar_values = 'D'
    input_SR2_names = 'effective_stress'
    input_SR2_values = 'sigma_eff'
    output_SR2_names = 'stress'
    output_SR2_values = 'sigma_expected'
  []
[]

[Tensors]
  # D = 0.3
  [D]
    type = Python
    expr = 'Scalar(torch.tensor(0.3, dtype=torch.float64))'
  []
  # sigma_eff = (10, 5, -2, 1.5, -0.5, 0.7)  (Mandel storage)
  [sigma_eff]
    type = Python
    expr = 'SR2(torch.tensor([10.0, 5.0, -2.0, 1.5, -0.5, 0.7], dtype=torch.float64))'
  []
  # sigma_expected = 0.7 * sigma_eff  (since (1 - D) = 0.7)
  [sigma_expected]
    type = Python
    expr = 'SR2(torch.tensor([7.0, 3.5, -1.4, 1.05, -0.35, 0.49], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = DamagedStress
  []
[]
