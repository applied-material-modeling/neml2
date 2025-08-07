[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/S_mean'
    output_SR2_values = 'S_mean'
  []
[]

[Tensors]
  [a]
    type = Scalar
    values = '1.2'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []
  [S]
    type = SR2
    values = "1 2 3 4 5 6
              2 3 4 4 5 6
              3 4 5 4 5 6
              4 5 6 4 5 6
              5 6 7 4 5 6
              6 7 8 4 5 6
              -1 -2 -3 -4 -5 -6
              -2 -3 -4 -4 -5 -6
              -3 -4 -5 -4 -5 -6
              -4 -5 -6 -4 -5 -6
              -5 -6 -7 -4 -5 -6
              -6 -7 -8 -4 -5 -6"
    batch_shape = '(12)'
  []
  [S_mean]
    type = SR2
    values = '0 0 0 0 0 0'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [model]
    type = SR2CrystalMean
    from = 'state/S'
    to = 'state/S_mean'
  []
[]
