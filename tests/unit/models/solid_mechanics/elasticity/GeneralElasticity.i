[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/internal/Ee'
    input_SR2_values = 'Ee'
    input_Rot_names = 'state/orientation'
    input_Rot_values = 'R'
    output_SR2_names = 'state/S'
    output_SR2_values = 'S'
    derivative_abs_tol = 1e-6
    derivative_rel_tol = 1e-4
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '0.09 0.04 -0.02'
  []
  [R]
    type = FillRot
    values = '0.13991834 0.18234513 0.85043991'
  []
  [S]
    type = FillSR2
    values = '10.14791738  4.65043712 -2.33254551 -0.74329637 1.01251401 1.25050411'
  []
  [C]
    type = SSR4
    values = " 100  2 3  4  5  6
               7  150  9 10 11 12
              13 14 300 16 17 18
              19 20 21 150 23 24
              25 26 27 28 200 30
              31 32 33 34 35 100"
    batch_shape = '(10)'
  []
[]

[Models]
  [model]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'C'
  []
[]
