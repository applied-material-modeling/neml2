[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'Js Jt'
    input_Scalar_values = '0.8 0.4'
    input_R2_names = 'F P'
    input_R2_values = 'F P'
    output_Scalar_names = 'pc'
    output_Scalar_values = '-100.10484187836497'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [F]
    type = FillR2
    values = "0.2 0.5 0.3
              1.0 8.0 7.0
              6.0 5.0 2.0"
  []
  [P]
    type = FillR2
    values = "3.0 2.0 4.0
              9.8 1.2 3.3
              4.4 7.3 2.1"
  []
[]

[Models]
  [advstress]
    type = AdvectiveStress
    coefficient = 1.0
    js = 'Js'
    jt = 'Jt'
    deformation_gradient = 'F'
    pk1_stress = 'P'
    advective_stress = 'pc'
  []
  [model]
    type = ComposedModel
    models = 'advstress'
  []
[]
