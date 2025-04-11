[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/V state/Vdot state/Ev'
    input_Scalar_values = '412 57 EV_correct'
    
    output_Scalar_names = 'state/Ev_dot'
    output_Scalar_values = 'Ev_rate'
  []
[]

[Tensors]
  [EV_correct]
    type = Scalar
    values = '0.31'
  []
  [Ev_rate]
    type = Scalar
    values = '0.06041221359'
  []
[]

[Models]
  [model]
    type = ScalarVolumeChangeEigenstrainRate
    volume = 'state/V'
    volume_rate = 'state/Vdot'
    eigenstrain = 'state/Ev'
    eigenstrain_rate = 'state/Ev_dot'
  []
[]
