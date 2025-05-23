[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/V state/Vdot'
    input_Scalar_values = '412 57'
    input_SR2_names = 'state/Ev'
    input_SR2_values = 'EV_correct'
    
    output_SR2_names = 'state/Ev_dot'
    output_SR2_values = 'Ev_rate'
  []
[]

[Tensors]
  [EV_correct]
    type = FillSR2
    values = '0.31 0.31 0.31 0 0 0'
  []
  [Ev_rate]
    type = FillSR2
    values = '0.06041221359 0.06041221359 0.06041221359 0 0 0'
  []
[]

[Models]
  [model]
    type = VolumeChangeEigenstrainRate
    volume = 'state/V'
    volume_rate = 'state/Vdot'
    eigenstrain = 'state/Ev'
    eigenstrain_rate = 'state/Ev_dot'
  []
[]
