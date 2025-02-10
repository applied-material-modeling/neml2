[Tensors]
  [kocks_rad]
    type = Orientation
    values = '0.1   0.2   0.3
              -0.05 0.15  0.1
              0     0     0'
    input_type = 'euler_angles'
    angle_convention = 'kocks'
    angle_type = 'radians'
  []
  [kocks_deg]
    type = Orientation
    values = '30   60   90
              -10  22   1
              0     0   0'
    input_type = 'euler_angles'
    angle_convention = 'kocks'
    angle_type = 'degrees'
  []
  [bunge_rad]
    type = Orientation
    values = '0.1   0.2   0.3
              -0.05 0.15  0.1
              0     0     0'
    input_type = 'euler_angles'
    angle_convention = 'bunge'
    angle_type = 'radians'
  []
  [bunge_deg]
    type = Orientation
    values = '30   60   90
              -10  22   1
              0     0   0'
    input_type = 'euler_angles'
    angle_convention = 'bunge'
    angle_type = 'degrees'
  []
  [roe_rad]
    type = Orientation
    values = '0.1   0.2   0.3
              -0.05 0.15  0.1
              0     0     0'
    input_type = 'euler_angles'
    angle_convention = 'roe'
    angle_type = 'radians'
  []
  [roe_deg]
    type = Orientation
    values = '30   60   90
              -10  22   1
              0     0   0'
    input_type = 'euler_angles'
    angle_convention = 'roe'
    angle_type = 'degrees'
  []
[]
