[Tensors]
  [points]
    type = LinspaceScalar
    start = 0
    end = 2
    nstep = 3
    dim = 0
    group = dynamic
  []
  [width]
    type = FullScalar
    value = 1
  []
  [height]
    type = FullScalar
    value = 2
  []
  [center]
    type = FullScalar
    value = 1
  []
  [g]
    type = GaussianScalar
    points = 'points'
    width = 'width'
    height = 'height'
    center = 'center'
  []
[]
