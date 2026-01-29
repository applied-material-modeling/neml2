[Tensors]
  [start]
    type = FullScalar
    value = 0
  []
  [end]
    type = FullScalar
    value = 2
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
    start = 'start'
    end = 'end'
    width = 'width'
    height = 'height'
    center = 'center'
    nstep = 3
    dim = 0
    group = dynamic
  []
[]
