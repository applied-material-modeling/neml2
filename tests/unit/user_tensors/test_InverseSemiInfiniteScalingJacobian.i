[Tensors]
  [x]
    type = LinspaceScalar
    start = 0
    end = 0.6
    nstep = 4
    dim = 0
    group = dynamic
  []
  [s]
    type = FullScalar
    value = 2
  []
  [y]
    type = InverseSemiInfiniteScalingJacobianScalar
    x = 'x'
    s = 's'
  []
[]
