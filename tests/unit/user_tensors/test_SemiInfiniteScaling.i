[Tensors]
  [x]
    type = LinspaceScalar
    start = 0
    end = 0.75
    nstep = 4
    dim = 0
    group = dynamic
  []
  [s]
    type = FullScalar
    value = 2
  []
  [y]
    type = SemiInfiniteScalingScalar
    x = 'x'
    s = 's'
  []
[]
