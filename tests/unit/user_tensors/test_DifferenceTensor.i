[Tensors]
  [points]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 4
    dim = 0
    group = intermediate
  []
  [diff]
    type = DifferenceScalar
    points = 'points'
  []
[]
