[Tensors]
  [a]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 4
    dim = 0
    group = dynamic
  []
  [b]
    type = FullScalar
    value = 3
  []
  [c]
    type = ProductUserTensorScalar
    a = 'a'
    b = 'b'
  []
[]
