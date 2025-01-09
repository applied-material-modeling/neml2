[Tensors]
  [a0]
    type = FullTensor
    batch_shape = '(2,1)'
    base_shape = '(2,3)'
    value = 1.2
  []
  [a1]
    type = FullTensor
    batch_shape = '(2,1)'
    base_shape = '(2,3)'
    value = 300.5
  []
  [a]
    type = LogspaceTensor
    start = 'a0'
    end = 'a1'
    nstep = 100
    dim = 0
  []
[]
