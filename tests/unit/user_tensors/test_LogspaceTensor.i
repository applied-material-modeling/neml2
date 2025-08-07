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
    group = dynamic
  []
  [b0]
    type = FullVec
    batch_shape = '(2,1,3)'
    intermediate_dimension = 2
    value = 1.2
  []
  [b1]
    type = FullVec
    batch_shape = '(2,1,3)'
    intermediate_dimension = 2
    value = 300.5
  []
  [b]
    type = LogspaceVec
    start = 'b0'
    end = 'b1'
    nstep = 100
    dim = -2
    group = intermediate
  []
[]
