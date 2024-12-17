[Tensors]
  [a0]
    type = FullScalar
    batch_shape = '(2,1)'
    value = 1.3
  []
  [a1]
    type = FullScalar
    batch_shape = '(2,1)'
    value = 100.5
  []
  [a]
    type = LogspaceScalar
    start = 'a0'
    end = 'a1'
    nstep = 100
    dim = 0
  []

  [b0]
    type = FullVec
    batch_shape = '(2,1)'
    value = 1.3
  []
  [b1]
    type = FullVec
    batch_shape = '(2,1)'
    value = 555
  []
  [b]
    type = LogspaceVec
    start = 'b0'
    end = 'b1'
    nstep = 100
    dim = 0
  []

  [c0]
    type = FullRot
    batch_shape = '(2,1)'
    value = 1.3
  []
  [c1]
    type = FullRot
    batch_shape = '(2,1)'
    value = 2
  []
  [c]
    type = LogspaceRot
    start = 'c0'
    end = 'c1'
    nstep = 100
    dim = 0
  []

  [d0]
    type = FullR2
    batch_shape = '(2,1)'
    value = 1.3
  []
  [d1]
    type = FullR2
    batch_shape = '(2,1)'
    value = 121
  []
  [d]
    type = LogspaceR2
    start = 'd0'
    end = 'd1'
    nstep = 100
    dim = 0
  []

  [e0]
    type = FullSR2
    batch_shape = '(2,1)'
    value = 1.3
  []
  [e1]
    type = FullSR2
    batch_shape = '(2,1)'
    value = 11
  []
  [e]
    type = LogspaceSR2
    start = 'e0'
    end = 'e1'
    nstep = 100
    dim = 0
  []

  [f0]
    type = FullR3
    batch_shape = '(2,1)'
    value = 1.3
  []
  [f1]
    type = FullR3
    batch_shape = '(2,1)'
    value = 1.4
  []
  [f]
    type = LogspaceR3
    start = 'f0'
    end = 'f1'
    nstep = 100
    dim = 0
  []

  [g0]
    type = FullSFR3
    batch_shape = '(2,1)'
    value = 1.3
  []
  [g1]
    type = FullSFR3
    batch_shape = '(2,1)'
    value = 999
  []
  [g]
    type = LogspaceSFR3
    start = 'g0'
    end = 'g1'
    nstep = 100
    dim = 0
  []

  [h0]
    type = FullR4
    batch_shape = '(2,1)'
    value = 1.3
  []
  [h1]
    type = FullR4
    batch_shape = '(2,1)'
    value = 2
  []
  [h]
    type = LogspaceR4
    start = 'h0'
    end = 'h1'
    nstep = 100
    dim = 0
  []

  [i0]
    type = FullSFR4
    batch_shape = '(2,1)'
    value = 1.3
  []
  [i1]
    type = FullSFR4
    batch_shape = '(2,1)'
    value = 1211
  []
  [i]
    type = LogspaceSFR4
    start = 'i0'
    end = 'i1'
    nstep = 100
    dim = 0
  []

  [j0]
    type = FullWFR4
    batch_shape = '(2,1)'
    value = 1.3
  []
  [j1]
    type = FullWFR4
    batch_shape = '(2,1)'
    value = 1211
  []
  [j]
    type = LogspaceWFR4
    start = 'j0'
    end = 'j1'
    nstep = 100
    dim = 0
  []

  [k0]
    type = FullSSR4
    batch_shape = '(2,1)'
    value = 1.3
  []
  [k1]
    type = FullSSR4
    batch_shape = '(2,1)'
    value = 1211
  []
  [k]
    type = LogspaceSSR4
    start = 'k0'
    end = 'k1'
    nstep = 100
    dim = 0
  []

  [l0]
    type = FullR5
    batch_shape = '(2,1)'
    value = 1.3
  []
  [l1]
    type = FullR5
    batch_shape = '(2,1)'
    value = 55
  []
  [l]
    type = LogspaceR5
    start = 'l0'
    end = 'l1'
    nstep = 100
    dim = 0
  []

  [m0]
    type = FullSSFR5
    batch_shape = '(2,1)'
    value = 1.3
  []
  [m1]
    type = FullSSFR5
    batch_shape = '(2,1)'
    value = 13
  []
  [m]
    type = LogspaceSSFR5
    start = 'm0'
    end = 'm1'
    nstep = 100
    dim = 0
  []
[]
