[Data]
  [scgeom]
    type = CrystalGeometry
    crystal_class = 'class_432'
    lattice_vectors = 'lvecs'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Tensors]
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []
  [lvecs]
    type = Vec
    batch_shape = '(3)'
    values = "1.2 0.0 0.0
              0.0 1.2 0.0
              0.0 0.0 1.2"
  []
  [class_432]
    type = SymmetryFromOrbifold
    orbifold = 432
  []
[]
