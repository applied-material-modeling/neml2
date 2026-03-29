[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [newton_with_line_search]
    type = NewtonWithLineSearch
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
  [newton_sc]
    type = Newton
    linear_solver = 'schur'
  []
  [schur]
    type = SchurComplement
    residual_primary_group = '0'
    unknown_primary_group = '0'
    primary_solver = 'lu'
    schur_solver = 'lu'
  []
[]
