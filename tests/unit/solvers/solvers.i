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
    primary_group = '0'
    primary_solver = 'lu'
    schur_group = '1'
    schur_solver = 'lu'
  []
[]
