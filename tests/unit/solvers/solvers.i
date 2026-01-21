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
[]
