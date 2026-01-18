[Solvers]
  [newton]
    type = Newton
    verbose = false
    linear_solver = 'lu'
  []
  [newton_with_line_search]
    type = NewtonWithLineSearch
    verbose = false
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
