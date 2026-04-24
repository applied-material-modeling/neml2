[Models]
  [eq1]
    type = ScalarVariableRate
    variable = 'a'
  []
  [eq2]
    type = ScalarVariableRate
    variable = 'b'
  []
  [eq3]
    type = ScalarVariableRate
    variable = 'c'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
