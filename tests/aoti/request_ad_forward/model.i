# A forward-only request_AD model: the machine-learning SurrogateFlowRate leaf
# (a Python torch surrogate whose first-order chain rule is auto-derived by
# request_AD / reverse-mode autograd). Exercises the request_AD AOTI path -- the
# forward segment's Jacobian is emitted by the reverse-mode _InputJacobianADModule
# (not the forward-mode chain rule), lowered through AOTInductor. No -d needed:
# request_AD auto-enables d(out)/d(in) for every (output, structural input) pair.
[Models]
  [model]
    type = SurrogateFlowRate
    von_mises_stress = 'vonmises_stress'
    temperature = 'temperature'
    equivalent_plastic_strain_rate = 'equivalent_plastic_strain_rate'
  []
[]
