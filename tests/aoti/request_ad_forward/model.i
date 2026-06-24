# A forward-only request_AD model: the machine-learning SurrogateFlowRate leaf
# (a Python torch surrogate whose first-order chain rule is auto-derived by
# request_AD / reverse-mode autograd). Exercises the request_AD AOTI path -- the
# request_AD leaf's reverse-mode local Jacobian traces inline into the forward-mode
# chain-rule graph and is lowered through AOTInductor. Pass `-d` to select pairs.
[Models]
  [model]
    type = SurrogateFlowRate
    von_mises_stress = 'vonmises_stress'
    temperature = 'temperature'
    equivalent_plastic_strain_rate = 'equivalent_plastic_strain_rate'
  []
[]
