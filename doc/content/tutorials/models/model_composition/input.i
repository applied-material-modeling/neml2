# Three small models from the catalog, plus a ComposedModel that wires
# them together via shared variable names.
#
#   eq1:  a_bar = I1(a)             (SR2Invariant)
#   eq2:  b_bar = vonMises(b)       (SR2Invariant)
#   eq3:  b_rate = b_bar * a + a_bar * b   (SR2LinearCombination)
#   eq:   the three glued together by ComposedModel
[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'a'
    invariant = 'a_bar'
    invariant_type = I1
  []
  [eq2]
    type = SR2Invariant
    tensor = 'b'
    invariant = 'b_bar'
    invariant_type = VONMISES
  []
  [eq3]
    type = SR2LinearCombination
    from = 'a b'
    to = 'b_rate'
    # The two weights are wired to the OUTPUTS of eq1 and eq2 by name.
    # The dependency resolver will treat 'a_bar' and 'b_bar' as
    # producer/consumer links instead of free parameters.
    weights = 'b_bar a_bar'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
