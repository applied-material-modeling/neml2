# The same three models without a ComposedModel — the caller is
# responsible for evaluating them in the right order and threading
# intermediate values into eq3's weight parameters by hand.
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
    weights = '1 1'
  []
[]
