# forward_provider_param: a ScalarConstantParameter PROVIDER (`k`) feeds its
# value as the nonlinear weight of a ScalarLinearCombination. Promoting the
# provider's `k.value` exercises the spec-caching-container rebuild (the
# dependency resolver places the promoted provider input ahead of the
# structural input, so the graph signature must be reordered structural-then-
# param to match the C++ runtime feed). Regression for the silent-wrong-
# derivative bug found battle-testing km_mixed_model.
[Models]
  [k]
    type = ScalarConstantParameter
    value = 3.0
  []
  [rate]
    type = ScalarLinearCombination
    from = 'x'
    to = 'y'
    weights = 'k'
  []
  [model]
    type = ComposedModel
    models = 'rate'
  []
[]
