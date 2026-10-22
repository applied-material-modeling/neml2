# forward_sub_batch_plain_output: a forward segment whose FIRST input is
# sub-batched and which also returns a plain-batch output. `forward`
# materializes every output against the call's dynamic batch; taking that
# batch from the first input with only its base stripped would keep the
# sub-batch axis (4,) and broadcast the plain output `y` from (2,) to (2, 4).
# Pins output_sub_batch_shapes() as [[4], []] and both outputs' shapes.

[Models]
  [scale_x]
    type = ScalarLinearCombination
    from = 'x'
    to = 'x_scaled'
    weights = '2.0'
  []
  [scale_t]
    type = ScalarLinearCombination
    from = 't'
    to = 'y'
    weights = '3.0'
  []
  [model]
    type = ComposedModel
    models = 'scale_x scale_t'
  []
[]

[Settings]
  [example_batch_shape]
    x = '(2; 4)'
    t = '(2,)'
  []
[]
