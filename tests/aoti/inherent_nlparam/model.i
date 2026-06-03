# inherent_nlparam: a parameter promoted to a runtime input via HIT (the same
# mode-3/4 path the calibration workflow uses). The exporter must see the
# pre-existing _nl_params entry on the leaf and emit a graph that takes the
# promoted value as a trailing input -- without any --parameter CLI flag.

[Models]
  [model]
    type = ScalarConstantParameter
    # `value` here is a HIT variable name (not a literal float), which triggers
    # declare_typed_parameter's mode-4 input-promotion path. The leaf adds
    # `value` to its input_spec and registers an NLParam entry for it.
    value = 'value'
  []
[]
