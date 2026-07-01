# A minimal two-input model for the cpp-aoti batch-independent-input test:
# ScalarLinearCombination C = A + B + offset. test_input_broadcast supplies A
# batched and B as a 0-dim scalar (the shape MOOSE produces for a TIME force);
# the runtime must broadcast B up to the call batch.
[Models]
  [model]
    type = ScalarLinearCombination
    from = 'A B'
    to = 'C'
    offset = 2.0
  []
[]
