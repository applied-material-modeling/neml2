# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingFlowSwitch.i.
# LinspaceScalar(batch_shape=(5)) -> Scalar(torch.linspace(...)) (dynamic batch, no sub-batch);
# the inline 0.5 / 0.75 inputs parse as scalar literals and broadcast against g.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'activation_energy rate_independent_flow_rate rate_dependent_flow_rate'
    input_Scalar_values = 'g 0.5 0.75'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = 'fr_correct'
  []
[]

[Models]
  [model]
    type = KocksMeckingFlowSwitch
    g0 = 0.5
    sharpness = 2.1
  []
[]

[Tensors]
  [g]
    type = Python
    expr = 'linspace(Scalar(0.1).dynamic_batch, Scalar(0.9).dynamic_batch, 5)'
  []
  [fr_correct]
    type = Python
    expr = 'Scalar([0.53927387, 0.5753837, 0.625, 0.6746163, 0.71072613])'
  []
[]
