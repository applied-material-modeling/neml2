# M (2 rows x 3 cols, sub_batch_ndim=2) contracted with u (3, sub_batch_ndim=1)
# gives J = M @ u (2, sub_batch_ndim=1). ModelUnitTest auto-checks the analytic
# pushforward for both M and u against autograd.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'M u'
    input_Scalar_values = 'M u'
    output_Scalar_names = 'J'
    output_Scalar_values = 'J'
  []
[]

[Tensors]
  [M]
    type = Python
    expr = 'Scalar(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), sub_batch_ndim=2)'
  []
  [u]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 3.0]), sub_batch_ndim=1)'
  []
  [J]
    type = Python
    expr = 'Scalar(torch.tensor([14.0, 32.0]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = IntermediateLinearContraction
    operator = 'M'
    field = 'u'
    out = 'J'
  []
[]
