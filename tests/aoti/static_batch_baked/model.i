# static_batch_baked: exercise dynamic_batch=false with a baked rank-1
# parameter. `E_per_batch` is a (2,)-shaped Scalar bound to the Young's
# modulus -- each batch element gets its own E. That parameter would
# specialize the dynamic batch dim under the default `dynamic_batch=true`
# (torch.export would emit "you marked batch as dynamic but specialized
# it to constant 2"); declaring `dynamic_batch = false` here pins the
# artifact at batch=2 by design, which is the principled escape hatch for
# baked per-batch parameters.

[Tensors]
  [E_per_batch]
    type = Python
    expr = '''
      Scalar(torch.tensor([100.0, 200.0], dtype=torch.float64))
    '''
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = 'E_per_batch 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'strain'
    stress = 'stress'
  []
[]

[Settings]
  example_batch_shape = '(2,)'
  dynamic_batch = false
[]
