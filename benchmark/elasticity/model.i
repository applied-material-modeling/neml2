# neml2
[Settings]
  example_batch_shape = '(${nbatch},)'
[]

[Tensors]
  # HIT-substituted shim so the verbatim triple-quoted Python blocks below
  # can reference nbatch as a bare identifier. ${...} substitution only
  # works inside single-line single-quoted HIT strings; triple-quoted
  # blocks are passed verbatim to the Python eval namespace.
  [nbatch]
    type = Python
    expr = '${nbatch}'
  []
  # end_time = LogspaceScalar(5, 5, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.logspace(5.0, 5.0, nbatch, dtype=torch.float64))
    '''
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, nbatch)
  [times]
    type = Python
    expr = '''
      result = Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []
  # max_strain = FillSR2(0.1, -0.05, -0.05) broadcast to (nbatch, 6)
  [max_strain]
    type = Python
    expr = '''
      SR2.fill(0.1, -0.05, -0.05).dynamic_batch.expand(nbatch)
    '''
  []
  # strains = LinspaceSR2(0, max_strain, 100) -> shape (100, nbatch, 6)
  [strains]
    type = Python
    expr = '''
      SR2(
          max_strain.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1)
      )
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '1e3 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
