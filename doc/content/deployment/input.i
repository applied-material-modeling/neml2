# Small worked example for the CLI utilities tutorial: a linear isotropic
# elastic model driven through a 5-step uniaxial tension history. The same
# input is reused across the neml2-run / neml2-inspect / neml2-compile
# subsections.

[Tensors]
  [times]
    type = Python
    expr = 'linspace(Scalar(0).dynamic_batch, Scalar(1).dynamic_batch, 5)'
  []
  [strains]
    type = Python
    expr = 'SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 6) * torch.linspace(0.0, 1.0, 5, dtype=torch.float64).reshape(5, 1))'
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'elasticity'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
    save_as = 'result.pt'
  []
[]
