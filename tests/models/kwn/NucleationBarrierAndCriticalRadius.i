# Translated from tests/unit/models/kwn/NucleationBarrierAndCriticalRadius.i.
# The C++ fixture has no structural inputs (the model exposes only parameters).
# The native ModelUnitTest requires at least one JVP comparison, so promote the
# two ``allow_nonlinear`` parameters (``surface_energy``,
# ``total_gibbs_free_energy_difference``) to runtime inputs (mode-4
# ``declare_typed_parameter``) by giving them bare variable names; the driver
# then supplies them via input_Scalar_values. This matches the C++ test's
# ``check_AD_parameter_derivatives = true`` intent.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'surface_energy total_gibbs_free_energy_difference'
    input_Scalar_values = 'gamma dg_total'
    output_Scalar_names = 'state/barrier state/R_crit'
    output_Scalar_values = 'barrier R_crit'
  []
[]

[Tensors]
  [gamma]
    type = Python
    expr = 'Scalar(torch.tensor(0.2, dtype=torch.float64))'
  []
  [dg_total]
    type = Python
    expr = 'Scalar(torch.tensor(4.0, dtype=torch.float64))'
  []
  [barrier]
    type = Python
    expr = 'Scalar(torch.tensor(0.03351032163829113, dtype=torch.float64))'
  []
  [R_crit]
    type = Python
    expr = 'Scalar(torch.tensor(0.2, dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = NucleationBarrierAndCriticalRadius
    surface_energy = 'surface_energy'
    total_gibbs_free_energy_difference = 'total_gibbs_free_energy_difference'
    molar_volume = 2.0
    nucleation_barrier = 'state/barrier'
    critical_radius = 'state/R_crit'
  []
[]
