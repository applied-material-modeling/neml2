# neml2
# Native port of tests/verification/solid_mechanics/viscoelasticity/zener/zener.i.
# Verification of ZenerElement (Standard Linear Solid) against the closed-form solution
# under a linear strain ramp followed by a hold. Reference data is in reference.csv
# (analytical strain history and analytical stress history, both in Mandel SR2 order).

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'reference.csv'
    column_names = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'reference.csv'
    column_names = 'exx eyy ezz eyz exz exy'
  []
  [stresses_ref]
    type = CSVSR2
    csv_file = 'reference.csv'
    column_names = 'sxx syy szz syz sxz sxy'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
  []
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.stress'
    SR2_values = 'stresses_ref'
    # First-order backward-Euler integration of the stiff Maxwell branch limits how tight
    # rtol can go (~2% local truncation at dt/tau = 0.05 in the ramp). 3% absorbs that while
    # still catching sign / factor / missing-term bugs.
    rtol = 3e-2
    atol = 1e-5
  []
[]

[Models]
  [zener]
    type = ZenerElement
    equilibrium_modulus = 1000
    maxwell_modulus = 5000
    maxwell_viscosity = 100
  []
  [integrate_Ev]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'zener integrate_Ev'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'viscous_strain'
    residuals = 'viscous_strain_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'viscous_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update zener'
    additional_outputs = 'viscous_strain'
  []
[]
