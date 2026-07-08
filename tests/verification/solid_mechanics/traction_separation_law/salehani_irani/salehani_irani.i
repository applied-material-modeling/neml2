# neml2
# Native port of tests/verification/solid_mechanics/traction_separation_law/salehani_irani/salehani_irani.i.
# Verification of SalehaniIraniTraction against the closed-form 3D exponential cohesive law.
# Reference data (jumps and tractions) is in reference.csv.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'reference.csv'
    column_names = 'time'
  []
  [jumps]
    type = CSVVec
    csv_file = 'reference.csv'
    column_names = 'delta_n delta_s1 delta_s2'
  []
  [tractions_ref]
    type = CSVVec
    csv_file = 'reference.csv'
    column_names = 'T_n T_s1 T_s2'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Vec_names = 'separation'
    prescribed_Vec_values = 'jumps'
  []
  [verification]
    type = Verification
    driver = 'driver'
    Vec_names = 'output.traction'
    Vec_values = 'tractions_ref'
    # No internal state and no implicit solve -- the model evaluates the analytical exponential
    # cohesive law in one shot. Tight tolerances catch any sign / factor / wiring regression.
    rtol = 1e-10
    atol = 1e-12
  []
[]

[Models]
  [decompose]
    type = VecComponents
    from = 'separation'
    to = 'normal_separation tangential_separation_1 tangential_separation_2'
  []
  [traction]
    type = SalehaniIraniTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    normal_strength = 1.0
    shear_strength = 1.0
  []
  [model]
    type = ComposedModel
    models = 'decompose traction'
  []
[]
