# neml2
# The initial state promoted to runtime inputs. f0 and N0 enter only through
# the constant offset NV0, so a missing action leaves the value right and the
# tangent zero -- invisible to a value-only check, which is why the driver's
# JVP comparison is the point of this fixture.
#
# c   = 4 * 0.61 / (3 sqrt(pi)) = 0.4589597020
# NV  = c * 0.6^1.5  / sqrt(3.0) = 0.1231528940
# NV0 = c * 0.01^1.5 / sqrt(2.0) = 0.0003245000
# opening = 0.1228283940
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction cavity_density initial_cavity_fraction initial_cavity_density'
    input_Scalar_values = 'f N f0 N0'
    output_Scalar_names = 'viscous_normal_opening'
    output_Scalar_values = 'opening'
    value_abs_tol = 1e-9
  []
[]

[Tensors]
  [f]
    type = Python
    expr = 'Scalar(0.6)'
  []
  [N]
    type = Python
    expr = 'Scalar(3.0)'
  []
  [f0]
    type = Python
    expr = 'Scalar(0.01)'
  []
  [N0]
    type = Python
    expr = 'Scalar(2.0)'
  []
  [opening]
    type = Python
    expr = '''
      Scalar(
          4.0 * 0.61 / (3.0 * math.sqrt(math.pi))
          * (0.6**1.5 / math.sqrt(3.0) - 0.01**1.5 / math.sqrt(2.0))
      )
    '''
  []
[]

[Models]
  [model]
    type = CavityVolumeOpening
    cavity_shape_factor = 0.61
    initial_cavity_fraction = 'initial_cavity_fraction'
    initial_cavity_density = 'initial_cavity_density'
  []
[]
