# A minimal model that routes through the neml2::opaque_pow custom op:
# PerzynaPlasticFlowRate computes flow_rate = (yield_function / reference_stress)^exponent
# via opaque_pow (the Inductor fusion barrier from neml2/types/functions.py).
# Compiled by the custom_op_fixture_compile ctest fixture and loaded by
# test_custom_ops through the Python-free cpp-aoti runtime, which must register
# neml2::opaque_pow in libneml2 itself.
[Models]
  [model]
    type = PerzynaPlasticFlowRate
    reference_stress = 150
    exponent = 6
  []
[]
