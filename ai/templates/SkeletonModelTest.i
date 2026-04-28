# Skeleton ModelUnitTest template
#
# Variable naming:
#   Bare variable names everywhere — no subspace prefixes.
#   History (old-state) values use a "~N" suffix, e.g.  S~1  for the value at the
#   previous step.  Residuals default to "<unknown>_residual".
#
# Tensor type suffixes:  Scalar, Vec, SR2, R2, WR2, R3, R4, SSR4
#
# Preferred workflow:
#   1. Use [Tensors] for Vec/SR2/R2 values — avoids component-splitting bugs.
#   2. Inline strings ("0.5") are fine for scalar inputs/outputs.
#   3. check_derivatives = true verifies set_value Jacobians via FD comparison.
#   4. Set jit = false when debugging tracing errors.
#   5. Run: ./build/dev/tests/unit/unit_tests "models"

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'

    # --- Scalar inputs (inline) ---
    input_Scalar_names = 'input_name'
    input_Scalar_values = '1.5'

    # --- Vec inputs (use [Tensors] for multi-component values) ---
    # input_Vec_names = 'displacement_jump'
    # input_Vec_values = 'u'            # reference the [Tensors] block name

    # --- SR2 inputs (use [Tensors]) ---
    # input_SR2_names = 'strain'
    # input_SR2_values = 'E'

    # --- History (old-state) inputs use the ~N suffix ---
    # input_Scalar_names = 'foo foo~1'   # value now and one step back
    # input_Scalar_values = '1.1 0.0'

    # --- Scalar outputs ---
    output_Scalar_names = 'output_name'
    output_Scalar_values = '3.0'        # expected value; compute with Python/numpy

    # --- Vec / SR2 outputs (use [Tensors]) ---
    # output_Vec_names = 'traction'
    # output_Vec_values = 'T_expected'

    # --- Residual outputs default to <unknown>_residual ---
    # output_Scalar_names = 'foo_residual'
    # output_Scalar_values = '...'

    # Derivative checks — leave true unless a specific degeneracy applies.
    # If disabled, add an explanatory comment immediately above it.
    check_values = true
    check_derivatives = true
    check_second_derivatives = true     # set false if second derivatives not implemented

    # Uncomment to debug tracing errors (skips TorchScript compilation):
    # jit = false
  []
[]

[Models]
  [model]
    type = SkeletonModel          # replace with the actual model type
    my_parameter = '2.0'          # scalar literal or a [Tensors] name
  []
[]

# [Tensors] block — define Vec, SR2, R2 values here to avoid parsing errors.
# Remove this block entirely if only scalar inputs/outputs are used.
[Tensors]
  # Example: a Vec with components (1, 0, 0)
  [u]
    type = FillVec
    values = '1 0 0'
  []

  # Example: an SR2 initialized from a single scalar (isotropic fill)
  [E]
    type = FillSR2
    values = 'e11 e22 e33'    # reference scalar [Tensors] entries, or inline
  []

  # Example: a FullScalar constant
  [e11]
    type = FullScalar
    value = 0.01
  []
[]
