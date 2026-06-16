# implicit_cross_group_sub_batch: 2-group BLOCK + DENSE implicit segment
# that exercises the cross-group off-diagonal Jacobian block where the row
# is sub-batched and the column is global. Smallest scenario that triggers
# the chain-rule sub_batch carving path (heuristic 1 prior to the typed-
# Tensor fix in es/assembled.py:_build_group_block).
#
# Unknowns:
#   * u_per   -- per-site Scalar with sub_batch=(3,)
#   * u_glob  -- global  Scalar
#
# Residuals:
#   r_per  = (u_per + u_glob) - g_per   (per-site, depends on BOTH unknowns)
#   r_glob = u_glob - g_glob            (global,   depends only on u_glob)
#
# The (per_res, u_glob) off-diagonal Jacobian block is the canonical case
# where the chain-rule pushforward emits a tangent without sub_batch axes
# (the global side has none) but the row group declares sub_batch=(3,).
# Pre-fix this assembled the wrong region split, post-fix the chain rule's
# typed wrapper carries sub_batch_ndim=0 explicitly and the assembly's
# sub_batch.broadcast_to(3) inserts the missing axis.
#
# Closed-form solution given g_per=(1, 2, 3), g_glob=5:
#   u_glob = g_glob = 5
#   u_per  = g_per - u_glob = (-4, -3, -2)
# AOTI eager-vs-compiled match is enforced by the test harness.

[Models]
  [sum_per]
    type = ScalarLinearCombination
    from = 'u_per u_glob'
    to = 'sum_per'
    weights = '1 1'
  []
  [r_per]
    type = ScalarLinearCombination
    from = 'sum_per g_per'
    to = 'r_per'
    weights = '1 -1'
  []
  [r_glob]
    type = ScalarLinearCombination
    from = 'u_glob g_glob'
    to = 'r_glob'
    weights = '1 -1'
  []
  [residual]
    type = ComposedModel
    models = 'sum_per r_per r_glob'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual'
    unknowns  = 'u_per ; u_glob'
    residuals = 'r_per ; r_glob'
  []
[]

[Solvers]
  [lu]
    type = DenseLU
  []
  [schur]
    type = SchurComplement
    residual_primary_group = '0'
    unknown_primary_group  = '0'
    primary_solver = 'lu'
    schur_solver   = 'lu'
  []
  [newton]
    type = Newton
    linear_solver = 'schur'
    abs_tol = 1e-12
    rel_tol = 1e-10
    max_its = 25
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]

[Settings]
  # Per-variable example shapes. The two per-site unknowns / givens carry
  # sub_batch=(3,); the global ones don't. Without these hints, the
  # default ((2,), ()) for every input falls back to dense layout and
  # silently bypasses the cross-group sub-batch carving the test is
  # supposed to pin.
  [example_batch_shape]
    # ``3`` labels the sub-batch axis. The composed-model resolver
    # uses these labels to track per-label structural state through
    # the chain (see ``EdgeInfo`` in ``neml2/chain_rule.py``).
    u_per  = '(2; 3)'
    u_glob = '(2,)'
    g_per  = '(2; 3)'
    g_glob = '(2,)'
  []
[]
