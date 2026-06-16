# implicit_grain_global_cross: 2-group BLOCK + DENSE implicit segment
# where the per-grain group carries explicit ``:grain`` labels AND a
# per-grain residual depends on a GLOBAL input (no sub-batch). This is
# the case that surfaced the IFT chain-rule sub-batch expand bug:
# the chain from a global input to a per-grain output produces a
# tangent that, without ``_retag_to_output``'s lower-rank-expand
# branch, would have ``sub_batch_ndim=0`` even though the output has
# ``sub_batch_ndim=1`` with a ``grain`` label -- the assembly's
# ``_normalize_block_to_cell_canonical`` then treats grain as missing
# from the block, folds nbatch into the row dim, and the per-col-var
# cat in ``_build_group_block`` fails with a ``Sizes of tensors must
# match except in dimension N`` mismatch (one col block has row=row_size,
# vorticity-like col blocks have row=nbatch×row_size).
#
# Minimal repro structure: chain ``g_glob → r_per`` where ``g_glob`` has
# no sub_batch and ``r_per`` has ``sub_batch=(grain,)``. This matches
# the mxpc ``vorticity → orientation_residual`` chain that the bug
# blocked.
#
# Closed-form solution given g_per=(1, 2, 3), g_glob=5:
#   u_glob = g_glob = 5
#   u_per  = g_per - u_glob = (-4, -3, -2)
# Same numerical result as ``implicit_cross_group_sub_batch``; what
# differs is the ``:grain`` labels on the per-grain inputs forcing
# the preserved-label assembly path through the cross-(global-input,
# per-grain-output) cell.

[Settings]
  [example_batch_shape]
    u_per     = '(2; 3)'
    g_per     = '(2; 3)'
    u_glob    = '(2,)'
    g_glob    = '(2,)'
    coupling  = '(2,)'
  []
[]

[Tensors]
  [coupling]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0], dtype=torch.float64).reshape(2))'
  []
[]

[Models]
  # INDIRECT chain g_glob -> intermediate -> r_per where:
  # * ScalarLinearCombination at `intermediate` has action `V -> 1*V`
  #   for g_glob (no broadcast against per-grain primal in the action).
  #   With per-grain ``u_per`` summed in, ``intermediate`` itself has
  #   per-grain sub_batch, but the contribution from g_glob's seed
  #   sits at sub_batch_ndim=0 at this boundary -- this is where
  #   ``_retag_to_output``'s expand branch needs to lift+label.
  # * ScalarMultiplication at `r_per` then does a NON-TRIVIAL broadcast
  #   in the chain action (`V -> coupling * intermediate * V` etc.),
  #   pulling the (still wrongly-shaped without fix) tangent through
  #   align_sub_batch, producing a tangent at r_per whose sub_batch
  #   axis carries an empty label even though the cell canonical is
  #   ``('grain',)``. The assembly then folds nbatch into the row dim
  #   and the per-col-var cat in ``_build_group_block`` fails.
  [intermediate]
    type = ScalarLinearCombination
    from = 'u_per g_glob'
    to = 'intermediate'
    weights = '1 1'
  []
  [r_per]
    type = ScalarMultiplication
    from = 'intermediate coupling'
    to = 'r_per_raw'
  []
  [r_per_match]
    type = ScalarLinearCombination
    from = 'r_per_raw g_per'
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
    models = 'intermediate r_per r_per_match r_glob'
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
