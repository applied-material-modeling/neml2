# Uses the external (out-of-package) model type `CtorLinAlgFailure` defined in
# `linalg_error_ctor_model.py`. The Python model's __init__ raises
# torch.linalg.LinAlgError; the eager guard (in construction mode) must
# classify this as FatalError, NOT as a recoverable ConvergenceError.
[Models]
  [model]
    type = CtorLinAlgFailure
    in_stress = 'in_stress'
    out_stress = 'out_stress'
  []
[]
