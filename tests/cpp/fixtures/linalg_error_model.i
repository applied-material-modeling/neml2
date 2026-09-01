# Uses the external (out-of-package) model type `SingularLinAlgFailure` defined in
# `linalg_error_model.py`. Loading this file requires the cpp-eager `--load` hook
# to import that module first; forward() raises torch.linalg.LinAlgError, which
# the eager guard normalizes to a recoverable neml2::aoti::ConvergenceError.
[Models]
  [model]
    type = SingularLinAlgFailure
    in_stress = 'in_stress'
    out_stress = 'out_stress'
  []
[]
