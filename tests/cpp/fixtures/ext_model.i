# Uses the external (out-of-package) model type `ExtScaleStress` defined in
# `ext_model.py`. Loading this file requires the cpp-eager `--load` hook to import
# that module first; otherwise `ExtScaleStress` is an unknown type.
[Models]
  [model]
    type = ExtScaleStress
    in_stress = 'in_stress'
    out_stress = 'out_stress'
  []
[]
