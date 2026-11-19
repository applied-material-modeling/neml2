# vf = sum_i 4/3 pi R_i^3 x_i: d vf / d x is the per-site row 4/3 pi R_i^3 ("full").

[Settings]
  [example_batch_shape]
    x = '(2; 4)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'logspace(Scalar(-1.0).sub_batch, Scalar(0.0).sub_batch, 4)'
  []
[]

[Models]
  [model]
    type = PrecipitateVolumeFraction
    number_density = 'x'
    volume_fraction = 'vf'
    radius = 'R'
  []
[]
