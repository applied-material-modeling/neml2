# Batched-magnitude variant of DumpInSmallestBin.i: ``magnitude`` carries a
# leading batch axis (size 3, e.g. a per-temperature KWN sweep) while
# cell_centers stays a global per-cell sub-batch (3 cells). The output must be
# batched (3, 3) with the magnitude in the first cell of each batch entry. This
# guards the zero-tail broadcast in DumpInSmallestBin.forward -- sourcing the
# tail from cell_centers (no batch dim) regressed the value path on a cat rank
# mismatch.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mag'
    input_Scalar_values = 'mag'
    output_Scalar_names = 'state/source'
    output_Scalar_values = 'source'
  []
[]

[Tensors]
  [mag]
    type = Python
    expr = 'Scalar(torch.tensor([2.5, 3.5, 5.0]))'
  []
  [centers]
    type = Python
    expr = 'Scalar(torch.tensor([0.1, 0.5, 1.0]), sub_batch_ndim=1)'
  []
  [source]
    type = Python
    expr = 'Scalar(torch.tensor([[2.5, 0.0, 0.0], [3.5, 0.0, 0.0], [5.0, 0.0, 0.0]]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = DumpInSmallestBin
    magnitude = 'state/mag'
    cell_centers = 'centers'
    dumped_source = 'state/source'
  []
[]
