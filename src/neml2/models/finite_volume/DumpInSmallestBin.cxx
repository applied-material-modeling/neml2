// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/finite_volume/DumpInSmallestBin.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
register_NEML2_object(DumpInSmallestBin);

OptionSet
DumpInSmallestBin::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Dump the source magnitude into the smallest cell-center bin.";

  options.set_input("magnitude");
  options.set("magnitude").doc() = "Source magnitude.";

  options.set_parameter<TensorName<Scalar>>("cell_centers");
  options.set("cell_centers").doc() = "Cell center locations.";

  options.set_output("dumped_source") = VariableName(STATE, "dumped_source");
  options.set("dumped_source").doc() = "Source dumped into the smallest bin.";

  return options;
}

DumpInSmallestBin::DumpInSmallestBin(const OptionSet & options)
  : Model(options),
    _magnitude(declare_input_variable<Scalar>("magnitude")),
    _cell_centers(declare_parameter<Scalar>("cell_centers", "cell_centers", true)),
    _source(declare_output_variable<Scalar>("dumped_source"))
{
}

void
DumpInSmallestBin::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  using indexing::Slice;

  const auto N = _cell_centers.intmd_size(-1);
  const auto tail = _cell_centers.intmd_slice(-1, Slice(1, N));
  const auto zero_tail = Scalar::zeros_like(tail);
  const auto mag = (_magnitude.intmd_dim() == 0
                        ? Scalar(_magnitude().unsqueeze(0), /*batch_dim=*/0, /*intmd_dim=*/1)
                        : _magnitude().intmd_slice(-1, Slice(0, 1)));
  const auto source = intmd_cat({mag, zero_tail}, 0);

  if (out)
    _source = source;

  if (dout_din)
  {
    if (_magnitude.is_dependent())
    {
      const auto zero_tail_vec = Scalar::zeros_like(zero_tail);
      if (_magnitude.intmd_dim() == 0)
      {
        const auto selector = intmd_cat({Scalar::ones_like(mag), zero_tail_vec}, 0);
        _source.d(_magnitude, 1, 1, 0) = selector;
      }
      else
      {
        const auto mag_map = imap_v<Scalar>(mag.options()).intmd_expand(mag.intmd_size(-1));
        const auto diag_mag = intmd_diagonalize(mag_map);
        const auto zero_row = zero_tail_vec.intmd_unsqueeze(1);
        _source.d(_magnitude, 2, 1, 1) = intmd_cat({diag_mag, zero_row}, 0);
      }
    }
  }
}
} // namespace neml2
