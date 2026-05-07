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

#include "neml2/models/finite_volume/SmearedDeltaSource.h"

#include "neml2/misc/types.h"
#include <cmath>

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(SmearedDeltaSource);

OptionSet
SmearedDeltaSource::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute a smeared Gaussian source term for a Dirac delta distribution.";

  options.add_input("magnitude", "Source magnitude.");

  options.add_input("location", "Source location.");

  options.add_parameter<Scalar>("width", "Gaussian width.");

  options.add_parameter<Scalar>("cell_centers", "Cell center locations.");

  options.add_output("smeared_source", "state/smeared_source", "Smeared Gaussian source.");

  return options;
}

SmearedDeltaSource::SmearedDeltaSource(const OptionSet & options)
  : Model(options),
    _magnitude(declare_input_variable<Scalar>("magnitude")),
    _location(declare_input_variable<Scalar>("location")),
    _width(declare_parameter<Scalar>("width", "width", /*allow_nonlinear=*/true)),
    _cell_centers(declare_parameter<Scalar>("cell_centers", "cell_centers", true)),
    _source(declare_output_variable<Scalar>("smeared_source"))
{
}

void
SmearedDeltaSource::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto centers = _cell_centers;
  const auto magnitude = _magnitude();
  const auto location = _location();
  const auto width = _width;

  const auto inv_width = 1.0 / width;
  const auto arg = (centers - location) * inv_width;
  const auto exp_term = exp(-0.5 * arg * arg);
  const auto prefactor = inv_width / std::sqrt(2.0 * neml2::pi);
  const auto gauss = prefactor * exp_term;
  const auto source = magnitude * gauss;

  if (out)
    _source = source;

  if (dout_din)
  {
    {
      if (_magnitude.intmd_dim() == 0)
        _source.d(_magnitude, 1, 1, 0) = gauss;
      else
      {
        const auto nbin = _magnitude.intmd_size(-1);
        const auto mag_map = imap_v<Scalar>(_magnitude.options()).intmd_expand(nbin);
        const auto diag_mag = intmd_diagonalize(mag_map);
        _source.d(_magnitude, 2, 1, 1) = gauss.intmd_unsqueeze(1) * diag_mag;
      }
    }

    {
      const auto dsource_dloc = magnitude * gauss * arg * inv_width;
      if (_location.intmd_dim() == 0)
        _source.d(_location, 1, 1, 0) = dsource_dloc;
      else
      {
        const auto nbin = _location.intmd_size(-1);
        const auto loc_map = imap_v<Scalar>(_location.options()).intmd_expand(nbin);
        const auto diag_loc = intmd_diagonalize(loc_map);
        _source.d(_location, 2, 1, 1) = dsource_dloc.intmd_unsqueeze(1) * diag_loc;
      }
    }

    if (const auto * const width_param = nl_param("width"))
    {
      const auto arg2 = arg * arg;
      const auto dsource_dwidth = magnitude * gauss * (arg2 - 1.0) * inv_width;
      if (width_param->intmd_dim() == 0)
        _source.d(*width_param, 1, 1, 0) = dsource_dwidth;
      else
      {
        const auto nbin = width_param->intmd_size(-1);
        const auto w_map = imap_v<Scalar>(_width.options()).intmd_expand(nbin);
        const auto diag_w = intmd_diagonalize(w_map);
        _source.d(*width_param, 2, 1, 1) = dsource_dwidth.intmd_unsqueeze(1) * diag_w;
      }
    }

    if (const auto * const centers_param = nl_param("cell_centers"))
    {
      const auto dsource_dcenters = -magnitude * gauss * arg * inv_width;
      if (centers_param->intmd_dim() == 0)
        _source.d(*centers_param, 1, 1, 0) = dsource_dcenters;
      else
      {
        const auto nbin = _cell_centers.intmd_size(-1);
        const auto x_map = imap_v<Scalar>(_cell_centers.options()).intmd_expand(nbin);
        const auto diag_x = intmd_diagonalize(x_map);
        _source.d(*centers_param, 2, 1, 1) = dsource_dcenters.intmd_unsqueeze(1) * diag_x;
      }
    }
  }
}
} // namespace neml2
