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

#include "neml2/models/solid_mechanics/crystal_plasticity/ResolvedShear.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SFFR4.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/diagonalize.h"

namespace neml2
{
register_NEML2_object(ResolvedShear);

OptionSet
ResolvedShear::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Calculates the resolved shears as \\f$ \\tau_i = \\sigma : Q "
                  "\\operatorname{sym}\\left(d_i \\otimes n_i \\right) Q^T \\f$ where \\f$ \\tau_i "
                  "\\f$ is the resolved shear on slip system i, \\f$ \\sigma \\f$ is the Cauchy "
                  "stress \\f$ Q \\f$ is the orientation matrix, \\f$ d_i \\f$ is the slip "
                  "direction, and \\f$ n_i \\f$ is the slip system normal.";

  options.set_output("resolved_shears") = VariableName(STATE, "internal", "resolved_shears");
  options.set("resolved_shears").doc() = "The name of the resolved shears";

  options.set_input("stress") = VariableName(STATE, "internal", "cauchy_stress");
  options.set("stress").doc() = "The name of the Cauchy stress tensor";

  options.set_input("orientation") = VariableName(STATE, "orientation_matrix");
  options.set("orientation").doc() = "The name of the orientation matrix";

  options.set<std::string>("crystal_geometry") = "crystal_geometry";
  options.set("crystal_geometry").doc() =
      "The name of the data object with the crystallographic information";
  return options;
}

ResolvedShear::ResolvedShear(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry"))),
    _rss(declare_output_variable<Scalar>("resolved_shears")),
    _S(declare_input_variable<SR2>("stress")),
    _R(declare_input_variable<R2>("orientation"))
{
}

void
ResolvedShear::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Schmid tensor
  const auto & M = _crystal_geometry.M();
  // unsqueeze R and S to broadcast over slip systems
  const auto R = _R().intmd_unsqueeze(-1);
  const auto S = _S().intmd_unsqueeze(-1);

  if (out)
    _rss = neml2::inner(M.rotate(R), S);

  if (dout_din)
  {
    if (_S.is_dependent())
      _rss.d(_S, 1, 1, 0) = M.rotate(R);

    if (_R.is_dependent())
      _rss.d(_R, 1, 1, 0) = R2::einsum("...ijk,...i->...jk", {M.drotate(R), S});
  }
}

} // namespace neml2
