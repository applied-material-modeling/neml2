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

#include "neml2/models/kwn/ProjectedDiffusivitySum.h"

#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(ProjectedDiffusivitySum);

OptionSet
ProjectedDiffusivitySum::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the projected diffusivity sum for SFFK and nucleation.";

  options.set_parameter<std::vector<TensorName<Scalar>>>("concentration_differences");
  options.set("concentration_differences").doc() = "Concentration differences for each species";

  options.set_parameter<std::vector<TensorName<Scalar>>>("diffusivities");
  options.set("diffusivities").doc() = "Species diffusivities";

  options.set<std::vector<VariableName>>("far_field_concentrations");
  options.set("far_field_concentrations").doc() = "Far-field concentrations";

  options.set_output("projected_diffusivity_sum");
  options.set("projected_diffusivity_sum").doc() = "Projected diffusivity sum";

  return options;
}

ProjectedDiffusivitySum::ProjectedDiffusivitySum(const OptionSet & options)
  : Model(options),
    _sum(declare_output_variable<Scalar>("projected_diffusivity_sum"))
{
  const auto dx_refs = options.get<std::vector<TensorName<Scalar>>>("concentration_differences");
  _dxs.resize(dx_refs.size());
  for (std::size_t i = 0; i < dx_refs.size(); i++)
    _dxs[i] =
        &declare_parameter<Scalar>("dx_" + std::to_string(i), dx_refs[i], /*allow_nonlinear=*/true);

  for (const auto & v : options.get<std::vector<VariableName>>("far_field_concentrations"))
    _x_infs.push_back(&declare_input_variable<Scalar>(v));

  const auto d_refs = options.get<std::vector<TensorName<Scalar>>>("diffusivities");

  neml_assert(d_refs.size() == _dxs.size(),
              "Number of diffusivities (",
              d_refs.size(),
              ") does not match number of concentration differences (",
              _dxs.size(),
              ").");
  neml_assert(_x_infs.size() == _dxs.size(),
              "Number of far-field concentrations (",
              _x_infs.size(),
              ") does not match number of concentration differences (",
              _dxs.size(),
              ").");

  _Ds.resize(d_refs.size());
  for (std::size_t i = 0; i < d_refs.size(); i++)
    _Ds[i] =
        &declare_parameter<Scalar>("D_" + std::to_string(i), d_refs[i], /*allow_nonlinear=*/true);
}

void
ProjectedDiffusivitySum::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto sum = Scalar::zeros_like(*_dxs[0]);

  for (std::size_t i = 0; i < _dxs.size(); i++)
    sum = sum + pow(*_dxs[i], 2.0) / (*_Ds[i] * (*_x_infs[i])());

  if (out)
    _sum = sum;

  if (dout_din)
  {
    for (std::size_t i = 0; i < _dxs.size(); i++)
    {
      const auto dx = (*_dxs[i]);
      const auto xinf = (*_x_infs[i])();
      const auto D = (*_Ds[i]);

      if (const auto * const dx_param = nl_param("dx_" + std::to_string(i)))
        _sum.d(*dx_param) = 2.0 * dx / (D * xinf);

      if (_x_infs[i]->is_dependent())
        _sum.d(*_x_infs[i]) = -pow(dx, 2.0) / (D * xinf * xinf);

      if (const auto * const D_param = nl_param("D_" + std::to_string(i)))
        _sum.d(*D_param) = -pow(dx, 2.0) / (D * D * xinf);
    }
  }
}
} // namespace neml2
