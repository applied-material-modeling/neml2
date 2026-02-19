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

#include "neml2/models/kwn/CurrentConcentration.h"

#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(CurrentConcentration);

OptionSet
CurrentConcentration::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the current matrix concentration from precipitate volume fractions.";

  options.set_parameter<TensorName<Scalar>>("initial_concentration");
  options.set("initial_concentration").doc() = "Initial concentration in solution";

  options.set<std::vector<VariableName>>("precipitate_volume_fractions");
  options.set("precipitate_volume_fractions").doc() =
      "Precipitate volume fraction variables";

  options.set_parameter<std::vector<TensorName<Scalar>>>("precipitate_concentrations");
  options.set("precipitate_concentrations").doc() = "Precipitate concentrations";

  options.set_output("current_concentration");
  options.set("current_concentration").doc() = "Current concentration in solution";

  return options;
}

CurrentConcentration::CurrentConcentration(const OptionSet & options)
  : Model(options),
    _x0(declare_parameter<Scalar>("x0", "initial_concentration", true)),
    _x(declare_output_variable<Scalar>("current_concentration"))
{
  for (const auto & v : options.get<std::vector<VariableName>>("precipitate_volume_fractions"))
    _fs.push_back(&declare_input_variable<Scalar>(v));

  const auto xp_refs = options.get<std::vector<TensorName<Scalar>>>("precipitate_concentrations");
  neml_assert(xp_refs.size() == _fs.size(),
              "Number of precipitate concentrations (",
              xp_refs.size(),
              ") does not match number of precipitate volume fractions (",
              _fs.size(),
              ").");

  _xps.resize(xp_refs.size());
  for (std::size_t i = 0; i < xp_refs.size(); i++)
    _xps[i] = &declare_parameter<Scalar>(
        "xp_" + std::to_string(i), xp_refs[i], /*allow_nonlinear=*/true);
}

void
CurrentConcentration::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto sum_f = Scalar::zeros_like((*_fs[0])());
  auto sum_fx = Scalar::zeros_like((*_fs[0])());

  for (std::size_t i = 0; i < _fs.size(); i++)
  {
    sum_f = sum_f + (*_fs[i])();
    sum_fx = sum_fx + (*_fs[i])() * (*_xps[i]);
  }

  const auto denom = 1.0 - sum_f;
  const auto numer = _x0 - sum_fx;

  if (out)
    _x = numer / denom;

  if (dout_din)
  {
    for (std::size_t i = 0; i < _fs.size(); i++)
      if (_fs[i]->is_dependent())
        _x.d(*_fs[i]) = (numer - denom * (*_xps[i])) / (denom * denom);

    if (const auto * const x0 = nl_param("x0"))
      _x.d(*x0) = 1.0 / denom;

    for (std::size_t i = 0; i < _xps.size(); i++)
    {
      if (const auto * const xp = nl_param("xp_" + std::to_string(i)))
        _x.d(*xp) = -(*_fs[i])() / denom;
    }
  }
}
} // namespace neml2
