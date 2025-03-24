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

#include "neml2/models/ScalarVariableMultiplication.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(ScalarVariableMultiplication);
OptionSet
ScalarVariableMultiplication::expected_options()
{

  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the multiplication (product) of multiple Scalar variable with a "
      "constant coefficient. Using inverse_condition, one can have the variable 'a' to be '1/a'";

  options.set<std::vector<VariableName>>("from_var");
  options.set("from_var").doc() = "Scalar variables to be multiplied";

  options.set_output("to_var");
  options.set("to_var").doc() = "The multiplicative product";

  options.set_parameter<TensorName<Scalar>>("constant_coefficient") = {TensorName<Scalar>("1")};
  options.set("constant_coefficient").doc() =
      "The constant coefficient multiply to the final product";

  options.set<std::vector<bool>>("inverse_condition") = {false};
  options.set("inverse_condition").doc() = "Whether to inverse the corresponding variable";

  return options;
}

ScalarVariableMultiplication::ScalarVariableMultiplication(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to_var")),
    _A(declare_parameter<Scalar>("A", "constant_coefficient"))
{
  for (const auto & fv : options.get<std::vector<VariableName>>("from_var"))
    _from.push_back(&declare_input_variable<Scalar>(fv));

  _inv = options.get<std::vector<bool>>("inverse_condition");
  neml_assert(_inv.size() == 1 || _inv.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " entries in inverse_condition, got ",
              _inv.size(),
              ".");

  // Expand the list of booleans to match the number of coefficients
  if (_inv.size() == 1)
    _inv = std::vector<bool>(_from.size(), _inv[0]);
}

void
ScalarVariableMultiplication::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto value = _A * (*_from[0]);
  if (_inv[0])
    value = _A / (*_from[0]);
  for (std::size_t i = 1; i < _from.size(); i++)
  {
    if (_inv[i])
      value = value / (*_from[i]);
    else
      value = value * (*_from[i]);
  }
  if (out)
  {
    _to = value;
  }

  if (dout_din)
  {
    for (std::size_t i = 0; i < _from.size(); i++)
    {
      if (_from[i]->is_dependent())
      {
        if (_inv[i])
          _to.d(*_from[i]) = -1.0 * value / (*_from[i]);
        else
          _to.d(*_from[i]) = value / (*_from[i]);
      }
    }
  }

  if (d2out_din2)
  {
  }
}

} // namespace neml2
