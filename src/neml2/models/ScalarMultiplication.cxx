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

#include "neml2/models/ScalarMultiplication.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(ScalarMultiplication);
OptionSet
ScalarMultiplication::expected_options()
{

  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the multiplication (product) of multiple Scalar variable with a "
                  "constant coefficient. Using reciprocal, one can have the reciprocity of "
                  "variable 'a', aka. '1/a'";

  options.set<std::vector<VariableName>>("from_var");
  options.set("from_var").doc() = "Scalar variables to be multiplied";

  options.set_output("to_var");
  options.set("to_var").doc() = "The multiplicative product";

  options.set_parameter<TensorName<Scalar>>("coefficient") = {TensorName<Scalar>("1")};
  options.set("coefficient").doc() = "The coefficient multiply to the final product";

  options.set<std::vector<bool>>("reciprocal") = {false};
  options.set("reciprocal").doc() =
      "List of boolens, one for each variable, in which the reciprocity of the corresponding "
      "variable is taken. When the length of this list is 1, the same reciprocal condition applies "
      "to all variables.";

  options.set<bool>("define_second_derivatives") = true;

  return options;
}

ScalarMultiplication::ScalarMultiplication(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to_var")),
    _A(declare_parameter<Scalar>("A", "coefficient"))
{
  for (const auto & fv : options.get<std::vector<VariableName>>("from_var"))
    _from.push_back(&declare_input_variable<Scalar>(fv));

  _inv = options.get<std::vector<bool>>("reciprocal");
  neml_assert(_inv.size() == 1 || _inv.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " entries in reciprocal, got ",
              _inv.size(),
              ".");

  // Expand the list of booleans to match the number of coefficients
  if (_inv.size() == 1)
    _inv = std::vector<bool>(_from.size(), _inv[0]);
}

void
ScalarMultiplication::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    auto r = _inv[0] ? _A / (*_from[0]) : _A * (*_from[0]);
    for (std::size_t i = 1; i < _from.size(); i++)
    {
      if (_inv[i])
        r = r / (*_from[i]);
      else
        r = r * (*_from[i]);
    }
    _to = r;
  }

  if (dout_din)
  {
    for (std::size_t i = 0; i < _from.size(); i++)
      if (_from[i]->is_dependent())
      {
        auto r = _inv[i] ? -_A / (*_from[i]) / (*_from[i]) : _A;
        for (std::size_t j = 0; j < _from.size(); j++)
          if (i != j)
          {
            if (_inv[j])
              r = r / (*_from[j]);
            else
              r = r * (*_from[j]);
          }
        _to.d(*_from[i]) = r;
      }
  }

  if (d2out_din2)
  {
    for (std::size_t i = 0; i < _from.size(); i++)
      if (_from[i]->is_dependent())
      {
        auto p = (_inv[i] ? -1.0 : 1.0);
        for (std::size_t j = 0; j < _from.size(); j++)
        {
          if (_from[j]->is_dependent())
          {
            auto q = (_inv[j] ? -1.0 : 1.0);
            if (i != j)
            {
              auto r = _A * p * neml2::pow((*_from[i])(), (p - 1)) * q *
                       neml2::pow((*_from[j])(), (q - 1));
              for (std::size_t k = 0; k < _from.size(); k++)
                if (k != i && k != j)
                  r = r * (_inv[k] ? 1. / (*_from[k])() : (*_from[k])());
              _to.d2(*_from[i], *_from[j]) = r;
            }
            else if (_inv[i])
            {
              auto r = _A * p * (p - 1) * neml2::pow((*_from[i])(), (p - 2));
              for (std::size_t k = 0; k < _from.size(); k++)
                if (k != i)
                  r = r * (_inv[k] ? 1. / (*_from[k])() : (*_from[k])());
              _to.d2(*_from[i], *_from[j]) = r;
            }
          }
        }
      }
  }
}
} // namespace neml2
