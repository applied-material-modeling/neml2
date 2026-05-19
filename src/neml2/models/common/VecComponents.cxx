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

#include "neml2/models/common/VecComponents.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
register_NEML2_object(VecComponents);

OptionSet
VecComponents::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Decompose a Vec into its three Scalar components.";

  options.add_input("from", "The Vec to decompose");
  options.add<std::vector<VariableName>, FType::OUTPUT>(
      "to", "Names of the three Scalar component outputs (in order: 0, 1, 2).");

  return options;
}

VecComponents::VecComponents(const OptionSet & options)
  : Model(options),
    _from(declare_input_variable<Vec>("from"))
{
  const auto to_names = options.get<std::vector<VariableName>>("to");
  neml_assert(to_names.size() == 3,
              "VecComponents requires exactly 3 output names; got ",
              to_names.size(),
              ".");
  _to[0] = &declare_output_variable<Scalar>(to_names[0]);
  _to[1] = &declare_output_variable<Scalar>(to_names[1]);
  _to[2] = &declare_output_variable<Scalar>(to_names[2]);
}

void
VecComponents::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    *_to[0] = _from()(0);
    *_to[1] = _from()(1);
    *_to[2] = _from()(2);
  }

  if (dout_din)
  {
    const auto one = Scalar::ones_like(_from()(0));
    const auto zero = Scalar::zeros_like(_from()(0));
    _to[0]->d(_from) = Vec::fill(one, zero, zero);
    _to[1]->d(_from) = Vec::fill(zero, one, zero);
    _to[2]->d(_from) = Vec::fill(zero, zero, one);
  }
}
} // namespace neml2
