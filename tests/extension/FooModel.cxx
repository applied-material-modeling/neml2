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

#include "FooModel.h"
#include "neml2/tensors/functions/imap.h"

namespace foo
{
// Since the neml2 registration macro directly registers to "Registry",
// we need to first bring it into the same scope.
using Registry = neml2::Registry;
register_NEML2_object(FooModel);

neml2::OptionSet
FooModel::expected_options()
{
  auto options = neml2::Model::expected_options();
  options.set_input("x");
  options.set_output("y");
  options.set<double>("c");
  return options;
}

FooModel::FooModel(const neml2::OptionSet & options)
  : neml2::Model(options),
    _x(declare_input_variable<neml2::Scalar>("x")),
    _y(declare_output_variable<neml2::Scalar>("y")),
    _c(options.get<double>("c"))
{
}

void
FooModel::set_value(bool out, bool dout, bool)
{
  if (out)
    _y = _c * _x + _c - 1.0;

  if (dout)
    _y.d(_x) = _c * neml2::imap_v<neml2::Scalar>(_x.options());
}
}
