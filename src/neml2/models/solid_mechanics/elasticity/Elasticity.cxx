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

#include "neml2/models/solid_mechanics/elasticity/Elasticity.h"

namespace neml2
{
OptionSet
Elasticity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Relate elastic strain to stress";

  options.set_input("strain") = VariableName(STATE, "internal", "Ee");
  options.set("strain").doc() = "Elastic strain";

  options.set_output("stress") = VariableName(STATE, "S");
  options.set("stress").doc() = "Stress";

  options.set<bool>("compliance") = false;
  options.set("compliance").doc() =
      "Whether the model defines the compliance relationship, i.e., mapping from stress to elastic "
      "strain. When set to false (default), the model maps elastic strain to stress.";

  options.set<bool>("rate_form") = false;
  options.set("rate_form").doc() =
      "Whether the model defines the stress-strain relationship in rate form. When set to true, "
      "the model maps elastic strain *rate* to stress *rate*.";

  return options;
}

Elasticity::Elasticity(const OptionSet & options)
  : Model(options),
    _compliance(options.get<bool>("compliance")),
    _rate_form(options.get<bool>("rate_form")),
    _strain(options.get<VariableName>("strain").with_suffix(_rate_form ? "_rate" : "")),
    _stress(options.get<VariableName>("stress").with_suffix(_rate_form ? "_rate" : "")),
    _from(declare_input_variable<SR2>(_compliance ? _stress : _strain)),
    _to(declare_output_variable<SR2>(_compliance ? _strain : _stress))
{
}
} // namespace neml2
