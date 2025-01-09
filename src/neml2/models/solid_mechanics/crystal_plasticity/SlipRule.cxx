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

#include "neml2/models/solid_mechanics/crystal_plasticity/SlipRule.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/list_tensors.h"

namespace neml2
{
OptionSet
SlipRule::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Parent class for all slip rules, which define the slip rate in terms of the "
                  "resolved shear and the slip system strength";

  options.set_output("slip_rates") = VariableName(STATE, "internal", "slip_rates");
  options.set("slip_rates").doc() = "Name of the slip rate tensor";

  options.set_input("resolved_shears") = VariableName(STATE, "internal", "resolved_shears");
  options.set("resolved_shears").doc() = "Name of the resolved shear tensor";

  options.set_input("slip_strengths") = VariableName(STATE, "internal", "slip_strengths");
  options.set("slip_strengths").doc() = "Name of the tensor containing the slip system strengths";

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";
  options.set("crystal_geometry_name").doc() =
      "Name of the Data object containing the crystallographic information";

  return options;
}

SlipRule::SlipRule(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    _g(declare_output_variable<Scalar>("slip_rates", _crystal_geometry.nslip())),
    _rss(declare_input_variable<Scalar>("resolved_shears", _crystal_geometry.nslip())),
    _tau(declare_input_variable<Scalar>("slip_strengths", _crystal_geometry.nslip()))
{
}
} // namespace neml2
