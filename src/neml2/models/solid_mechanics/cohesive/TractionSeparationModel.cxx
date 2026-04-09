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

#include "neml2/models/solid_mechanics/cohesive/TractionSeparationModel.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
OptionSet
TractionSeparationModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Abstract base for cohesive zone traction-separation laws. "
                  "Derived classes map the interface displacement jump "
                  "\\f$ \\boldsymbol{\\delta} = [\\delta_n,\\, \\delta_{s1},\\, \\delta_{s2}] \\f$ "
                  "(normal + two tangential components in the local interface frame) "
                  "to the interface traction vector "
                  "\\f$ \\mathbf{T} = [T_n,\\, T_{s1},\\, T_{s2}] \\f$.";

  options.set_input("displacement_jump") = VariableName(FORCES, "displacement_jump");
  options.set("displacement_jump").doc() =
      "Interface displacement jump vector in the local interface frame, "
      "\\f$ \\boldsymbol{\\delta} = [\\delta_n,\\, \\delta_{s1},\\, \\delta_{s2}] \\f$, "
      "where \\f$ \\delta_n \\f$ is the normal (opening) component and "
      "\\f$ \\delta_{s1}, \\delta_{s2} \\f$ are the two tangential (sliding) components. "
      "Positive \\f$ \\delta_n \\f$ indicates interface opening. "
      "Default axis path: forces/displacement_jump.";

  options.set_output("traction") = VariableName(STATE, "traction");
  options.set("traction").doc() =
      "Interface traction vector in the local interface frame, "
      "\\f$ \\mathbf{T} = [T_n,\\, T_{s1},\\, T_{s2}] \\f$, "
      "where \\f$ T_n \\f$ is the normal traction and "
      "\\f$ T_{s1}, T_{s2} \\f$ are the tangential traction components. "
      "Default axis path: state/traction.";

  return options;
}

TractionSeparationModel::TractionSeparationModel(const OptionSet & options)
  : Model(options),
    _delta(declare_input_variable<Vec>("displacement_jump")),
    _traction(declare_output_variable<Vec>("traction"))
{
}
} // namespace neml2
