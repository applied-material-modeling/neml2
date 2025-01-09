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

#include "neml2/user_tensors/SymmetryFromOrbifold.h"
#include "neml2/tensors/crystallography.h"

namespace neml2
{
register_NEML2_object(SymmetryFromOrbifold);

OptionSet
SymmetryFromOrbifold::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Returns a tensor of symmetry operations for a given symmetr group represented "
                  "in orbifold notation.";

  options.set<std::string>("orbifold");
  options.set("orbifold").doc() =
      "A string giving the orbifold representation of the group, for example 432 for the typical "
      "cubic crystal system defined by chiral octahedral symmetry";
  return options;
}

SymmetryFromOrbifold::SymmetryFromOrbifold(const OptionSet & options)
  : R2(crystallography::symmetry_operations_from_orbifold(options.get<std::string>("orbifold"))),
    UserTensorBase(options)
{
}
} // namespace neml2
