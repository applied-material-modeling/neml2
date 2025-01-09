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

#include "neml2/models/solid_mechanics/crystal_plasticity/CrystalMean.h"
#include "neml2/models/crystallography/CrystalGeometry.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/mean.h"

namespace neml2
{
using SR2CrystalMean = CrystalMean<SR2>;
register_NEML2_object(SR2CrystalMean);

template <typename T>
OptionSet
CrystalMean<T>::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Average the variable over all crystals.";

  options.set_input("from");
  options.set("from").doc() = "Variable to average";

  options.set_output("to");
  options.set("to").doc() = "The averaged variable";

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";
  options.set("crystal_geometry_name").doc() =
      "The name of the Data object containing the crystallographic information";

  return options;
}

template <typename T>
CrystalMean<T>::CrystalMean(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    _from(declare_input_variable<SR2>("from", _crystal_geometry.nslip())),
    _to(declare_output_variable<T>("to"))
{
}

template <typename T>
void
CrystalMean<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _to = batch_mean(T(_from), -1);

  if (dout_din)
    if (_from.is_dependent())
    {
      const auto I = T::identity_map(_from.options())
                         .base_reshape({T::const_base_storage, T::const_base_storage})
                         .batch_expand_as(T(_from));
      _to.d(_from) = Tensor(I / _from.list_size(0), _from.batch_sizes()).base_transpose(0, 1);
    }
}
} // namespace neml2
