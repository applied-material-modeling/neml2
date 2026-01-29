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

#include "neml2/models/IntermediateDiff.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/diff.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{
template <typename T>
OptionSet
IntermediateDiff<T>::expected_options()
{
  OptionSet options = Reduction<T>::expected_options();
  options.doc() = "Finite difference along an intermediate dimension";

  options.set<Size>("dim");
  options.set("dim").doc() = "Intermediate dimension to take the finite difference";

  options.set<Size>("n") = 1;
  options.set("n").doc() = "Order of the finite difference";

  return options;
}

template <typename T>
IntermediateDiff<T>::IntermediateDiff(const OptionSet & options)
  : Reduction<T>(options),
    _from(this->template declare_input_variable<T>("from")),
    _dim(options.get<Size>("dim")),
    _n(options.get<Size>("n"))
{
}

template <typename T>
void
IntermediateDiff<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _to = intmd_diff(_from(), _n, _dim);

  if (dout_din)
    if (_from.is_dependent())
    {
      const auto dim = utils::normalize_dim(_dim, 0, _from.intmd_dim());
      const auto I = imap_v<T>(_from.options()).intmd_expand(_from.intmd_sizes());
      const auto Id = intmd_diagonalize(I, dim);
      _to.d(_from) = intmd_diff(Id, _n, dim);
    }
}

#define REGISTER_INTERMEDIATEDIFF(T)                                                               \
  using T##IntermediateDiff = IntermediateDiff<T>;                                                 \
  register_NEML2_object(T##IntermediateDiff);                                                      \
  template class IntermediateDiff<T>
FOR_ALL_PRIMITIVETENSOR(REGISTER_INTERMEDIATEDIFF);
} // namespace neml2
