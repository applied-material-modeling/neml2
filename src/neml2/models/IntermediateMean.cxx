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

#include "neml2/models/IntermediateMean.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/mean.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
template <typename T>
OptionSet
IntermediateMean<T>::expected_options()
{
  OptionSet options = Reduction<T>::expected_options();
  options.doc() = "Average an intermediate dimension";
  return options;
}

template <typename T>
IntermediateMean<T>::IntermediateMean(const OptionSet & options)
  : Reduction<T>(options)
{
}

template <typename T>
void
IntermediateMean<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _to = intmd_mean(_from(), _dim);

  if (dout_din)
    if (_from.is_dependent())
    {
      const auto dim = utils::normalize_dim(_dim, 0, _from.intmd_dim());
      auto I = TensorShape(_from.intmd_sizes());
      I.erase(I.begin() + dim);
      const auto n = utils::numel(I);

      const auto deriv = imap_v<T>(_from.options()) / _from.intmd_size(_dim);
      _to.d(_from) = intmd_diagonalize(deriv.intmd_expand(n))
                         .intmd_reshape(utils::add_shapes(I, I))
                         .intmd_unsqueeze(0)
                         .intmd_expand(_from.intmd_size(dim), 0)
                         .intmd_movedim(0, dim + _from.intmd_dim() - 1);
    }
}

#define REGISTER_INTERMEDIATEMEAN(T)                                                               \
  using T##IntermediateMean = IntermediateMean<T>;                                                 \
  register_NEML2_object(T##IntermediateMean);                                                      \
  template class IntermediateMean<T>
FOR_ALL_PRIMITIVETENSOR(REGISTER_INTERMEDIATEMEAN);
} // namespace neml2
