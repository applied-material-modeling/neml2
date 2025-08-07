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

#pragma once

#include "neml2/tensors/PrimitiveTensor.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
// Helper to concatenate two integer sequences
template <class Seq1, class Seq2>
struct concat_seq;
template <Size... A, Size... B>
struct concat_seq<std::integer_sequence<Size, A...>, std::integer_sequence<Size, B...>>
{
  using type = std::integer_sequence<Size, A..., B...>;
};
template <class Seq1, class Seq2>
using concat_seq_t = typename concat_seq<Seq1, Seq2>::type;

// Turn integer_sequence<Size, ...> into a PrimitiveTensor<Tensor, ...>
template <class Seq>
struct DTensorBase;
template <Size... S>
struct DTensorBase<std::integer_sequence<Size, S...>>
{
  using type = PrimitiveTensor<Tensor, S...>;
};
template <class Seq>
using DTensorBase_t = typename DTensorBase<Seq>::type;

/**
 * @brief Abstract representation of the derivative of a primitive tensor with respect to another
 * primitive tensor.
 */
template <class T1, class T2>
class DTensor
  : public DTensorBase_t<
        concat_seq_t<typename T1::base_sizes_sequence, typename T2::base_sizes_sequence>>
{
public:
  /// Special member functions
  DTensor() = default;

  /// Copy constructor
  template <class Derived2>
  DTensor(const TensorBase<Derived2> & tensor);

  /// Implicit conversion to a Tensor
  template <class U, typename = std::enable_if_t<std::is_base_of_v<TensorBase<U>, U>>>
  U as() const;
};

template <class T1, class T2>
template <class Derived2>
DTensor<T1, T2>::DTensor(const TensorBase<Derived2> & tensor)
  : DTensorBase_t<concat_seq_t<typename T1::base_sizes_sequence, typename T2::base_sizes_sequence>>(
        tensor)
{
}

template <class T1, class T2>
template <class U, typename>
U
DTensor<T1, T2>::as() const
{
  return U(*this, this->dynamic_sizes(), this->intmd_dim());
}
} // namespace neml2
