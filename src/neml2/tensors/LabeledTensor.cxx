// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/tensors/LabeledTensor.h"

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
template <class Derived, Size D>
LabeledTensor<Derived, D>::LabeledTensor(const torch::Tensor & tensor,
                                         const std::array<const LabeledAxis *, D> & axes)
  : _tensor(tensor, tensor.dim() - D),
    _axes(axes)
{
  // Check that the size of the tensor was compatible
  neml_assert_dbg(base_sizes() == storage_size(), "LabeledTensor does not have the right size");
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::LabeledTensor(const Tensor & tensor,
                                         const std::array<const LabeledAxis *, D> & axes)
  : _tensor(tensor),
    _axes(axes)
{
  // Check that the size of the tensor was compatible
  neml_assert_dbg(base_sizes() == storage_size(), "LabeledTensor does not have the right size");
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::LabeledTensor(const Derived & other)
  : _tensor(other),
    _axes(other.axes())
{
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::operator=(const Derived & other)
{
  _tensor = other.tensor();
  _axes = other.axes();
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::operator Tensor() const
{
  return _tensor;
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::operator torch::Tensor() const
{
  return _tensor;
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::empty(TensorShapeRef batch_size,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TensorShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(Tensor::empty(batch_size, s, options), axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::empty_like(const Derived & other)
{
  return Derived(Tensor::empty_like(other), other.axes());
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::zeros(TensorShapeRef batch_size,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TensorShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(Tensor::zeros(batch_size, s, options), axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::zeros_like(const Derived & other)
{
  return Derived(Tensor::zeros_like(other), other.axes());
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::clone(torch::MemoryFormat memory_format) const
{
  return Derived(_tensor.clone(memory_format), _axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::detach() const
{
  return Derived(_tensor.detach(), _axes);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::detach_()
{
  _tensor.detach_();
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::zero_()
{
  _tensor.zero_();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::batch_dim() const
{
  return _tensor.batch_dim();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::base_dim() const
{
  return D;
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::batch_sizes() const
{
  return _tensor.batch_sizes();
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::base_sizes() const
{
  return _tensor.base_sizes();
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::storage_size() const
{
  return base_sizes();
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::slice(Size i, const std::string & name) const
{
  indexing::TensorIndices idx(base_dim(), torch::indexing::Slice());
  idx[i] = _axes[i]->indices(name);

  auto new_axes = _axes;
  new_axes[i] = &_axes[i]->subaxis(name);

  return Derived(_tensor.base_index(idx), new_axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::batch_index(indexing::TensorIndices indices) const
{
  return Derived(_tensor.batch_index(indices), _axes);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::batch_index_put(indexing::TensorIndices indices,
                                           const torch::Tensor & other)
{
  _tensor.batch_index_put(indices, other);
}

template <class Derived, Size D>
Tensor
LabeledTensor<Derived, D>::base_index(indexing::TensorIndices indices) const
{
  return _tensor.base_index(indices);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::base_index_put(indexing::TensorIndices indices,
                                          const torch::Tensor & other)
{
  _tensor.base_index_put(indices, other);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::operator-() const
{
  return Derived(-_tensor, _axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::to(const torch::TensorOptions & options) const
{
  return Derived(_tensor.to(options), _axes);
}

template class LabeledTensor<LabeledVector, 1>;
template class LabeledTensor<LabeledMatrix, 2>;
template class LabeledTensor<LabeledTensor3D, 3>;
}
