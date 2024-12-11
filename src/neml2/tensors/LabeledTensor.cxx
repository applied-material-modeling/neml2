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
  for (Size i = 0; i < D; i++)
    neml_assert_dbg(base_size(i) == _axes[i]->storage_size(),
                    "LabeledTensor does not have the right size at dimension ",
                    i,
                    ". Expected ",
                    _axes[i]->storage_size(),
                    ", got ",
                    base_size(i));
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::LabeledTensor(const Tensor & tensor,
                                         const std::array<const LabeledAxis *, D> & axes)
  : _tensor(tensor),
    _axes(axes)
{
  // Check that the size of the tensor was compatible
  for (Size i = 0; i < D; i++)
    neml_assert_dbg(base_size(i) == _axes[i]->storage_size(),
                    "LabeledTensor does not have the right size at dimension ",
                    i,
                    ". Expected ",
                    _axes[i]->storage_size(),
                    ", got ",
                    base_size(i));
}

template <class Derived, Size D>
LabeledTensor<Derived, D>::LabeledTensor(const Derived & other)
  : _tensor(other),
    _axes(other.axes())
{
}

template <class Derived, Size D>
LabeledTensor<Derived, D> &
LabeledTensor<Derived, D>::operator=(const Derived & other)
{
  _tensor = other.tensor();
  _axes = other.axes();
  return *this;
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
LabeledTensor<Derived, D>::empty(TensorShapeRef batch_shape,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TensorShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(Tensor::empty(batch_shape, s, options), axes);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::zeros(TensorShapeRef batch_shape,
                                 const std::array<const LabeledAxis *, D> & axes,
                                 const torch::TensorOptions & options)
{
  TensorShape s;
  s.reserve(axes.size());
  std::transform(axes.begin(),
                 axes.end(),
                 std::back_inserter(s),
                 [](const LabeledAxis * axis) { return axis->storage_size(); });
  return Derived(Tensor::zeros(batch_shape, s, options), axes);
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
Derived
LabeledTensor<Derived, D>::to(const torch::TensorOptions & options) const
{
  return Derived(_tensor.to(options), _axes);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::copy_(const torch::Tensor & other)
{
  _tensor.copy_(other);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::zero_()
{
  _tensor.zero_();
}

template <class Derived, Size D>
bool
LabeledTensor<Derived, D>::requires_grad() const
{
  return _tensor.requires_grad();
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::requires_grad_(bool req)
{
  _tensor.requires_grad_(req);
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::operator-() const
{
  return Derived(-_tensor, _axes);
}

template <class Derived, Size D>
torch::TensorOptions
LabeledTensor<Derived, D>::options() const
{
  return _tensor.options();
}

template <class Derived, Size D>
torch::Dtype
LabeledTensor<Derived, D>::scalar_type() const
{
  return _tensor.scalar_type();
}

template <class Derived, Size D>
torch::Device
LabeledTensor<Derived, D>::device() const
{
  return _tensor.device();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::dim() const
{
  return _tensor.dim();
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::sizes() const
{
  return _tensor.sizes();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::size(Size d) const
{
  return _tensor.size(d);
}

template <class Derived, Size D>
bool
LabeledTensor<Derived, D>::batched() const
{
  return _tensor.batched();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::batch_dim() const
{
  return _tensor.batch_dim();
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::batch_sizes() const
{
  return _tensor.batch_sizes();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::batch_size(Size d) const
{
  return _tensor.batch_size(d);
}

template <class Derived, Size D>
TensorShapeRef
LabeledTensor<Derived, D>::base_sizes() const
{
  return _tensor.base_sizes();
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::base_size(Size d) const
{
  return _tensor.base_size(d);
}

template <class Derived, Size D>
Size
LabeledTensor<Derived, D>::base_storage() const
{
  return _tensor.base_storage();
}

template <class Derived, Size D>
Derived
LabeledTensor<Derived, D>::batch_index(indexing::TensorIndicesRef indices) const
{
  return Derived(_tensor.batch_index(indices), _axes);
}

template <class Derived, Size D>
Tensor
LabeledTensor<Derived, D>::base_index(indexing::TensorLabelsRef labels) const
{
  return _tensor.base_index(labels_to_indices(labels));
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::batch_index_put_(indexing::TensorIndicesRef indices,
                                            const torch::Tensor & other)
{
  _tensor.batch_index_put_(indices, other);
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::base_index_put_(indexing::TensorLabelsRef labels, const Tensor & other)
{
  _tensor.base_index_put_(
      labels_to_indices(labels),
      other.reshape(utils::add_shapes(other.batch_sizes(), storage_shape(labels))));
}

template <class Derived, Size D>
TensorShape
LabeledTensor<Derived, D>::storage_shape(indexing::TensorLabelsRef labels) const
{
  TensorShape s(labels.size());
  for (size_t i = 0; i < labels.size(); i++)
    s[i] = _axes[i]->storage_size(labels[i]);
  return s;
}

template <class Derived, Size D>
indexing::TensorIndices
LabeledTensor<Derived, D>::labels_to_indices(indexing::TensorLabelsRef labels) const
{
  neml_assert_dbg(labels.size() == D, "Wrong label size, must be ", D, ", got ", labels.size());
  indexing::TensorIndices indices;
  for (size_t i = 0; i < labels.size(); i++)
    indices.push_back(_axes[i]->indices(labels[i]));
  return indices;
}

template <class Derived, Size D>
void
LabeledTensor<Derived, D>::fill(const Derived & other, Size odim, bool recursive)
{
  neml_assert_dbg(odim < D, "Common axis for fill must be less than the tensor dimension");
  for (Size i = 0; i < Size(axes().size()) && i != odim; i++)
    neml_assert_dbg(axis(i) == other.axis(i),
                    "Can only accumulate matrices with conformal common axes");
  const auto cindices = axis(odim).common_indices(other.axis(odim), recursive);
  for (const auto & [idx, idx_other] : cindices)
  {
    indexing::TensorIndices indices(base_dim(), indexing::Slice());
    indexing::TensorIndices indices_other(base_dim(), indexing::Slice());
    indices[odim] = idx;
    indices_other[odim] = idx_other;
    _tensor.base_index_put_(indices, other.tensor().base_index(indices_other));
  }
}

template class LabeledTensor<LabeledVector, 1>;
template class LabeledTensor<LabeledMatrix, 2>;
template class LabeledTensor<LabeledTensor3D, 3>;
}
