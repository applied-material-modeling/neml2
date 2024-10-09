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

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/misc/math.h"

namespace neml2
{
LabeledVector
LabeledVector::slice(const std::string & name) const
{
  return LabeledVector(_tensor.base_index({_axes[0]->indices(name)}), {&_axes[0]->subaxis(name)});
}

void
LabeledVector::fill(const LabeledVector & other, bool recursive)
{
  const auto indices = axis(0).common_indices(other.axis(0), recursive);
  for (const auto & [idx, idx_other] : indices)
    _tensor.base_index_put_({idx}, other.tensor().base_index({idx_other}));
}

std::map<LabeledAxisAccessor, Tensor>
LabeledVector::split_variables(bool qualified) const
{
  std::map<LabeledAxisAccessor, Tensor> ret;

  const auto vars = qualified ? axis(0).qualified_variable_names() : axis(0).variable_names();
  const auto vals = tensor().split(axis(0).variable_sizes(), -1);
  for (std::size_t i = 0; i < vars.size(); ++i)
    ret[vars[i]] = Tensor(vals[i], batch_sizes());

  return ret;
}

std::map<LabeledAxisAccessor, LabeledVector>
LabeledVector::split_subaxes(bool qualified) const
{
  std::map<LabeledAxisAccessor, LabeledVector> ret;

  const auto keys = qualified ? axis(0).qualified_subaxis_names() : axis(0).subaxis_names();
  const auto subaxes = axis(0).subaxis_names();
  const auto vals = tensor().split(axis(0).subaxis_sizes(), -1);
  for (std::size_t i = 0; i < subaxes.size(); ++i)
    ret[keys[i]] = LabeledVector(Tensor(vals[i], batch_sizes()), {&axis(0).subaxis(subaxes[i])});

  return ret;
}

LabeledVector
LabeledVector::assemble(std::vector<Tensor> & vals, const LabeledAxis & axis)
{
  const auto batch_sizes = utils::broadcast_batch_sizes(vals);
  const auto options =
      torch::TensorOptions().dtype(utils::same_dtype(vals)).device(utils::same_device(vals));
  for (std::size_t i = 0; i < vals.size(); ++i)
    if (!vals[i].defined())
      vals[i] = Tensor::zeros(batch_sizes, axis.storage_size(i), options);
    else
      vals[i] = vals[i].batch_expand(batch_sizes);

  return LabeledVector(math::base_cat(vals, -1), {&axis});
}

namespace utils
{
bool
allclose(const LabeledVector & a, const LabeledVector & b, Real rtol, Real atol)
{
  if (a.axis(0) != b.axis(0))
    return false;

  for (auto var : a.axis(0).variable_names())
    if (!torch::allclose(a.base_index(var), b.base_index(var), rtol, atol))
      return false;

  return true;
}
}
} // namespace neml2
