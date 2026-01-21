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

#include "neml2/models/ModelNonlinearSystem.h"
#include "neml2/neml2.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/base/LabeledAxis.h"
#include "neml2/tensors/TensorValue.h"
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "python/neml2/macros.h"

// Forward declarations
void def(pybind11::module_ &, pybind11::class_<neml2::LabeledAxisAccessor> &);
void def(pybind11::module_ &, pybind11::class_<neml2::LabeledAxis> &);
void def(pybind11::module_ &, pybind11::class_<neml2::TensorValueBase> &);
void def(pybind11::module_ &, pybind11::class_<neml2::Factory> &);
void def(pybind11::module_ &, pybind11::class_<neml2::Model, std::shared_ptr<neml2::Model>> &);
void def(pybind11::module_ &, pybind11::class_<neml2::HVector> &);
void def(pybind11::module_ &, pybind11::class_<neml2::HMatrix> &);
void def(pybind11::module_ &, pybind11::class_<neml2::ModelNonlinearSystem> &);

// Type casters are only for cross-module types used in function signatures
DEFAULT_TYPECASTER(neml2::LabeledAxisAccessor, "Union[neml2.core.VariableName, str]");
DEFAULT_TYPECASTER(neml2::LabeledAxis, "neml2.core.LabeledAxis");
DEFAULT_TYPECASTER(neml2::TensorValueBase, "neml2.core.TensorValue");
DEFAULT_TYPECASTER(neml2::Factory, "neml2.core.Factory");
DEFAULT_TYPECASTER(neml2::Model, "neml2.core.Model");
DEFAULT_TYPECASTER_SHARED_PTR(neml2::Model, "neml2.core.Model");
DEFAULT_TYPECASTER(neml2::HVector, "neml2.core.HVector");
DEFAULT_TYPECASTER(neml2::HMatrix, "neml2.core.HMatrix");
DEFAULT_TYPECASTER(neml2::ModelNonlinearSystem, "neml2.core.ModelNonlinearSystem");

namespace pybind11::detail
{
/**
 * @brief This specialization exposes neml2::TensorShape
 */
template <>
struct type_caster<neml2::TensorShape>
{
public:
  PYBIND11_TYPE_CASTER(neml2::TensorShape, const_name("tuple[int, ...]"));

  bool load(handle src, bool)
  {
    if (!src)
      return false;

    // Accept any Python sequence (tuple/list/torch.Size behaves like a sequence)
    if (!isinstance<sequence>(src))
      return false;

    sequence seq = reinterpret_borrow<sequence>(src);
    value.clear();
    value.reserve(len(seq));
    for (handle item : seq)
      value.push_back(item.cast<neml2::Size>());

    return true;
  }

  static handle
  cast(const neml2::TensorShape & src, return_value_policy /*policy*/, handle /*parent*/)
  {
    tuple t(src.size());
    for (size_t i = 0; i < src.size(); ++i)
      t[i] = pybind11::int_(src[i]);
    return t.release();
  }
};
} // namespace pybind11::detail
