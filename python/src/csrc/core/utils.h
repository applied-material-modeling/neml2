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

#include "neml2/models/map_types_fwd.h"

#include <pybind11/pybind11.h>

#include "csrc/core/types.h"
#include "csrc/tensors/types.h"

namespace neml2
{
class Model;
}

/// Unpack a Python sequence into a std::vector<neml2::Tensor>
std::vector<neml2::Tensor>
unpack_tensor_list(const pybind11::sequence & pytensors,
                   const std::function<neml2::TensorShapeRef(std::size_t)> & base_shape_fn);

/// Unpack a Python dictionary into a neml2::ValueMap
neml2::ValueMap unpack_value_map(
    const pybind11::dict & pyvals,
    bool assembly,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn);

/// Unpack a Python dictionary of dictionaries into a neml2::DerivMap
neml2::DerivMap unpack_deriv_map(
    const pybind11::dict & pyderivs,
    bool assembly,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn_i,
    const std::function<neml2::TensorShapeRef(const neml2::VariableName &)> & base_shape_fn_j);

/// Pack a neml2::ValueMap into a std::map<std::string, neml2::Tensor>
std::map<std::string, neml2::Tensor> pack_value_map(const neml2::ValueMap & vals);

/// Pack a neml2::DerivMap into a std::map<std::string, std::map<std::string, neml2::Tensor>>
std::map<std::string, std::map<std::string, neml2::Tensor>>
pack_deriv_map(const neml2::DerivMap & derivs);

/// Unpack a Python object into a neml2::VariableName (i.e., either a VariableName or a string)
neml2::VariableName unpack_variable_name(const pybind11::handle & obj);
