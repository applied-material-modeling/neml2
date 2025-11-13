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

namespace neml2
{
class Model;
}

/// Get an already registered pybind11 class from a module
template <typename T>
pybind11::class_<T>
get_pycls(const pybind11::module_ & m, const char * name)
{
  auto pyc = m.attr(name);
  return pybind11::class_<T>(pyc);
}

/// Get an already registered pybind11 class given the module name
template <typename T>
pybind11::class_<T>
get_pycls(const char * mname, const char * name)
{
  auto m = pybind11::module_::import(mname);
  return get_pycls<T>(m, name);
}

/// Unpack a Python dictionary into a neml2::ValueMap
neml2::ValueMap unpack_tensor_map(const pybind11::dict & pyinputs,
                                  const neml2::Model * model = nullptr);
