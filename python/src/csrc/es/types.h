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
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "macros.h"
#include "csrc/core/types.h"

// Forward declarations
void
def(pybind11::module_ &,
    pybind11::class_<neml2::ModelNonlinearSystem, std::shared_ptr<neml2::ModelNonlinearSystem>> &);
void def(pybind11::module_ &, pybind11::class_<neml2::AxisLayout> &);
void def(pybind11::module_ &, pybind11::class_<neml2::SparseVector> &);
void def(pybind11::module_ &, pybind11::class_<neml2::SparseMatrix> &);
void def(pybind11::module_ &, pybind11::class_<neml2::AssembledVector> &);
void def(pybind11::module_ &, pybind11::class_<neml2::AssembledMatrix> &);

// Type casters are only for cross-module types used in function signatures
DEFAULT_TYPECASTER(neml2::ModelNonlinearSystem, "neml2.es.NonlinearSystem");
DEFAULT_TYPECASTER_SHARED_PTR(neml2::ModelNonlinearSystem, "neml2.es.NonlinearSystem");
DEFAULT_TYPECASTER(neml2::AxisLayout, "neml2.es.AxisLayout");
DEFAULT_TYPECASTER(neml2::SparseVector, "neml2.es.SparseVector");
DEFAULT_TYPECASTER(neml2::SparseMatrix, "neml2.es.SparseMatrix");
DEFAULT_TYPECASTER(neml2::AssembledVector, "neml2.es.AssembledVector");
DEFAULT_TYPECASTER(neml2::AssembledMatrix, "neml2.es.AssembledMatrix");
