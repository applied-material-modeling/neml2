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

#include <filesystem>
#include <string>

#include "neml2/csrc/eager/Model.h"
#include "neml2/csrc/eager/eager_export.h"

namespace neml2::eager
{
/// Load the model named `model_name` from the HIT input file `input_file` for
/// eager (uncompiled) evaluation in an embedded Python interpreter. This is the
/// C++ mirror of Python's `neml2.factory.load_model(input_file, model_name)`
/// and the eager counterpart of @ref neml2::aoti::load_model -- but it consumes
/// the *original* `.i`, not a `neml2-compile` stub.
///
/// Returns by value; @ref Model is non-movable, so this relies on C++17
/// guaranteed copy elision (the returned prvalue is constructed directly into
/// the caller's storage).
///
/// Throws @ref neml2::aoti::FatalError on any failure (interpreter bootstrap,
/// missing `neml2` package, parse error, unknown model name).
EAGER_EXPORT Model load_model(const std::filesystem::path & input_file,
                              const std::string & model_name);
} // namespace neml2::eager
