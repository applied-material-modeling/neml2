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
#include <memory>
#include <string>

#include "neml2/csrc/dispatchers/DispatchedModel.h"
#include "neml2/csrc/dispatchers/WorkScheduler.h"
#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
/**
 * @brief Load a compiled model from a `neml2-compile` stub `.i`.
 *
 * The C++ mirror of Python's `load_model(path, name)`. Parses the stub with
 * `nmhit`: locates `[Models]/<model_name>` (which must be an `AOTIModel` shim),
 * reads its `artifact_path` (the per-device artifact folder), and applies the
 * solver config carried in the referenced `[Solvers]` block.
 *
 * `scheduler` is the C++-only dispatch opt-in -- it is never read from the `.i`.
 * When omitted (`nullptr`) the model runs without dispatch on `at::kCPU` (the
 * artifact's `cpu/` subfolder must exist). To dispatch, or to target a non-CPU
 * device, pass a scheduler whose device names the artifact subfolder to load
 * (e.g. `SimpleScheduler{"cuda", N}`); a missing subfolder is an error.
 *
 * Returns a @ref DispatchedModel -- the universal runnable handle. With no
 * scheduler and the whole batch in one chunk it is a zero-overhead pass-through
 * over the underlying `Model`.
 */
AOTI_EXPORT DispatchedModel load_model(const std::filesystem::path & stub_path,
                                       const std::string & model_name,
                                       std::shared_ptr<WorkScheduler> scheduler = nullptr);
} // namespace neml2::aoti
