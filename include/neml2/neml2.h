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

#include "neml2/config.h"
#include "neml2/base/Factory.h"
#include "neml2/models/Model.h"

/**
 * Some neml2 libraries use the factory-registry pattern to dynamically register available
 * objects. However, some linkers will not include object files from the library if no symbols
 * from those object files are referenced.
 *
 * Of course, to work around this issue, one can use linker flags like --no-as-needed to force the
 * linker to include all object files from the library. However, this is not an ideal solution as it
 * may break other parts of the link line.
 *
 * The approach we take here is to define a symbol in the neml2_models library that is always
 * referenced when the library is used.
 */
extern "C" void _neml2_force_link_models();
extern "C" void _neml2_force_link_solvers();
extern "C" void _neml2_force_link_user_tensors();
extern "C" void _neml2_force_link_drivers();

namespace neml2
{
/**
 * @brief A convenient function to parse all options from an input file
 *
 * Previously loaded input options will be discarded!
 *
 * @warning All threads share the same input options, so in principle this function is not intended
 * to be called inside a threaded region.
 *
 * @param path Path to the input file to be parsed
 * @param additional_input Additional cliargs to pass to the parser
 */
std::unique_ptr<Factory> load_input(const std::filesystem::path & path,
                                    const std::string & additional_input = "");

/**
 * @brief A convenient function to load an input file and get a model
 *
 * @param path Path to the input file to be parsed
 * @param mname Name of the model
 */
std::shared_ptr<Model> load_model(const std::filesystem::path & path, const std::string & mname);
} // namespace neml2
