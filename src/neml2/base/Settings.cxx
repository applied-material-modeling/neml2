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

#include <ATen/Parallel.h>

#include "neml2/misc/defaults.h"
#include "neml2/base/Settings.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/base/OptionSet.h"

namespace neml2
{
OptionSet
Settings::expected_options()
{
  OptionSet options;
  options.section() = "Settings";
  options.doc() = "Global settings for tensors, models, etc.";

  options.set<std::string>("type") = "Settings";

  EnumSelection dtype_selection(
      {"Float16", "Float32", "Float64"},
      {static_cast<int>(kFloat16), static_cast<int>(kFloat32), static_cast<int>(kFloat64)},
      NEML2_DEFAULT_DTYPE_STR);
  options.set<EnumSelection>("default_floating_point_type") = dtype_selection;
  options.set("default_floating_point_type").doc() =
      "Default floating point type for tensors. Options are " + dtype_selection.candidates_str();

  EnumSelection int_dtype_selection({"Int8", "Int16", "Int32", "Int64"},
                                    {static_cast<int>(kInt8),
                                     static_cast<int>(kInt16),
                                     static_cast<int>(kInt32),
                                     static_cast<int>(kInt64)},
                                    NEML2_DEFAULT_INTEGER_DTYPE_STR);
  options.set<EnumSelection>("default_integer_type") = int_dtype_selection;
  options.set("default_integer_type").doc() =
      "Default integer type for tensors. Options are " + int_dtype_selection.candidates_str();

  options.set<std::string>("default_device") = NEML2_DEFAULT_DEVICE_STR;
  options.set("default_device").doc() =
      "Default device on which tensors are created and models are evaluated. The string supplied "
      "must follow the following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies "
      "the device type, and :<device-index> optionally specifies a device index. For example, "
      "device='cpu' sets the target compute device to be CPU, and device='cuda:1' sets the target "
      "compute device to be CUDA with device ID 1.";

  options.set<Real>("machine_precision") = NEML2_DEFAULT_MACHINE_PRECISION;
  options.set("machine_precision").doc() =
      "Machine precision used at various places to workaround singularities like division-by-zero.";

  options.set<Real>("tolerance") = NEML2_DEFAULT_TOLERANCE;
  options.set("tolerance").doc() = "Tolerance used in various algorithms.";

  options.set<Real>("tighter_tolerance") = NEML2_DEFAULT_TIGHTER_TOLERANCE;
  options.set("tighter_tolerance").doc() = "A tighter tolerance used in various algorithms.";

  options.set<std::string>("buffer_name_separator") = NEML2_DEFAULT_BUFFER_NAME_SEPARATOR;
  options.set("buffer_name_separator").doc() = "Nested buffer name separator. The default is '_'. "
                                               "For example, a sub-model 'foo' which declares "
                                               "a buffer 'bar' will have a buffer named 'foo_bar'.";

  options.set<std::string>("parameter_name_separator") = NEML2_DEFAULT_PARAMETER_NAME_SEPARATOR;
  options.set("parameter_name_separator").doc() =
      "Parameter name separator. The default is '_'. For example, a sub-model 'foo' which declares "
      "a parameter 'bar' will have a parameter named 'foo_bar'.";

  return options;
}

void
Settings::apply(const OptionSet & options)
{
  // Default floating point dtype
  default_dtype() = options.get<EnumSelection>("default_floating_point_type").as<Dtype>();
  c10::set_default_dtype(scalarTypeToTypeMeta(default_dtype()));

  // Default integral dtype
  default_integer_dtype() = options.get<EnumSelection>("default_integer_type").as<Dtype>();

  // Default device
  default_device() = Device(options.get<std::string>("default_device"));

  // Machine precision
  machine_precision() = options.get<Real>("machine_precision");

  // Tolerances
  tolerance() = options.get<Real>("tolerance");
  tighter_tolerance() = options.get<Real>("tighter_tolerance");

  // Buffer/parameter name separator
  buffer_name_separator() = options.get<std::string>("buffer_name_separator");
  parameter_name_separator() = options.get<std::string>("parameter_name_separator");
}
} // namespace neml2
