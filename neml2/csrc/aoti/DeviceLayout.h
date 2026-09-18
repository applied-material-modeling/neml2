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

#include <string>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

// Device-family helpers shared across the AOTI runtime and the dispatcher.
//
// The on-disk artifact layout is `<root>/<device>/<dtype>/<segment>.pt2`. The
// `<device>` segment is the lowercase family name (`cpu`, `cuda`, `xpu`, ...),
// which MUST agree byte-for-byte with what the Python writer (`neml2-compile`,
// via `neml2._accelerator.folder_name`) produces. `c10::DeviceTypeName(..., true)`
// is torch's own canonical lowercase spelling for each device type, so keeping
// both sides on it means adding a new accelerator family is a one-line change
// (extend `KNOWN_FAMILIES` on the Python side, extend `is_accelerator` here).
//
// Internal header -- not shipped with the wheel. Keep it inline so no extra
// translation unit and no new library dependencies are pulled in.

namespace neml2::aoti
{

// Lowercase family name for the `<device>/<dtype>/` artifact-leaf folder.
// Uses `c10::DeviceTypeName(..., /*lower_case=*/true)` so the spelling stays
// aligned with torch's own device-type enumeration; every known accelerator
// (CPU, CUDA, XPU, HIP, MPS, ...) round-trips through this consistently.
inline std::string
device_folder_name(at::Device d)
{
  return c10::DeviceTypeName(d.type(), /*lower_case=*/true);
}

// True when the device type is one NEML2 treats as an accelerator (a family
// distinct from CPU that owns its own artifact leaf and can be pinned to a
// specific index in the runtime dispatcher). Extend when adding a new family
// on the Python side.
inline bool
is_accelerator(at::DeviceType t)
{
  return t == at::kCUDA || t == at::kXPU || t == at::kHIP || t == at::kMPS;
}

} // namespace neml2::aoti
