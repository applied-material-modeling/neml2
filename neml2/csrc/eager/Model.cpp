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

#include <map>
#include <string>
#include <vector>

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Registers the at::Tensor / at::Device / at::ScalarType <-> Python casters
// used to marshal across the embedded-interpreter boundary.
#include <torch/csrc/utils/pybind.h>

#include "neml2/csrc/eager/internal.h"

namespace py = pybind11;

namespace neml2::eager
{
namespace
{
// Run a Python-touching operation under the GIL and normalize any failure to
// neml2::aoti::FatalError -- mirroring the aoti runtime's `_guarded` policy of
// presenting foreign torch/python errors through the NEML2 exception taxonomy.
// The Python error message is read while the GIL is still held (the inner
// catch), so it is safe to format here.
template <class F>
auto
guarded(const char * op, F && f) -> decltype(f())
{
  try
  {
    py::gil_scoped_acquire gil;
    try
    {
      return f();
    }
    catch (const py::error_already_set & e)
    {
      throw neml2::aoti::FatalError(std::string("neml2::eager::Model::") + op + ": " + e.what());
    }
  }
  catch (const neml2::aoti::Exception &)
  {
    throw; // already a NEML2 exception -- pass through unchanged
  }
  catch (const std::exception & e)
  {
    throw neml2::aoti::FatalError(std::string("neml2::eager::Model::") + op + ": " + e.what());
  }
}
} // namespace

// Drop the pybind adapter reference under the GIL. After this the `adapter`
// member is a null handle, so its own destructor is a no-op and the remaining
// members + the InterpreterGuard tear down without needing the GIL. Guard on
// Py_IsInitialized in case a host finalized the interpreter out from under us.
Model::Impl::~Impl()
{
  if (Py_IsInitialized())
  {
    py::gil_scoped_acquire gil;
    adapter = py::object();
  }
}

Model::Model(const std::filesystem::path & input_file,
             const std::string & model_name,
             std::optional<at::Device> device_override)
{
  // Construct the Impl first -- this brings up the embedded interpreter via the
  // InterpreterGuard member (no GIL required yet). Kept outside `guarded`
  // because acquiring the GIL before the interpreter exists is invalid.
  try
  {
    _impl = std::make_unique<Impl>();
  }
  catch (const std::exception & e)
  {
    throw neml2::aoti::FatalError(
        std::string("neml2::eager::Model: failed to start the embedded Python interpreter: ") +
        e.what());
  }

  guarded(
      "Model",
      [&]
      {
        // Force torch's pybind tensor caster registration (mirrors the
        // _aoti binding), then import the Python-side adapter.
        py::module_::import("torch");
        auto mod = py::module_::import("neml2.eager");
        py::object device = device_override.has_value() ? py::cast(*device_override) : py::none();
        _impl->adapter = mod.attr("_EagerModel")(input_file.string(), model_name, device);

        // Cache metadata once, under the GIL, so the accessors are GIL-free.
        _impl->input_names = _impl->adapter.attr("input_names").cast<std::vector<std::string>>();
        _impl->output_names = _impl->adapter.attr("output_names").cast<std::vector<std::string>>();
        _impl->input_sizes = _impl->adapter.attr("input_sizes").cast<std::vector<int>>();
        _impl->output_sizes = _impl->adapter.attr("output_sizes").cast<std::vector<int>>();
        _impl->device = _impl->adapter.attr("device").cast<at::Device>();
        _impl->dtype = _impl->adapter.attr("dtype").cast<at::ScalarType>();
      });
}

Model::~Model() = default;

const std::vector<std::string> &
Model::input_names() const noexcept
{
  return _impl->input_names;
}

const std::vector<std::string> &
Model::output_names() const noexcept
{
  return _impl->output_names;
}

const std::vector<int> &
Model::input_sizes() const noexcept
{
  return _impl->input_sizes;
}

const std::vector<int> &
Model::output_sizes() const noexcept
{
  return _impl->output_sizes;
}

std::map<std::string, at::Tensor>
Model::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  return guarded("forward",
                 [&]() -> std::map<std::string, at::Tensor>
                 {
                   // std::map<string,Tensor> -> py dict (stl + torch casters);
                   // the adapter returns a dict[str, Tensor] cast back to a map.
                   py::object out = _impl->adapter.attr("forward")(inputs);
                   return out.cast<std::map<std::string, at::Tensor>>();
                 });
}

at::Device
Model::device() const noexcept
{
  return _impl->device;
}

at::ScalarType
Model::dtype() const noexcept
{
  return _impl->dtype;
}
} // namespace neml2::eager
