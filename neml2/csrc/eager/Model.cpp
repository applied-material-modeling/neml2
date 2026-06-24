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
#include <utility>
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
// True if the Python exception `e` is (or derives from) the
// `neml2.aoti._aoti.ConvergenceError` registered by the aoti pybind module --
// the Python face of a solver divergence / max-iters thrown in libneml2.so. The
// import is a fast sys.modules lookup (neml2 is already imported by the time any
// op runs); guarded so a failed lookup degrades to "not a convergence error"
// rather than throwing while an exception is in flight. GIL must be held.
bool
is_convergence_error(const py::error_already_set & e)
{
  try
  {
    return e.matches(py::module_::import("neml2.aoti._aoti").attr("ConvergenceError"));
  }
  catch (...)
  {
    return false;
  }
}

// Run a Python-touching operation under the GIL and normalize any failure to the
// NEML2 exception taxonomy -- mirroring the aoti runtime's `_guarded` policy of
// presenting foreign torch/python errors through that taxonomy. The GIL is held
// for the whole call (including the catches), so reading a Python exception's
// message + type is safe.
template <class F>
auto
guarded(const char * op, F && f) -> decltype(f())
{
  const std::string prefix = std::string("neml2::eager::Model::") + op + ": ";
  py::gil_scoped_acquire gil;
  try
  {
    return f();
  }
  catch (const neml2::aoti::Exception &)
  {
    // A NEML2 C++ exception thrown directly on this side -- pass it through
    // unchanged so its recoverable() bit survives. (Not reachable on the pure
    // Python-marshalling path today, but cheap insurance against a future op
    // that calls into libneml2 C++ directly.)
    throw;
  }
  catch (const py::error_already_set & e)
  {
    // A Python exception crossed the embedded boundary. A solver divergence /
    // max-iters originates as neml2::aoti::ConvergenceError in libneml2.so and
    // surfaces here as the registered neml2.aoti._aoti.ConvergenceError; re-raise
    // it as the recoverable C++ type so a host can cut the step and retry.
    // Everything else is a fatal config / shape / dtype error.
    if (is_convergence_error(e))
      throw neml2::aoti::ConvergenceError(prefix + e.what());
    throw neml2::aoti::FatalError(prefix + e.what());
  }
  catch (const std::exception & e)
  {
    throw neml2::aoti::FatalError(prefix + e.what());
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
             std::optional<at::Device> device_override,
             const std::vector<std::string> & load)
{
  // Construct the Impl first -- this brings up the embedded interpreter via the
  // InterpreterGuard member (no GIL required yet). Kept outside `guarded`
  // because acquiring the GIL before the interpreter exists is invalid. A
  // failure here (interpreter bootstrap) is catastrophic and propagates as-is.
  _impl = std::make_unique<Impl>();

  guarded(
      "Model",
      [&]
      {
        // Force torch's pybind tensor caster registration (mirrors the
        // _aoti binding), then import the Python-side adapter.
        py::module_::import("torch");
        auto mod = py::module_::import("neml2.eager");
        // Load any external Python extensions into the registry BEFORE
        // constructing the model, so `factory.load_model` (invoked by
        // `_EagerModel` below) can resolve their `@register_neml2_object` types.
        // Reuses the exact CLI `--load` plumbing for identical semantics.
        if (!load.empty())
          py::module_::import("neml2.cli._extensions").attr("load_user_extensions")(load);
        py::object device = device_override.has_value() ? py::cast(*device_override) : py::none();
        _impl->adapter = mod.attr("_EagerModel")(input_file.string(), model_name, device);

        // Cache metadata once, under the GIL, so the accessors are GIL-free.
        _impl->input_names = _impl->adapter.attr("input_names").cast<std::vector<std::string>>();
        _impl->output_names = _impl->adapter.attr("output_names").cast<std::vector<std::string>>();
        _impl->input_base_shapes =
            _impl->adapter.attr("input_base_shapes").cast<std::vector<std::vector<int64_t>>>();
        _impl->output_base_shapes =
            _impl->adapter.attr("output_base_shapes").cast<std::vector<std::vector<int64_t>>>();
        _impl->param_names = _impl->adapter.attr("param_names").cast<std::vector<std::string>>();
        _impl->param_base_shapes =
            _impl->adapter.attr("param_base_shapes").cast<std::vector<std::vector<int64_t>>>();
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

const std::vector<std::vector<int64_t>> &
Model::input_base_shapes() const noexcept
{
  return _impl->input_base_shapes;
}

const std::vector<std::vector<int64_t>> &
Model::output_base_shapes() const noexcept
{
  return _impl->output_base_shapes;
}

// NOTE on tensor lifetime: the tensors returned by forward / jvp / jacobian
// originate in Python (the native model's outputs) and are handed back as
// at::Tensor across the boundary. The caller destroys them later in C++ without
// holding the GIL. That is safe: a Python-origin tensor carries torch's
// c10::impl::PyInterpreter hook, whose decref reacquires the GIL when it must
// drop the backing PyObject, so no caller-side GIL management is required.

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

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::jvp(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & tangents) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>;
  return guarded("jvp",
                 [&]() -> Ret
                 {
                   // The adapter returns (outputs, jvp_outputs) -- a 2-tuple of
                   // dict[str, Tensor] cast back to a pair of maps.
                   py::object out = _impl->adapter.attr("jvp")(inputs, tangents);
                   return out.cast<Ret>();
                 });
}

std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>
Model::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>;
  return guarded("jacobian",
                 [&]() -> Ret
                 {
                   // The adapter returns (outputs, J) -- a dict[str, Tensor] and
                   // the nested dict[str, dict[str, Tensor]] of variable-pair
                   // blocks (*B, *out_base, *in_base).
                   py::object out = _impl->adapter.attr("jacobian")(inputs);
                   return out.cast<Ret>();
                 });
}

const std::vector<std::string> &
Model::param_names() const noexcept
{
  return _impl->param_names;
}

const std::vector<std::vector<int64_t>> &
Model::param_base_shapes() const noexcept
{
  return _impl->param_base_shapes;
}

std::map<std::string, at::Tensor>
Model::named_parameters() const
{
  return guarded("named_parameters",
                 [&]() -> std::map<std::string, at::Tensor>
                 {
                   // The adapter returns dict[str, Tensor] of detached parameter
                   // values keyed by qualified name.
                   py::object out = _impl->adapter.attr("named_parameters")();
                   return out.cast<std::map<std::string, at::Tensor>>();
                 });
}

void
Model::set_parameter(const std::string & name, const at::Tensor & value)
{
  guarded("set_parameter", [&]() { _impl->adapter.attr("set_parameter")(name, value); });
}

std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>
Model::param_jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>;
  return guarded("param_jacobian",
                 [&]() -> Ret
                 {
                   // The adapter returns (outputs, P) -- a dict[str, Tensor] and
                   // the nested dict[str, dict[str, Tensor]] of parameter-pair
                   // blocks (*B, *out_base, *param_base).
                   py::object out = _impl->adapter.attr("param_jacobian")(inputs);
                   return out.cast<Ret>();
                 });
}

std::map<std::string, at::Tensor>
Model::param_vjp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & cotangents) const
{
  using Ret = std::map<std::string, at::Tensor>;
  return guarded("param_vjp",
                 [&]() -> Ret
                 {
                   // The adapter returns dL/d(param) keyed by parameter qname for
                   // the loss L = sum_o <cotangent_o, out_o>.
                   py::object out = _impl->adapter.attr("param_vjp")(inputs, cotangents);
                   return out.cast<Ret>();
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
