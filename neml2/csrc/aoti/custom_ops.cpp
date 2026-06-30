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

// ----------------------------------------------------------------------------
// C++ registration of NEML2's custom Torch operators into libneml2
// ----------------------------------------------------------------------------
// A compiled `.pt2` can reference NEML2 custom ops that `torch.export`
// deliberately preserves -- chiefly `neml2::opaque_pow`, the Inductor fusion
// barrier defined in `neml2/types/functions.py` (used by crystal-plasticity's
// PowerLawSlipRule and PerzynaPlasticFlowRate). The Python package registers
// those ops on import, but the cpp-aoti / cpp-dispatch runtimes load the
// artifact through libneml2 WITHOUT importing the Python package -- so libneml2
// must register the same schema + a runtime kernel itself, or the dispatcher
// throws "Could not find schema for neml2::opaque_pow" when the artifact loads.

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/pow.h>
#include <torch/library.h>

namespace neml2::aoti
{
namespace
{
// Runtime kernel for `neml2::opaque_pow`. The op is opaque only to Inductor's
// fusion pass; semantically it is a plain power. The fake (abstract) impl and
// the autograd backward are export-time concerns already baked into the
// artifact, so the C++ runtime needs only this forward.
at::Tensor
opaque_pow(const at::Tensor & base, const at::Tensor & exponent)
{
  return at::pow(base, exponent);
}

// Register at library load -- but only if the op is not already present. A host
// that imports the Python neml2 package (the py-aoti route) has already defined
// `neml2::opaque_pow`, and a second `def` would abort the process; guarding lets
// a single libneml2 serve both the pure-C++ host (we register) and the Python
// host (we defer). The kernel is registered on `CompositeExplicitAutograd` so a
// single definition serves every backend (cpu + cuda).
struct CustomOpRegistrar
{
  CustomOpRegistrar()
  {
    // Two constraints, both to avoid pulling in __cxa_call_terminate -- a
    // libstdc++ EH symbol the wheel/dev link toolchain fails to resolve when
    // executables link against libneml2 (this is the first torch::Library in the
    // library):
    //   * no exception may escape this static-initializer ctor (an escape would
    //     terminate the process), hence the try/catch; and
    //   * the torch::Library must have no destructor (a noexcept dtor's
    //     terminate-on-throw thunk is the actual culprit here), hence it is
    //     heap-allocated and intentionally never freed -- the registration must
    //     persist for the whole process anyway.
    // The expected "op already defined by the Python neml2 package" case
    // (py-aoti) is handled by the findOp guard, not by the catch.
    try
    {
      if (c10::Dispatcher::singleton().findOp({"neml2::opaque_pow", ""}).has_value())
        return;
      auto * lib =
          new torch::Library(torch::Library::FRAGMENT, "neml2", std::nullopt, __FILE__, __LINE__);
      lib->def("opaque_pow(Tensor base, Tensor exponent) -> Tensor");
      lib->impl("opaque_pow",
                torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, TORCH_FN(opaque_pow)));
    }
    catch (...)
    {
    }
  }
};

const CustomOpRegistrar registrar;
} // namespace
} // namespace neml2::aoti
