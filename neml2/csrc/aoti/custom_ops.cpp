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

#include <mutex>

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

} // namespace

void
ensure_neml2_custom_ops_registered()
{
  // Registered lazily -- on first aoti Model construction, not from a static
  // initializer at libneml2 load -- and only if absent. A single process can
  // hold BOTH libneml2 and the embedded-Python interpreter (MOOSE's cpp-eager
  // runtime, or py-aoti); registering at load time raced the Python package's
  // identical `def` in neml2/types/functions.py ("registered multiple times").
  // Deferring to first use means the cpp-eager path -- which constructs no aoti
  // Model -- never registers in C++, so the Python package owns the op there;
  // the pure cpp-aoti path (no Python) registers it here; py-aoti (Python
  // imported first) skips via the findOp guard.
  //
  // The torch::Library is heap-allocated and never freed: it must outlive the
  // process, and a destructor would pull in the __cxa_call_terminate EH symbol
  // the wheel/dev link toolchain cannot resolve.
  static std::once_flag flag;
  std::call_once(flag,
                 []
                 {
                   if (c10::Dispatcher::singleton().findOp({"neml2::opaque_pow", ""}).has_value())
                     return;
                   auto * lib = new torch::Library(
                       torch::Library::FRAGMENT, "neml2", std::nullopt, __FILE__, __LINE__);
                   lib->def("opaque_pow(Tensor base, Tensor exponent) -> Tensor");
                   lib->impl("opaque_pow",
                             torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
                                             TORCH_FN(opaque_pow)));
                 });
}
} // namespace neml2::aoti
