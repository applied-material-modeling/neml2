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
//
// This uses torch's STABLE ABI (`torch/csrc/stable/library.h`) rather than the
// regular `torch::Library` C++ API. `torch::Library::def` lowers to a `_def`
// overload whose signature carries a `Tag` type that moved namespaces across the
// supported torch range (`at::Tag` -> `torch::headeronly::Tag`); a libneml2
// built against the max torch then fails to load against the min torch with
// "undefined symbol: torch::Library::_def(...)". The stable entry points
// (`aoti_torch_library_def`, which takes only a schema string -- no Tag -- plus
// `torch_library_impl` / `torch_call_dispatcher`) exist across the whole range,
// so the one wheel imports on every supported torch. See the torch-compat matrix
// (`.github/workflows/compat.yaml`).

#include <mutex>

#include <ATen/core/dispatch/Dispatcher.h>          // c10::Dispatcher::findOp (registration guard)
#include <torch/csrc/inductor/aoti_runtime/utils.h> // AOTI_TORCH_ERROR_CODE_CHECK (header-only)
#include <torch/csrc/stable/library.h> // StableLibrary + torch_call_dispatcher + StableIValue

namespace neml2::aoti
{
namespace
{
// Boxed runtime kernel for `neml2::opaque_pow`. The op is opaque only to
// Inductor's fusion pass; semantically and by signature it is exactly
// `aten::pow.Tensor_Tensor`, so we forward the boxed `[base, exponent]` stack
// straight to that op through the stable dispatcher. Forwarding the stack avoids
// any `at::Tensor` round-trip, so no unstable-ABI symbol leaks into the kernel
// body either. `TORCH_ABI_VERSION` is the extension's BUILD version (per the
// `torch_call_dispatcher` contract), which lets an older runtime libtorch
// negotiate compatibility. The fake/abstract impl and autograd backward are
// export-time concerns already baked into the artifact, so the runtime needs
// only this forward.
void
opaque_pow_boxed(StableIValue * stack, uint64_t /*num_args*/, uint64_t /*num_outputs*/)
{
  AOTI_TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::pow", "Tensor_Tensor", stack, TORCH_ABI_VERSION));
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
  // The stable-ABI library objects are heap-allocated and never freed: they must
  // outlive the process, and a destructor would pull in the __cxa_call_terminate
  // EH symbol the wheel/dev link toolchain cannot resolve. `def` (schema) and
  // `impl` (kernel) are separate library objects so the kernel can be pinned to
  // the CompositeExplicitAutograd key -- a bare FRAGMENT `impl` has no key slot.
  static std::once_flag flag;
  std::call_once(
      flag,
      []
      {
        if (c10::Dispatcher::singleton().findOp({"neml2::opaque_pow", ""}).has_value())
          return;
        using StableLibrary = torch::stable::detail::StableLibrary;
        auto * def_lib =
            new StableLibrary(StableLibrary::Kind::FRAGMENT, "neml2", "", __FILE__, __LINE__);
        def_lib->def("opaque_pow(Tensor base, Tensor exponent) -> Tensor");
        auto * impl_lib = new StableLibrary(
            StableLibrary::Kind::IMPL, "neml2", "CompositeExplicitAutograd", __FILE__, __LINE__);
        impl_lib->impl("opaque_pow", &opaque_pow_boxed);
      });
}
} // namespace neml2::aoti
