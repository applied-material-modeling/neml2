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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/Model.h" // neml2::aoti::VariablePairJacobian (shared return type)
#include "neml2/csrc/eager/eager_export.h"

namespace neml2::eager
{
/**
 * @brief Eager (uncompiled) runtime for a Python-native NEML2 model.
 *
 * A drop-in sibling of @ref neml2::aoti::Model that skips the AOTI compile
 * step entirely: it embeds a CPython interpreter, loads the model in Python
 * via `neml2.factory.load_model` (the same path `neml2-run` uses), and
 * evaluates `forward` by marshalling `at::Tensor`s across the C++/Python
 * boundary. It is the fast-to-start, slow-to-run counterpart of the AOTI
 * runtime -- intended for downstream C++ **unit tests** that cannot afford a
 * minutes-long `neml2-compile`. For production, use @ref neml2::aoti::Model /
 * @ref neml2::dispatchers::DispatchedModel.
 *
 * **Original input file, no stub.** Unlike `aoti::Model` (constructed from a
 * `_meta.json`) this is constructed from the *original* `.i` file plus the
 * model's name in the `[Models]` section -- there is no compiled artifact and
 * no generated stub. @ref load_model is the convenience factory.
 *
 * **API parity.** `forward` and the metadata accessors (`input_names` /
 * `output_names` / `input_base_shapes` / `output_base_shapes` /
 * `parameter_base_shapes` / `device` / `dtype`) have signatures identical to
 * `aoti::Model`, so test code switches between the two runtimes by changing only
 * the load call. The Python adapter reports names/sizes through the same helper
 * the AOTI metadata uses, so the two agree for the same model. `forward`, `jvp`,
 * and `jacobian` are all implemented (the latter two on the native model's `v=`
 * chain rule).
 *
 * **Sub-batch.** Variables are `(*B, *sub_batch_shape, *base_shape)`, as on
 * `aoti::Model`. The sub-batch extents come from the input file's
 * `[Settings]/example_batch_shape` (see @ref input_sub_batch_shapes); an
 * undeclared input is plain-batch. Derivatives (`jvp` / `jacobian`) are taken
 * with respect to plain-batch inputs only, and the parameter derivatives are
 * plain-batch only.
 *
 * **PImpl + visibility.** Every implementation detail -- the embedded
 * interpreter handle and the `pybind11` adapter object -- lives behind an
 * opaque `Impl` (in the non-shipped `internal.h`). The installed header is
 * Python-free; the shared library exports only the `EAGER_EXPORT`-tagged
 * surface below.
 *
 * **Exceptions.** Reuses the @ref neml2::aoti exception taxonomy (linked from
 * `libneml2.so`): a Python-side failure is normalized to
 * @ref neml2::aoti::FatalError, except a solver divergence / max-iters, which is
 * re-raised as the recoverable @ref neml2::aoti::ConvergenceError (it originates
 * as that type in `libneml2.so` and survives the C++ -> Python -> C++ round trip
 * via the `neml2.aoti._aoti.ConvergenceError` registration).
 *
 * **Threading.** Each Python-touching call holds the GIL, so concurrent
 * `forward` calls from multiple host threads are serialized. This is fine for
 * the unit-test use case; for parallel throughput use the AOTI / dispatcher
 * path.
 *
 * See `doc/content/model_compilation/eager.md`.
 */
class EAGER_EXPORT Model
{
public:
  /// Load the model named `model_name` from the HIT input file `input_file`
  /// (the original `.i`, not a compiled stub) by constructing it in an embedded
  /// Python interpreter. Throws @ref neml2::aoti::FatalError if the interpreter
  /// cannot start, the `neml2` package is not importable, the file cannot be
  /// parsed, `model_name` is not found, or the file's
  /// `[Settings]/example_batch_shape` names an unknown variable or its output
  /// sub-batch shapes cannot be worked out (the compile plan run at load fails,
  /// even if the model itself would evaluate).
  ///
  /// `device_override`, when set, moves the model onto that device (e.g.
  /// `cuda:1`); when omitted, the model's natural device is used.
  ///
  /// `load` is an optional list of external Python extensions (file paths,
  /// package directories, or importable dotted module names) imported into the
  /// embedded interpreter BEFORE the model is constructed -- the embedded-eager
  /// counterpart of the `--load` flag on `neml2-run` / `neml2-compile`. Each
  /// imported module self-registers its `@register_neml2_object` types into the
  /// factory, so a downstream C++ consumer can drive Python-authored models that
  /// live outside the installed `neml2` package. Same resolution + ordering as
  /// the CLI (see `neml2.cli._extensions.load_user_extensions`).
  explicit Model(const std::filesystem::path & input_file,
                 const std::string & model_name,
                 std::optional<at::Device> device_override = std::nullopt,
                 const std::vector<std::string> & load = {});

  /// Declared here and defined out-of-line where `Impl` is complete, as the
  /// PImpl idiom requires.
  ~Model();

  // Non-copyable, non-movable -- the Impl owns a pybind object tied to the
  // embedded interpreter. Held by callers as an automatic, a unique_ptr, or
  // returned by value from @ref load_model (guaranteed copy elision).
  Model(const Model &) = delete;
  Model(Model &&) = delete;
  Model & operator=(const Model &) = delete;
  Model & operator=(Model &&) = delete;

  /// Master inputs / outputs in declaration order. Cached at load; no GIL.
  const std::vector<std::string> & input_names() const noexcept;
  const std::vector<std::string> & output_names() const noexcept;

  /// Per-name base shape (Scalar -> `{}`, SR2 -> `{6}`, R2 -> `{3,3}`). Inputs
  /// must be passed at their canonical `(*B, *sub_batch_shape, *base_shape)`
  /// shape. Cached at load; no GIL. Mirrors
  /// @ref neml2::aoti::Model::input_base_shapes.
  const std::vector<std::vector<int64_t>> & input_base_shapes() const noexcept;
  const std::vector<std::vector<int64_t>> & output_base_shapes() const noexcept;

  /// Per-input sub-batch shape, parallel to `input_names()`. Identical
  /// signature to @ref neml2::aoti::Model::input_sub_batch_shapes, so a host
  /// introspects either runtime the same way.
  ///
  /// On this runtime an entry reports the *declared* extent, read from the
  /// input file's `[Settings]/example_batch_shape`, so an empty entry means
  /// "not declared as sub-batched" rather than "plain-batch". An undeclared
  /// input is evaluated as plain-batch. Cached at load; no GIL.
  const std::vector<std::vector<int64_t>> & input_sub_batch_shapes() const noexcept;

  /// Per-output sub-batch shape, parallel to `output_names()`. Identical
  /// signature to @ref neml2::aoti::Model::output_sub_batch_shapes, so a host
  /// introspects either runtime the same way -- but *not* the same semantics.
  ///
  /// When the input file declares a sub-batch extent in
  /// `[Settings]/example_batch_shape`, the entries are what `neml2-compile`
  /// would record for the same declaration, worked out at load without
  /// compiling or solving. Without a declaration every entry is empty, meaning
  /// "not declared", not "plain-batch": a model can still return a sub-batched
  /// output (e.g. one built on a sub-batched parameter grid). On the AOTI route
  /// the answer is always definite. A host that cares about the difference
  /// treats an undeclared output as plain-batch and checks the shapes it gets
  /// on the first evaluation. Cached at load; no GIL.
  const std::vector<std::vector<int64_t>> & output_sub_batch_shapes() const noexcept;

  /// Evaluate the model. `inputs` is keyed by the names returned by
  /// `input_names()` and shaped `(*B, *sub_batch_shape, *base_shape)` (see
  /// `input_sub_batch_shapes()`); missing keys throw, a non-canonical trailing
  /// shape is rejected, and a tensor that does not carry its declared sub-batch
  /// extent is rejected. Returns one contiguous tensor per name in
  /// `output_names()` at `(*B, *out_sub_batch_shape, *out_base_shape)`.
  /// Acquires the GIL.
  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const;

  /// Evaluate + JVP. `tangents` is keyed like `inputs`, each at `(*B, *in_base)`;
  /// a missing key defaults to a zero tangent. Returns
  /// `{outputs, jvp_outputs}` -- both maps keyed by `output_names()`;
  /// `jvp_outputs[name]` is the directional derivative at the output's natural
  /// `(*B, *out_sub_batch_shape, *out_base_shape)`. A tangent for a sub-batched
  /// input is rejected. Mirrors @ref neml2::aoti::Model::jvp. Holds the GIL.
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const;

  /// Evaluate + full Jacobian as unflattened variable-pair blocks. Returns
  /// `{outputs, J}` where `J[out_name][in_name]` is
  /// `(*B, *out_sub_batch, *out_base, *in_base)` for a plain-batch input (see
  /// @ref neml2::aoti::VariablePairJacobian), matching
  /// @ref neml2::aoti::Model::jacobian.
  ///
  /// `derivatives` takes `neml2-compile -d` specs; empty selects every output
  /// against the plain-batch inputs. A sub-batched input is included only when
  /// named; its block is `(*B, *sub_batch, *out_base, *in_base)` when site-local,
  /// else `(*B, *out_sub_batch, *out_base, *in_sub_batch, *in_base)` (the
  /// "paired" / "full" forms of @ref neml2::aoti::Model::derivative_forms).
  /// Acquires the GIL.
  std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs,
           const std::vector<std::string> & derivatives = {}) const;

  /// Per-calibration-parameter natural base shape, keyed by qualified name
  /// (e.g. `"elasticity.E"` -> `{}` for a Scalar, `{6}` for an SR2) -- the
  /// parameter analogue of `input_base_shapes()`. The map keys are the
  /// calibration parameters; same keys as `named_parameters()`. Identical
  /// signature to @ref neml2::aoti::Model::parameter_base_shapes, so consumers
  /// introspect the parameter surface the same way on either runtime. Cached at
  /// load; no GIL.
  const std::map<std::string, std::vector<int64_t>> & parameter_base_shapes() const noexcept;

  /// Current values of the calibration parameters, keyed by qualified name,
  /// each a detached tensor at its `(*param_base)` shape. The read side of the
  /// parameter surface MOOSE drives. Acquires the GIL.
  std::map<std::string, at::Tensor> named_parameters() const;

  /// Replace a calibration parameter's value (the write side). *name* is the
  /// qualified parameter name; *value* is taken at its `(*param_base)` shape and
  /// copied into the live parameter on the embedded model (forwards to torch).
  /// Acquires the GIL.
  void set_parameter(const std::string & name, const at::Tensor & value);

  /// Evaluate + parameter Jacobian as unflattened variable-pair blocks. Returns
  /// `{outputs, P}` where `P[out_name][param_qname]` is `(*B, *out_base,
  /// *param_base)` -- the parameter analogue of `jacobian()` (reverse-mode AD
  /// over the model's parameters; the input chain rule is untouched). Holds the GIL.
  std::pair<std::map<std::string, at::Tensor>, neml2::aoti::VariablePairJacobian>
  param_jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  /// Parameter VJP / adjoint: `dL/d(param)` keyed by parameter qualified name for
  /// the loss `L = sum_o <cotangent_o, out_o>`. `cotangents` is keyed by output
  /// name (each at the output's `(*B, *out_base)` shape). One reverse pass total
  /// -- the cheap form for many-parameter inverse optimization, the parameter
  /// analogue of `param_jacobian()`. Holds the GIL.
  std::map<std::string, at::Tensor>
  param_vjp(const std::map<std::string, at::Tensor> & inputs,
            const std::map<std::string, at::Tensor> & cotangents) const;

  /// Device + dtype the model runs on. Cached at load; no GIL.
  at::Device device() const noexcept;
  at::ScalarType dtype() const noexcept;

private:
  // Opaque implementation; defined in the non-shipped internal.h + Model.cpp.
  // EAGER_NO_EXPORT keeps the nested type out of the shared library's export
  // table (it holds pybind/Python members that must never be part of the ABI).
  struct EAGER_NO_EXPORT Impl;
  std::unique_ptr<Impl> _impl;
};
} // namespace neml2::eager
