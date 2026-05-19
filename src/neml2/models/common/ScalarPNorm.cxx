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

#include "neml2/models/common/ScalarPNorm.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(ScalarPNorm);

OptionSet
ScalarPNorm::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Weighted \\f$ p \\f$-norm of an arbitrary number of Scalar inputs: "
      "\\f$ y = (\\sum_i w_i |x_i|^p + \\varepsilon)^{1/p} \\f$. The weights default to 1 "
      "(giving the standard \\f$ p \\f$-norm) and can be set per-input via the `weights` "
      "option, mirroring `LinearCombination`. The dtype-aware regularizer "
      "\\f$ \\varepsilon \\f$ comes from `neml2::machine_precision()`.";

  options.add<std::vector<VariableName>, FType::INPUT>("from", "Scalar variables to be combined");
  options.add_output("to", "The weighted p-norm output");
  options.add_parameter<Scalar>("exponent", "The exponent");

  options.add<std::vector<TensorName<Scalar>>, FType::BUFFER>(
      "weights",
      {TensorName<Scalar>("1")},
      "Per-input weights. List length must be 1 or `from`-length; a single value is broadcast "
      "to all inputs.");
  options.add<std::vector<bool>>(
      "weight_as_parameter",
      {false},
      "If true, declare the weights as (trainable) parameters; otherwise as buffers. List "
      "length must be 1 or `from`-length; a single value applies to all weights.");

  return options;
}

ScalarPNorm::ScalarPNorm(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to")),
    // allow_nonlinear=false: this model does not provide _to.d(*p).
    _p(declare_parameter<Scalar>("p", "exponent", false))
{
  for (const auto & fv : options.get<std::vector<VariableName>>("from"))
    _from.push_back(&declare_input_variable<Scalar>(fv));
  neml_assert(!_from.empty(), "ScalarPNorm requires at least one input.");

  // Per-input weight declarations follow the LinearCombination idiom: weights are buffers (or
  // optionally parameters via `weight_as_parameter`), set at construction time from a
  // TensorName<Scalar> reference. They are *not* input Variables that flow from another model.
  auto w_as_param = options.get<std::vector<bool>>("weight_as_parameter");
  neml_assert(w_as_param.size() == 1 || w_as_param.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " entries in weight_as_parameter, got ",
              w_as_param.size(),
              ".");
  if (w_as_param.size() == 1)
    w_as_param = std::vector<bool>(_from.size(), w_as_param[0]);

  const auto w_refs = options.get<std::vector<TensorName<Scalar>>>("weights");
  neml_assert(w_refs.size() == 1 || w_refs.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " weights, got ",
              w_refs.size(),
              ".");

  _weights.resize(_from.size());
  for (std::size_t i = 0; i < _from.size(); i++)
  {
    const auto & w_ref = w_refs.size() == 1 ? w_refs[0] : w_refs[i];
    if (w_as_param[i])
      _weights[i] =
          &declare_parameter<Scalar>("w_" + std::to_string(i), w_ref, /*allow_nonlinear=*/false);
    else
      _weights[i] = &declare_buffer<Scalar>("w_" + std::to_string(i), w_ref);
  }
}

void
ScalarPNorm::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_from[0]->scalar_type());

  // Sum_i w_i * |x_i|^p. The `where`-mask-and-safe-base pattern guards the autograd path for
  // the `p` parameter against the `pow`-base-zero singularity: d/dp [x^p] = x^p * log(x), and
  // PyTorch evaluates that as 0 * -inf = NaN at x=0 (rather than the analytically-correct 0,
  // the limit of x^p log(x) as x -> 0+). We substitute a safe base of 1 in the masked-off
  // branch and zero the term out, so the autograd graph never sees log(0). The forward value
  // is unchanged (|0|^p = 0 either way). The mask is detached so the boolean comparison
  // doesn't contribute to the graph. The CPU `at::pow_backward` happens to special-case
  // base=0 silently; the CUDA kernel does not, so this surfaces as `nan` only on CUDA.
  auto add_term = [this](std::size_t i)
  {
    const auto base = neml2::abs((*_from[i])());
    const auto safe = (base > 0.0).detach();
    const auto safe_base = neml2::where(safe, base, Scalar::ones_like(base));
    const auto term = neml2::where(safe, neml2::pow(safe_base, _p), Scalar::zeros_like(base));
    return (*_weights[i]) * term;
  };

  auto sum = add_term(0);
  for (std::size_t i = 1; i < _from.size(); i++)
    sum = sum + add_term(i);

  // y = (sum + eps)^(1/p). The regularizer keeps y bounded below by eps^(1/p) > 0,
  // so 1/y in the Jacobian stays finite.
  const auto inv_p = 1.0 / _p;
  const auto y = neml2::pow(sum + eps, inv_p);

  if (out)
    _to = y;

  if (dout_din)
  {
    // dy/dx_i = w_i * sign(x_i) * |x_i|^(p-1) * y^(1-p).
    // For p=2 with w_i=1 this collapses to x_i / y.
    const auto y_pow = neml2::pow(y, 1.0 - _p);
    for (std::size_t i = 0; i < _from.size(); i++)
    {
      const auto xi = (*_from[i])();
      _to.d(*_from[i]) =
          (*_weights[i]) * neml2::sign(xi) * neml2::pow(neml2::abs(xi), _p - 1.0) * y_pow;
    }
  }
}
} // namespace neml2
