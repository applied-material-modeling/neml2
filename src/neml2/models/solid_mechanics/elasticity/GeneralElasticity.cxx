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

#include "neml2/models/solid_mechanics/elasticity/GeneralElasticity.h"
#include "neml2/tensors/SSFR5.h"
#include "neml2/tensors/SSSSR8.h"

namespace neml2
{
register_NEML2_object(GeneralElasticity);

OptionSet
GeneralElasticity::expected_options()
{
  OptionSet options = AnisotropicElasticity::expected_options();
  options.doc() += " This verion implements a general relation using the elasticity tensor, "
                   "expressed as an SSR4 object";

  options.set_parameter<TensorName>("elastic_stiffness_tensor");
  options.set("elastic_stiffness_tensor").doc() = "Elastic stiffness tensor";

  return options;
}

GeneralElasticity::GeneralElasticity(const OptionSet & options)
  : AnisotropicElasticity(options),
    _T(declare_parameter<SSR4>("T", "elastic_stiffness_tensor", /*allow nonlinear=*/true))
{
}

void
GeneralElasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto A = _T.rotate(_R);
  const auto Ainv = _compliance ? A.inverse() : SSR4();

  if (out)
    _to = (_compliance ? Ainv : A) * _from;

  if (dout_din)
  {
    if (_from.is_dependent())
      _to.d(_from) = _compliance ? Ainv : A;

    if (_R.is_dependent())
    {
      const auto dA_dR = _T.drotate(_R);
      if (_compliance)
        _to.d(_R) = Tensor(at::einsum("...ijkl,...klm,...j", {A.dinverse(), dA_dR, _from}),
                           A.batch_sizes());
      else
        _to.d(_R) = Tensor(at::einsum("...ijk,...j", {dA_dR, _from}), A.batch_sizes());
    }

    if (const auto * const T = nl_param("T"))
    {
      const auto dA_dT = _T.drotate_self(_R);
      if (_compliance)
        _to.d(*T) = Tensor(at::einsum("...ijkl,...klmn,...j", {A.dinverse(), dA_dT, _from}),
                           A.batch_sizes());
      else
        _to.d(*T) = Tensor(at::einsum("...ijkl,...j", {dA_dT, _from}), A.batch_sizes());
    }
  }
}
} // namespace neml2
