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

#include "neml2/models/finite_volume/FiniteVolumeAppendBoundaryCondition.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(FiniteVolumeAppendBoundaryCondition);

OptionSet
FiniteVolumeAppendBoundaryCondition::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Append a boundary condition value to the intermediate dimension.";

  options.set<VariableName>("input");
  options.set("input").doc() = "Input tensor to append the boundary condition to.";

  options.set_parameter<TensorName<Scalar>>("bc_value");
  options.set("bc_value").doc() = "Boundary condition value to append.";

  options.set_output("output");
  options.set("output").doc() =
      "Output tensor name. Defaults to input + '_with_bc_left' or '_with_bc_right'.";

  EnumSelection side_selection(
      {"left", "right"}, {static_cast<int>(Side::LEFT), static_cast<int>(Side::RIGHT)}, "left");
  options.set<EnumSelection>("side") = side_selection;
  options.set("side").doc() =
      "Which side to append the boundary condition value to. Options are: " + side_selection.join();

  return options;
}

FiniteVolumeAppendBoundaryCondition::FiniteVolumeAppendBoundaryCondition(const OptionSet & options)
  : Model(options),
    _input(declare_input_variable<Scalar>(options.get<VariableName>("input"))),
    _bc_value(declare_parameter<Scalar>("bc_value", "bc_value", true)),
    _side(options.get<EnumSelection>("side").as<Side>()),
    _output(options.get("output").user_specified()
                ? declare_output_variable<Scalar>("output")
                : declare_output_variable<Scalar>(options.get<VariableName>("input").with_suffix(
                      _side == Side::LEFT ? "_with_bc_left" : "_with_bc_right")))
{
}

void
FiniteVolumeAppendBoundaryCondition::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    auto bc_value = (_bc_value.intmd_dim() == 0 ? Scalar(_bc_value.unsqueeze(0), 0, 1) : _bc_value);
    bc_value = bc_value.dynamic_expand_as(_input());
    if (_side == Side::LEFT)
      _output = intmd_cat({bc_value, _input()}, 0);
    else
      _output = intmd_cat({_input(), bc_value}, 0);
  }

  if (dout_din)
  {
    // Build explicit block-identity derivatives along the intermediate dimension.
    //
    // Let input have N intermediate entries. The output has N+1 entries after appending a BC.
    // d(output)/d(input) is a (N+1)xN matrix with an identity block and a zero row.
    // d(output)/d(bc) is a (N+1)x1 vector with a single 1 at the appended location.
    const auto in_map = imap_v<Scalar>(_input.options()).intmd_expand(_input.intmd_size(-1));
    auto bc_value = (_bc_value.intmd_dim() == 0 ? Scalar(_bc_value.unsqueeze(0), 0, 1) : _bc_value);
    bc_value = bc_value.dynamic_expand_as(_input());
    const auto bc_map = imap_v<Scalar>(bc_value.options()).intmd_expand(bc_value.intmd_size(-1));
    const auto diag_in = intmd_diagonalize(in_map);        // N x N identity in intmd dims
    const auto zero_row = 0.0 * in_map.intmd_unsqueeze(0); // 1 x N zeros
    const auto zero_vec = 0.0 * in_map;                    // N zeros

    const auto bc_vec =
        (_side == Side::LEFT ? intmd_cat({bc_map, zero_vec}, 0) : intmd_cat({zero_vec, bc_map}, 0));
    const auto bc_mat = bc_vec.intmd_unsqueeze(1); // (N+1) x 1

    if (_input.is_dependent())
    {
      if (_side == Side::LEFT)
        _output.d(_input, 2, 1, 1) = intmd_cat({zero_row, diag_in}, 0);
      else
        _output.d(_input, 2, 1, 1) = intmd_cat({diag_in, zero_row}, 0);
    }

    if (const auto * const bc = nl_param("bc_value"))
      _output.d(*bc, 2, 1, 1) = bc_mat;
  }
}
} // namespace neml2
