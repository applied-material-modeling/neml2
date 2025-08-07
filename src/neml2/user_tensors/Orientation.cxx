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

#include "neml2/user_tensors/Orientation.h"

#include "neml2/tensors/Quaternion.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/norm_sq.h"

namespace neml2
{
register_NEML2_object(Orientation);

OptionSet
Orientation::expected_options()
{
  OptionSet options = UserTensorBase<Rot>::expected_options();

  options.doc() = "An orientation, internally defined as a set of Modified Rodrigues parameters "
                  "given by \\f$ r = n \\tan{\\frac{\\theta}{4}} \\f$ with \\f$ n \\f$ the axis of "
                  "rotation and \\f$ \\theta \\f$ the rotation angle about that axis.  However, "
                  "this class provides a variety of ways to define the orientation in terms of "
                  "other, more common representations.";

  options.set<std::string>("input_type") = "euler_angles";
  options.set("input_type").doc() =
      "The method used to define the angles, 'euler_angles' or 'random'";

  options.set<std::string>("angle_convention") = "kocks";
  options.set("angle_convention").doc() = "Euler angle convention, 'Kocks', 'Roe', or 'Bunge'";

  options.set<std::string>("angle_type") = "degrees";
  options.set("angle_type").doc() = "Type of angles, either 'degrees' or 'radians'";

  options.set<std::vector<double>>("values") = {};
  options.set("values").doc() = "Input Euler angles, as a flattened n-by-3 matrix";

  options.set<bool>("normalize") = false;
  options.set("normalize").doc() =
      "If true do a shadow parameter replacement of the underlying MRP representation to move the "
      "inputs farther away from the singularity";

  options.set<unsigned int>("quantity") = 1;
  options.set("quantity").doc() = "Number (batch size) of random orientations";

  return options;
}

Orientation::Orientation(const OptionSet & options)
  : UserTensorBase<Rot>(options)
{
}

static Rot
expand_as_needed(const Rot & input, unsigned int inp_size)
{
  if (inp_size > 1)
    return input.dynamic_expand({inp_size});

  return input;
}

Rot
Orientation::make() const
{
  const auto & options = this->input_options();
  std::string input_type = options.get<std::string>("input_type");

  Rot R;
  if (input_type == "euler_angles")
  {
    auto vals = options.get<std::vector<double>>("values");
    auto t = neml2::Tensor::create(vals);
    auto v = Vec(t.reshape({-1, 3}), 0);
    R = expand_as_needed(Rot::fill_euler_angles(v,
                                                options.get<std::string>("angle_convention"),
                                                options.get<std::string>("angle_type")),
                         options.get<unsigned int>("quantity"));
  }
  else if (input_type == "random")
  {
    R = Rot::rand({options.get<unsigned int>("quantity")}, {});
  }
  else
    throw NEMLException("Unknown Orientation input_type " + input_type);

  if (options.get<bool>("normalize"))
    return neml2::where(norm_sq(R) < 1.0, R, R.shadow());

  return R;
}

} // namespace neml2
