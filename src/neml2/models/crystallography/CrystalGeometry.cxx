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

#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/base/Factory.h"
#include "neml2/base/TensorName.h"
#include "neml2/tensors/crystallography.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/MillerIndex.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/normalize_gcd.h"
#include "neml2/tensors/functions/vdot.h"
#include "neml2/tensors/functions/cross.h"
#include "neml2/tensors/functions/norm.h"
#include "neml2/tensors/functions/unit.h"
#include "neml2/tensors/functions/outer.h"
#include "neml2/misc/assertions.h"

namespace neml2::crystallography
{

register_NEML2_object(CrystalGeometry);

OptionSet
CrystalGeometry::expected_options()
{
  OptionSet options = Data::expected_options();

  options.doc() =
      "A Data object storing basic crystallographic information for a given crystal system.";

  options.set_buffer<TensorName<R2>>("crystal_class");
  options.set("crystal_class").doc() = "The set of symmetry operations defining the crystal class.";

  options.set_buffer<TensorName<Vec>>("lattice_vectors");
  options.set("lattice_vectors").doc() =
      "The three lattice vectors defining the crystal translational symmetry";

  options.set_buffer<TensorName<MillerIndex>>("slip_directions");
  options.set("slip_directions").doc() = "A list of Miller indices defining the slip directions";

  options.set_buffer<TensorName<MillerIndex>>("slip_planes");
  options.set("slip_planes").doc() = "A list of Miller indices defining the slip planes";

  return options;
}

CrystalGeometry::CrystalGeometry(const OptionSet & options)
  : CrystalGeometry(options, options.get<Factory *>("_factory"))
{
}

CrystalGeometry::CrystalGeometry(const OptionSet & options, Factory * factory)
  : CrystalGeometry(options,
                    options.get<TensorName<R2>>("crystal_class").resolve(factory),
                    options.get<TensorName<Vec>>("lattice_vectors").resolve(factory),
                    options.get<TensorName<MillerIndex>>("slip_directions").resolve(factory),
                    options.get<TensorName<MillerIndex>>("slip_planes").resolve(factory))
{
}

CrystalGeometry::CrystalGeometry(const OptionSet & options,
                                 const R2 & cclass,
                                 const Vec & lattice_vectors,
                                 const MillerIndex & slip_directions,
                                 const MillerIndex & slip_planes)
  : CrystalGeometry(options,
                    cclass,
                    lattice_vectors,
                    setup_schmid_tensors(lattice_vectors, cclass, slip_directions, slip_planes))
{
}

CrystalGeometry::CrystalGeometry(const OptionSet & options,
                                 const R2 & cclass,
                                 const Vec & lattice_vectors,
                                 std::tuple<Vec, Vec, Scalar, std::vector<Size>> slip_data)
  : Data(options),
    _sym_ops(cclass),
    _lattice_vectors(declare_buffer<Vec>("lattice_vectors", lattice_vectors)),
    _reciprocal_lattice_vectors(declare_buffer<Vec>("reciprocal_lattice_vectors",
                                                    make_reciprocal_lattice(_lattice_vectors))),
    _slip_directions(declare_buffer<MillerIndex>("slip_directions", "slip_directions")),
    _slip_planes(declare_buffer<MillerIndex>("slip_planes", "slip_planes")),
    _cartesian_slip_directions(
        declare_buffer<Vec>("cartesian_slip_directions", std::get<0>(slip_data))),
    _cartesian_slip_planes(declare_buffer<Vec>("cartesian_slip_planes", std::get<1>(slip_data))),
    _burgers(declare_buffer<Scalar>("burgers", std::get<2>(slip_data))),
    _slip_offsets(std::get<3>(slip_data)),
    _A(declare_buffer<R2>("schmid_tensors",
                          neml2::outer(neml2::unit(_cartesian_slip_directions),
                                       neml2::unit(_cartesian_slip_planes)))),
    _M(declare_buffer<SR2>("symmetric_schmid_tensors", SR2(_A))),
    _W(declare_buffer<WR2>("skew_symmetric_schmid_tensors", WR2(_A)))
{
}

Vec
CrystalGeometry::a1() const
{
  return _lattice_vectors.intmd_index({0});
}

Vec
CrystalGeometry::a2() const
{
  return _lattice_vectors.intmd_index({1});
}

Vec
CrystalGeometry::a3() const
{
  return _lattice_vectors.intmd_index({2});
}

Vec
CrystalGeometry::b1() const
{
  return _reciprocal_lattice_vectors.intmd_index({0});
}

Vec
CrystalGeometry::b2() const
{
  return _reciprocal_lattice_vectors.intmd_index({1});
}

Vec
CrystalGeometry::b3() const
{
  return _reciprocal_lattice_vectors.intmd_index({2});
}

Size
CrystalGeometry::nslip() const
{
  return _slip_offsets.back();
}

Size
CrystalGeometry::nslip_groups() const
{
  // NOLINTNEXTLINE(*-narrowing-conversions)
  return _slip_offsets.size() - 1;
}

Size
CrystalGeometry::nslip_in_group(Size i) const
{
  neml_assert_dbg(i < nslip_groups(), "Slip group index out of range");
  return _slip_offsets[i + 1] - _slip_offsets[i];
}

Vec
CrystalGeometry::make_reciprocal_lattice(const Vec & lattice_vectors)
{
  auto a1 = lattice_vectors.intmd_index({0});
  auto a2 = lattice_vectors.intmd_index({1});
  auto a3 = lattice_vectors.intmd_index({2});

  auto rl = intmd_stack({neml2::cross(a2, a3) / neml2::vdot(a1, neml2::cross(a2, a3)),
                         neml2::cross(a3, a1) / neml2::vdot(a2, neml2::cross(a3, a1)),
                         neml2::cross(a1, a2) / neml2::vdot(a3, neml2::cross(a1, a2))});

  return rl;
}

static Vec
miller_to_cartesian(const Vec & A, const MillerIndex & d)
{
  const auto dr = Vec(normalize_gcd(d)).to(get_default_dtype());
  return Vec::fill(neml2::vdot(A.intmd_index({0}), dr),
                   neml2::vdot(A.intmd_index({1}), dr),
                   neml2::vdot(A.intmd_index({2}), dr));
}

std::tuple<Vec, Vec, Scalar, std::vector<Size>>
CrystalGeometry::setup_schmid_tensors(const Vec & A,
                                      const R2 & cls,
                                      const MillerIndex & slip_directions,
                                      const MillerIndex & slip_planes)
{
  // We need the reciprocol lattice
  Vec B = make_reciprocal_lattice(A);

  // List of slip directions and planes needs to be consistent
  if (slip_directions.intmd_sizes() != slip_planes.intmd_sizes())
    neml_assert("Input slip directions and planes must have the same intermediate sizes");

  auto cmds = slip_directions;
  if (cmds.intmd_dim() == 0)
    cmds = cmds.intmd_unsqueeze(0);

  auto cmps = slip_planes;
  if (cmps.intmd_dim() == 0)
    cmps = cmps.intmd_unsqueeze(0);

  // Loop over each slip system
  std::vector<Vec> cartesian_slip_directions;
  std::vector<Vec> cartesian_slip_planes;
  std::vector<Scalar> burgers_vectors;
  std::vector<Size> offsets = {0};

  for (Size i = 0; i < cmds.intmd_size(-1); i++)
  {
    // Get the cartesian slip plane and direction
    auto cmd = cmds.intmd_index({i});
    auto cmp = cmps.intmd_index({i});

    // Get the families of symmetry-equivalent planes and directions
    auto direction_options = unique_bidirectional(cls, miller_to_cartesian(A, cmd));
    auto plane_options = unique_bidirectional(cls, miller_to_cartesian(B, cmp));

    // Accept the ones that are perpendicular
    // We could do this in a vectorized manner, but I don't think it's worth it as
    // this code only runs once
    Size last = offsets.back();
    for (Size j = 0; j < direction_options.intmd_size(-1); j++)
    {
      auto di = direction_options.intmd_index({j});
      auto dps = neml2::vdot(plane_options, di);
      auto inds = at::where(at::isclose(at::abs(dps), at::scalar_tensor(0.0, dps.dtype()))).front();
      // We could very easily vectorize this loop, but again whatever
      for (Size kk = 0; kk < inds.sizes()[0]; kk++)
      {
        Size k = inds.index({kk}).item<Size>();
        auto pi = plane_options.intmd_index({k});
        cartesian_slip_directions.push_back(neml2::unit(di));
        cartesian_slip_planes.push_back(neml2::unit(pi));
        burgers_vectors.push_back(neml2::norm(di));
        last += 1;
      }
    }
    offsets.push_back(last);
  }

  return std::make_tuple(intmd_stack(cartesian_slip_directions),
                         intmd_stack(cartesian_slip_planes),
                         intmd_stack(burgers_vectors),
                         offsets);
}

} // namespace neml2
