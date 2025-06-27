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
  : CrystalGeometry(options, options.get<Factory *>("factory"))
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
                          (_cartesian_slip_directions / _cartesian_slip_directions.norm())
                              .outer(_cartesian_slip_planes / _cartesian_slip_planes.norm()))),
    _M(declare_buffer<SR2>("symmetric_schmid_tensors", SR2(_A))),
    _W(declare_buffer<WR2>("skew_symmetric_schmid_tensors", WR2(_A)))
{
}

Vec
CrystalGeometry::a1() const
{
  return _lattice_vectors.batch_index({0});
}

Vec
CrystalGeometry::a2() const
{
  return _lattice_vectors.batch_index({1});
}

Vec
CrystalGeometry::a3() const
{
  return _lattice_vectors.batch_index({2});
}

Vec
CrystalGeometry::b1() const
{
  return _reciprocal_lattice_vectors.batch_index({0});
}

Vec
CrystalGeometry::b2() const
{
  return _reciprocal_lattice_vectors.batch_index({1});
}

Vec
CrystalGeometry::b3() const
{
  return _reciprocal_lattice_vectors.batch_index({2});
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
  neml_assert_dbg(i < nslip_groups());
  return _slip_offsets[i + 1] - _slip_offsets[i];
}

Vec
CrystalGeometry::make_reciprocal_lattice(const Vec & lattice_vectors)
{
  auto a1 = lattice_vectors.batch_index({0});
  auto a2 = lattice_vectors.batch_index({1});
  auto a3 = lattice_vectors.batch_index({2});

  Vec rl = Vec(at::stack({a2.cross(a3) / a1.dot(a2.cross(a3)),
                          a3.cross(a1) / a2.dot(a3.cross(a1)),
                          a1.cross(a2) / a3.dot(a1.cross(a2))}));

  return rl;
}

Vec
CrystalGeometry::miller_to_cartesian(const Vec & A, const MillerIndex & d)
{
  // Take advantage that a collection of 3 vectors is a R2
  return R2(ATensor(A)) * d.reduce().to_vec();
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
  if (slip_directions.batch_sizes() != slip_planes.batch_sizes())
    neml_assert("Input slip directions and planes must have the same batch sizes");

  auto bshape = slip_planes.batch_sizes().concrete();
  auto nbatch = slip_planes.batch_dim();

  // Loop over each slip system
  std::vector<ATensor> cartesian_slip_directions;
  std::vector<ATensor> cartesian_slip_planes;
  std::vector<ATensor> burgers_vectors;
  std::vector<Size> offsets = {0};

  for (Size i = 0; i < bshape[nbatch - 1]; i++)
  {
    // Get the cartesian slip plane and direction
    auto cmd = slip_directions.batch_index({indexing::Ellipsis, i});
    auto cmp = slip_planes.batch_index({indexing::Ellipsis, i});

    // Get the families of symmetry-equivalent planes and directions
    auto direction_options = unique_bidirectional(cls, miller_to_cartesian(A, cmd));
    auto plane_options = unique_bidirectional(cls, miller_to_cartesian(B, cmp));

    // Accept the ones that are perpendicular
    // We could do this in a vectorized manner, but I don't think it's worth it as
    // this code only runs once
    Size last = offsets.back();
    for (Size j = 0; j < direction_options.batch_size(-1).concrete(); j++)
    {
      auto di = direction_options.batch_index({indexing::Ellipsis, j});
      auto dps = plane_options.dot(di);
      auto inds = at::where(at::isclose(at::abs(dps), at::scalar_tensor(0.0, dps.dtype()))).front();
      // We could very easily vectorize this loop, but again whatever
      for (Size kk = 0; kk < inds.sizes()[0]; kk++)
      {
        Size k = inds.index({kk}).item<Size>();
        auto pi = plane_options.batch_index({indexing::Ellipsis, k});
        cartesian_slip_directions.push_back(di / di.norm());
        cartesian_slip_planes.push_back(pi / pi.norm());
        burgers_vectors.push_back(di.norm());
        last += 1;
      }
    }
    offsets.push_back(last);
  }

  return std::make_tuple(Vec(at::stack(cartesian_slip_directions)),
                         Vec(at::stack(cartesian_slip_planes)),
                         Scalar(at::stack(burgers_vectors)),
                         offsets);
}

} // namespace neml2
