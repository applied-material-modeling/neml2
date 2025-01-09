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

#include "neml2/base/Registry.h"
#include "neml2/models/Data.h"

namespace neml2
{
class Scalar;
class Vec;
class R2;
class SR2;
class WR2;
class MillerIndex;

namespace crystallography
{

/// @brief Defines the geometry of a crystal system
/// This includes a basic definition of the crystal lattice,
/// via Bravais vectors and a CrystalClass object defining the
/// crystal symmetry as well as the definition of the geometry
/// of each slip system.
class CrystalGeometry : public Data
{
public:
  /// Input options
  static OptionSet expected_options();

  /// Setup from parameter set
  CrystalGeometry(const OptionSet & options);

  /// Alternate constructor not relying on options
  CrystalGeometry(const OptionSet & options, const R2 & cclass, const Vec & lattice_vectors);

  /// accessor for the first lattice vector
  Vec a1() const;
  /// accessor for the second lattice vector
  Vec a2() const;
  /// accessor for the third lattice vector
  Vec a3() const;

  /// accessor for the first reciprocal lattice vector
  Vec b1() const;
  /// accessor for the second reciprocal lattice vector
  Vec b2() const;
  /// accessor for the third reciprocal lattice vector
  Vec b3() const;

  /// Total number of slip systems
  Size nslip() const;
  /// Number of slip groups
  Size nslip_groups() const;
  /// Number of slip systems in a given group
  Size nslip_in_group(Size i) const;

  /// Accessor for the slip directions
  const Vec & cartesian_slip_directions() const { return _cartesian_slip_directions; };
  /// Accessor for the slip planes
  const Vec & cartesian_slip_planes() const { return _cartesian_slip_planes; };
  /// Accessor for the burgers vector
  const Scalar & burgers() const { return _burgers; };

  /// Accessor for the full Schmid tensors
  const R2 & A() const { return _A; };
  /// Accessor for the symmetric Schmid tensors
  const SR2 & M() const { return _M; };
  /// Accessor for the skew-symmetric Schmid tensors
  const WR2 & W() const { return _W; };

  /// Accessor for the crystal class symmetry operators
  const R2 & symmetry_operators() const { return _sym_ops; };

  /// Slice a Tensor to provide only the batch associated with a slip system
  // The slice happens along the last batch axis
  template <class Derived,
            typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
  Derived slip_slice(const Derived & tensor, Size grp) const;

private:
  /// Delegated constructor to setup schmid tensors and slice indices at once
  CrystalGeometry(const OptionSet & options,
                  const R2 & cclass,
                  const Vec & lattice_vectors,
                  std::tuple<Vec, Vec, Scalar, std::vector<Size>> slip_data);

  /// Helper to setup reciprocal lattice
  static Vec make_reciprocal_lattice(const Vec & lattice_vectors);

  /// Helper to setup slip systems
  static std::tuple<Vec, Vec, Scalar, std::vector<Size>>
  setup_schmid_tensors(const Vec & A,
                       const R2 & cls,
                       const MillerIndex & slip_directions,
                       const MillerIndex & slip_planes);

  /// Helper to return Cartesian vector given miller indices
  static Vec miller_to_cartesian(const Vec & A, const MillerIndex & d);

private:
  /// Crystal symmetry class with operators
  const R2 & _sym_ops;
  /// Lattice vectors
  const Vec & _lattice_vectors;
  /// Reciprocal lattice vectors
  const Vec & _reciprocal_lattice_vectors;
  /// Crystallographic slip directions
  const MillerIndex & _slip_directions;
  /// Crystallographic slip planes
  const MillerIndex & _slip_planes;

  /// Cartesian slip directions
  const Vec & _cartesian_slip_directions;
  /// Cartesian slip planes
  const Vec & _cartesian_slip_planes;
  /// Burgers vector lengths
  const Scalar & _burgers;
  /// Offsets into batch tensors for each slip group
  const std::vector<Size> _slip_offsets;

  /// Output: full Schmid tensor for each slip system
  const R2 & _A;
  /// Output: symmetric Schmid tensors for each slip system
  const SR2 & _M;
  /// Output: skew-symmetry Schmid tensors for each slip system
  const WR2 & _W;
};

template <class Derived, typename>
Derived
CrystalGeometry::slip_slice(const Derived & tensor, Size grp) const
{
  if (grp >= nslip_groups())
    throw NEMLException("Invalid slip group index");
  return tensor.batch_index(
      {indexing::Ellipsis, indexing::Slice(_slip_offsets[grp], _slip_offsets[grp + 1])});
}

} // namespace crystallography
} // namespace neml2
