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

#include "neml2/tensors/Tensor.h"
#include "neml2/base/OptionSet.h"

namespace neml2
{
/**
 * @brief Definition of a nonlinear system of equations.
 *
 */
class NonlinearSystem
{
public:
  /**
   * Convenience struct to hold residual to prevent developers from accidentally confusing the
   * scaled and unscaled residual
   */
  template <bool scaled>
  struct RES : public Tensor
  {
    RES() = default;

    /// Conversion from Tensor must be explicit
    explicit RES(const Tensor & r)
      : Tensor(r)
    {
    }

    /// Conversion between scaled and unscaled must be explicit
    explicit RES(const RES<!scaled> & r)
      : Tensor(r)
    {
    }
  };

  /**
   * Convenience struct to hold Jacobian to prevent developers from accidentally confusing the
   * scaled and unscaled residual
   */
  template <bool scaled>
  struct JAC : public Tensor
  {
    JAC() = default;

    /// Conversion from Tensor must be explicit
    explicit JAC(const Tensor & J)
      : Tensor(J)
    {
    }

    /// Conversion between scaled and unscaled must be explicit
    explicit JAC(const JAC<!scaled> & J)
      : Tensor(J)
    {
    }
  };

  /**
   * Convenience struct to hold solution to prevent developers from accidentally confusing the
   * scaled and unscaled solution (or solution increment, search direction, etc.)
   */
  template <bool scaled>
  struct SOL : public Tensor
  {
    SOL() = default;

    /// Conversion from Tensor must be explicit
    explicit SOL(const Tensor & u)
      : Tensor(u)
    {
    }

    /// Conversion between scaled and unscaled must be explicit
    explicit SOL(const SOL<!scaled> & u)
      : Tensor(u)
    {
    }
  };

public:
  static OptionSet expected_options();

  static void disable_automatic_scaling(OptionSet & options);

  static void enable_automatic_scaling(OptionSet & options);

  NonlinearSystem(const OptionSet & options);

  /**
   * @brief Compute algebraic Jacobian-based automatic scaling following
   * https://cs.stanford.edu/people/paulliu/files/cs517-project.pdf
   *
   * In a nutshell, given the original linearized system
   *
   * \f$ \mathrm{J} \Delta \mathrm{x} = -\mathrm{r} \f$
   *
   * Instead of solving for \f$ \Delta \mathrm{x} \f$ directly, we solve for a scaled version of it:
   *
   * \f$ \mathrm{J} \mathrm{C} \Delta \mathrm{x}' = -\mathrm{r} \f$
   *
   * where \f$ \mathrm{C} \f$ is a diagonal matrix, and apparently \f$ \Delta \mathrm{x} =
   * \mathrm{C} \Delta \mathrm{x}' \f$. Then, left-multiply both sides by another diagonal matrix
   * \f$ \mathrm{R} \f$, we get
   *
   * \f$ \mathrm{R} \mathrm{J} \mathrm{C} \Delta \mathrm{x}' = -\mathrm{R} \mathrm{r} \f$
   *
   * which is equivalent to
   *
   * \f$ \mathrm{J}' \Delta \mathrm{x}' = -\mathrm{r}' \f$
   *
   * where \f$ \mathrm{J}' = \mathrm{R} \mathrm{J} \mathrm{C} \f$ is the scaled Jacobian, and \f$
   * \mathrm{r}' = \mathrm{R} \mathrm{r} \f$ is the scaled residual. The goal of automatic scaling
   * is to find the scaling matrices so that max-norm of the rows and columns of the scaled Jacobian
   * is as close to 1 as possible.
   *
   * @param x Unscaled initial guess used to compute the initial unscaled residual and Jacobian
   * @param verbose Print automatic scaling convergence information
   */
  virtual void init_scaling(const SOL<false> & x, const bool verbose = false);

  /// Apply scaling to the residual
  RES<true> scale(const RES<false> & r) const;
  /// Remove scaling to the residual
  RES<false> unscale(const RES<true> & r) const;
  /// Apply scaling to the Jacobian
  JAC<true> scale(const JAC<false> & J) const;
  /// Remove scaling to the Jacobian
  JAC<false> unscale(const JAC<true> & J) const;
  /// Apply scaling to the solution
  SOL<true> scale(const SOL<false> & u) const;
  /// Remove scaling to the solution
  SOL<false> unscale(const SOL<true> & u) const;

  /// Set the current guess
  virtual void set_guess(const SOL<true> & x) final;
  /// Set the _unscaled_ current guess
  virtual void set_guess(const SOL<false> & x) = 0;
  /// Assemble and return the residual
  template <bool scaled>
  RES<scaled> residual();
  /// Convenient shortcut to set the current guess, assemble and return the residual
  template <bool scaled>
  RES<scaled> residual(const SOL<scaled> & x);
  /// Assemble and return the Jacobian
  template <bool scaled>
  JAC<scaled> Jacobian();
  /// Convenient shortcut to set the current guess, assemble and return the Jacobian
  template <bool scaled>
  JAC<scaled> Jacobian(const SOL<scaled> & x);
  /// Assemble and return the residual and Jacobian
  template <bool scaled>
  std::tuple<RES<scaled>, JAC<scaled>> residual_and_Jacobian();
  /// Convenient shortcut to set the current guess, assemble and return the residual and Jacobian
  template <bool scaled>
  std::tuple<RES<scaled>, JAC<scaled>> residual_and_Jacobian(const SOL<scaled> & x);

protected:
  /**
   * @brief Compute the _unscaled_ residual and Jacobian
   *
   * @param r Pointer to the residual vector -- nullptr if not requested
   * @param J Pointer to the Jacobian matrix -- nullptr if not requested
   */
  virtual void assemble(RES<false> * r, JAC<false> * J) = 0;

  /// If true, do automatic scaling
  const bool _autoscale;

  /// Tolerance for convergence check of the iterative automatic scaling algorithm
  const Real _autoscale_tol;

  /// Maximum number of iterations allowed for the iterative automatic scaling algorithm
  const unsigned int _autoscale_miter;

  /// Flag to indicate whether scaling matrices have been computed
  bool _scaling_matrices_initialized;

  /// Row scaling "matrix" -- since it's a batched diagonal matrix, we are only storing its diagonals
  Tensor _row_scaling;

  /// Column scaling "matrix" -- since it's a batched diagonal matrix, we are only storing its diagonals
  Tensor _col_scaling;

private:
  void ensure_scaling_matrices_initialized_dbg() const;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

template <bool scaled>
NonlinearSystem::RES<scaled>
NonlinearSystem::residual()
{
  RES<false> r;
  assemble(&r, nullptr);
  if constexpr (scaled)
    return scale(r);
  else
    return r;
}

template <bool scaled>
NonlinearSystem::RES<scaled>
NonlinearSystem::residual(const NonlinearSystem::SOL<scaled> & x)
{
  set_guess(x);
  return residual<scaled>();
}

template <bool scaled>
NonlinearSystem::JAC<scaled>
NonlinearSystem::Jacobian()
{
  JAC<false> J;
  assemble(nullptr, &J);
  if constexpr (scaled)
    return scale(J);
  else
    return J;
}

template <bool scaled>
NonlinearSystem::JAC<scaled>
NonlinearSystem::Jacobian(const NonlinearSystem::SOL<scaled> & x)
{
  set_guess(x);
  return Jacobian<scaled>();
}

template <bool scaled>
std::tuple<NonlinearSystem::RES<scaled>, NonlinearSystem::JAC<scaled>>
NonlinearSystem::residual_and_Jacobian()
{
  RES<false> r;
  JAC<false> J;
  assemble(&r, &J);
  if constexpr (scaled)
    return {scale(r), scale(J)};
  else
    return {r, J};
}

template <bool scaled>
std::tuple<NonlinearSystem::RES<scaled>, NonlinearSystem::JAC<scaled>>
NonlinearSystem::residual_and_Jacobian(const NonlinearSystem::SOL<scaled> & x)
{
  set_guess(x);
  return residual_and_Jacobian<scaled>();
}
} // namespace neml2
