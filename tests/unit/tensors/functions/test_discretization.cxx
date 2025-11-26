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

#include <catch2/catch_test_macros.hpp>

#include "neml2/tensors/TraceableTensorShape.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/functions/discretization/assemble.h"
#include "neml2/tensors/functions/discretization/interpolate.h"
#include "neml2/tensors/functions/discretization/scatter.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/functions/mm.h"

using namespace neml2;

TEST_CASE("discretization", "[tensors][functions]")
{
  // Consider a 2D mesh with 4 elements, 9 nodes, and 2 displacement variables.
  // Assume 1st order quadrature where each element has 4 quadrature points.
  Size nelem = 4;
  Size ndofe = 4;
  Size nqp = 4;
  Size ndof = 18;

  // The mesh connectivity looks like:
  //    0---1---2
  //    | 0 | 1 |
  //    3---4---5
  //    | 2 | 3 |
  //    6---7---8
  //
  // Assuming 1st order Lagrange basis, the dof map looks like:
  //
  //  u_x
  //    0---1---2
  //    | 0 | 1 |
  //    3---4---5
  //    | 2 | 3 |
  //    6---7---8
  //
  //  u_y
  //    9--10--11
  //    | 0 | 1 |
  //   12--13--14
  //    | 2 | 3 |
  //   15--16--17
  //
  // Create dof maps:
  // The dof maps have shape (4, 4;)
  //                          |  |
  //                   Nelem__|  |
  //                      Ndofe__|
  auto dof_map_ux =
      Tensor::create({{0, 3, 4, 1}, {1, 4, 5, 2}, {3, 6, 7, 4}, {4, 7, 8, 5}}, 2, 0, kInt64);
  auto dof_map_uy = Tensor::create(
      {{9, 12, 13, 10}, {10, 13, 14, 11}, {12, 15, 16, 13}, {13, 16, 17, 14}}, 2, 0, kInt64);

  // We will be using 1st order Lagrange basis functions.
  // The basis tensor has shape (4, 4;)
  //                             |  |
  //                      Ndofe__|  |
  //                          Nqp___|
  // The basis functions are the same for all elements.
  const double x1 = -std::sqrt(3.0) / 3;
  const double x2 = std::sqrt(3.0) / 3;
  auto xi = Tensor::create({x1, x1, -x1, -x1}, 1);
  auto eta = Tensor::create({x2, -x2, x2, -x2}, 1);
  auto phi1 = (1 - xi) * (1 + eta) / 4;
  auto phi2 = (1 - xi) * (1 - eta) / 4;
  auto phi3 = (1 + xi) * (1 - eta) / 4;
  auto phi4 = (1 + xi) * (1 + eta) / 4;
  auto phi = dynamic_stack({phi1, phi2, phi3, phi4}).dynamic_expand({nelem, ndofe, nqp});

  // Create the parametric basis function gradients.
  // The basis gradient tensor has shape (4, 4; 3)
  //                                      |  |  |
  //                               Ndofe__|  |  |
  //                                   Nqp___|  |
  //                                      Ndim__|
  // It's unbatched because it's the same for all elements (in the parametric space).
  auto dphi1_dxi = -(1 + eta) / 4;
  auto dphi1_deta = (1 - xi) / 4;
  auto dphi2_dxi = -(1 - eta) / 4;
  auto dphi2_deta = -(1 - xi) / 4;
  auto dphi3_dxi = (1 - eta) / 4;
  auto dphi3_deta = -(1 + xi) / 4;
  auto dphi4_dxi = (1 + eta) / 4;
  auto dphi4_deta = (1 + xi) / 4;
  auto zero = Tensor::zeros({4});
  auto dphi1 = base_stack({dphi1_dxi, dphi1_deta, zero});
  auto dphi2 = base_stack({dphi2_dxi, dphi2_deta, zero});
  auto dphi3 = base_stack({dphi3_dxi, dphi3_deta, zero});
  auto dphi4 = base_stack({dphi4_dxi, dphi4_deta, zero});
  auto dphi = dynamic_stack({dphi1, dphi2, dphi3, dphi4});

  // Isotroparametric mapping from the parametric space to the physical space:
  //
  // This is too much for me to prototype, so let's assume the mapping is identity...
  // In reality, any finite element code could do this part.
  //
  // The physical basis gradient tensor has shape (4, 4, 4; 3)
  //                                               |  |  |  |
  //                                        Nelem__|  |  |  |
  //                                           Ndofe__|  |  |
  //                                               Nqp___|  |
  //                                                  Ndim__|
  auto dphi_phys = dphi.dynamic_expand({nelem, ndofe, nqp}).clone();

  // The solution vector is a 1D tensor with shape (18)
  auto sol = at::linspace(0.0, 17.0, 18);

  // Step 1: Scatter the solution vector to the dof map
  auto elem_dof_ux = discretization::scatter(sol, dof_map_ux);
  auto elem_dof_uy = discretization::scatter(sol, dof_map_uy);
  REQUIRE(elem_dof_ux.dynamic_sizes() == dof_map_ux.dynamic_sizes());
  REQUIRE(elem_dof_ux.base_sizes() == dof_map_ux.base_sizes());
  REQUIRE(elem_dof_uy.dynamic_sizes() == dof_map_uy.dynamic_sizes());
  REQUIRE(elem_dof_uy.base_sizes() == dof_map_uy.base_sizes());

  // Step 2: Interpolate the scattered solution to the quadrature points
  // The interpolated solution will have shape (4, 4;)
  //                                            |  |
  //                                     Nelem__|  |
  //                                          Nqp__|
  auto ux_h = discretization::interpolate(elem_dof_ux, phi);
  auto uy_h = discretization::interpolate(elem_dof_uy, phi);
  REQUIRE(ux_h.dynamic_sizes() == TensorShape{nelem, nqp});
  REQUIRE(uy_h.dynamic_sizes() == TensorShape{nelem, nqp});

  // Step 3: Interpolate the scattered solution to get the variable gradient at the
  // quadrature points
  // The interpolated gradient will have shape (4, 4; 3)
  //                                            |  |  |
  //                                     Nelem__|  |  |
  //                                         Nqp___|  |
  //                                            Ndim__|
  // Hint: Since I was lazy, I assumed the mapping is identity, so the gradient in the physical
  // space is the same as the gradient in the parametric space. Moreover, look at the way I defined
  // the nodal solution (the ASCII pictures above), the solution gradient is the same at every
  // quadrature point.
  auto grad_ux_h = discretization::interpolate(elem_dof_ux, dphi_phys);
  auto grad_uy_h = discretization::interpolate(elem_dof_uy, dphi_phys);
  REQUIRE(grad_ux_h.dynamic_sizes() == TensorShape{nelem, nqp});
  REQUIRE(grad_ux_h.base_sizes() == TensorShape{3});
  REQUIRE(grad_uy_h.dynamic_sizes() == TensorShape{nelem, nqp});
  REQUIRE(grad_uy_h.base_sizes() == TensorShape{3});

  // Step 4: Calculate the strain tensor at the quadrature points
  // The strain tensor will have shape (4, 4; 6)
  //                                    |  |
  //                             Nelem__|  |
  //                                 Nqp___|
  auto grad_uz_h = Tensor::zeros_like(grad_ux_h);
  auto grad_u = R2(base_stack({grad_ux_h, grad_uy_h, grad_uz_h}));
  auto strain = SR2(grad_u);
  REQUIRE(strain.dynamic_sizes() == TensorShape{nelem, nqp});
  REQUIRE(strain.base_sizes() == TensorShape{6});

  // Step 5: Perform the constitutive update to map from strain to stress
  // The stress tensor will have shape (4, 4; 6)
  //                                    |  |
  //                             Nelem__|  |
  //                                 Nqp___|
  auto model = load_model("models/solid_mechanics/elasticity/LinearIsotropicElasticity.i", "model");
  VariableName strain_name(STATE, "internal", "Ee");
  VariableName stress_name(STATE, "S");
  auto stress = model->value({{strain_name, strain}})[stress_name];
  stress = R2(SR2(stress)); // Convert from symmetric to full R2
  REQUIRE(stress.dynamic_sizes() == TensorShape{nelem, nqp});
  REQUIRE(stress.base_sizes() == TensorShape{3, 3});

  // Step 6: Calculate the residual
  // Assuming Galerkin, i.e., test and shape functions are the same.
  // The element residual is then
  //   r_i = \sum_q \phi_{,j} \sigma_{ij} J_q W_q T_q
  // where
  //   \phi_{,j} is the gradient of the shape function,
  //   \sigma_{ij} is the stress tensor,
  //   J_q is the determinant of the Jacobian at quadrature point q,
  //   W_q is the quadrature weight.
  //   T_q is the coordinate transformation factor.
  //
  // I'm assuming an identity mapping, so the Jacobian is 1.
  // For cartesian, the coordinate transformation is 1.
  // With 1st order Gauss quadrature, the weights are [1, 1, 1, 1] for the 4 qps.
  //
  // Summary of tensor shapes:
  //   dphi_phys: (4, 4, 4; 3)
  //               |  |  |  |
  //        Nelem__|  |  |  |
  //           Ndofe__|  |  |
  //               Nqp___|  |
  //                  Ndim__|
  //
  //   stress:    (4, 4; 6)
  //               |  |
  //        Nelem__|  |
  //            Nqp___|
  //
  //   J:         (4, 4;)
  //               |  |
  //        Nelem__|  |
  //            Nqp___|
  //
  //   W:         (4;)
  //               |
  //          Nqp__|
  //
  //   T:         ()
  dphi_phys = dphi_phys.dynamic_transpose(0, 1);
  auto J = Scalar::full(1.0).dynamic_expand({nelem, nqp});
  auto W = Scalar::full(1.0).dynamic_expand({nqp});
  auto T = Scalar::full(1.0);
  auto re_qp = base_sum(dphi_phys.base_unsqueeze(0) * stress, 1) * (J * W * T);
  auto re = dynamic_sum(re_qp, -1).dynamic_transpose(0, 1);
  // The element residual has shape (4, 4; 3)
  //                                 |  |  |
  //                          Nelem__|  |  |
  //                             Ndofe__|  |
  //                                 Ndim__|
  REQUIRE(re.dynamic_sizes() == TensorShape{nelem, ndofe});
  REQUIRE(re.base_sizes() == TensorShape{3});

  // Final step, assemble the element residual into the global residual vector
  // The global residual vector has shape (18)
  auto r = at::zeros({ndof});
  discretization::assemble_(r, re.base_index({0}), dof_map_ux);
  discretization::assemble_(r, re.base_index({1}), dof_map_uy);
  auto r_correct = Tensor::create({-19.230769,
                                   -76.923077,
                                   -57.692308,
                                   38.461538,
                                   0.0,
                                   -38.461538,
                                   57.692308,
                                   76.923077,
                                   19.230769,
                                   -134.61538,
                                   -346.15385,
                                   -211.53846,
                                   76.923077,
                                   0.0,
                                   -76.923077,
                                   211.53846,
                                   346.15385,
                                   134.61538});
  REQUIRE(r.allclose(r_correct, 0, 1e-4));
}
