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

#include "neml2/jit/TraceableTensorShape.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/functions/fem.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/functions/bmm.h"

using namespace neml2;

TEST_CASE("fem", "[tensors][functions]")
{
  // Consider a 2D mesh with 4 elements, 9 nodes, and 2 displacement variables.
  // Assume 1st order quadrature where each element has 4 quadrature points.
  Size nelem = 4;
  Size ndofe = 4;
  Size nqp = 4;
  Size nvar = 2;
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
  // Converting dof map to a tensor:
  auto dof_map = Tensor::create({{{0, 3, 4, 1}, {9, 12, 13, 10}},
                                 {{1, 4, 5, 2}, {10, 13, 14, 11}},
                                 {{3, 6, 7, 4}, {12, 15, 16, 13}},
                                 {{4, 7, 8, 5}, {13, 16, 17, 14}}},
                                1,
                                kInt64)
                     .base_transpose(0, 1);
  // The dof map has shape (4; 4, 2)
  //                        |  |  |
  //                 Nelem__|  |  |
  //                    Ndofe__|  |
  //                        Nvar__|

  // The basis functions for the quadrature points are:
  const Real x1 = -std::sqrt(3.0) / 3;
  const Real x2 = std::sqrt(3.0) / 3;
  auto xi = Tensor::create({x1, x1, -x1, -x1});
  auto eta = Tensor::create({x2, -x2, x2, -x2});
  auto phi1 = (1 - xi) * (1 + eta) / 4;
  auto phi2 = (1 - xi) * (1 - eta) / 4;
  auto phi3 = (1 + xi) * (1 - eta) / 4;
  auto phi4 = (1 + xi) * (1 + eta) / 4;
  auto phi = base_stack({phi1, phi2, phi3, phi4}, 0);
  // The basis tensor has shape (; 4, 4)
  //                               |  |
  //                        Ndofe__|  |
  //                            Nqp___|
  // It's unbatched because it's the same for all elements.

  // The parametric basis function gradients for the quadrature points are:
  auto dphi1_dxi = -(1 + eta) / 4;
  auto dphi1_deta = (1 - xi) / 4;
  auto dphi2_dxi = -(1 - eta) / 4;
  auto dphi2_deta = -(1 - xi) / 4;
  auto dphi3_dxi = (1 - eta) / 4;
  auto dphi3_deta = -(1 + xi) / 4;
  auto dphi4_dxi = (1 + eta) / 4;
  auto dphi4_deta = (1 + xi) / 4;
  auto zero = Tensor::zeros({4});
  auto dphi1 = base_stack({dphi1_dxi, dphi1_deta, zero}, -1);
  auto dphi2 = base_stack({dphi2_dxi, dphi2_deta, zero}, -1);
  auto dphi3 = base_stack({dphi3_dxi, dphi3_deta, zero}, -1);
  auto dphi4 = base_stack({dphi4_dxi, dphi4_deta, zero}, -1);
  auto dphi = base_stack({dphi1, dphi2, dphi3, dphi4}, 0);
  // The basis gradient tensor has shape (; 4, 4, 3)
  //                                        |  |  |
  //                                 Ndofe__|  |  |
  //                                     Nqp___|  |
  //                                        Ndim__|
  // It's unbatched because it's the same for all elements (in the parametric space).

  // Isotroparametric mapping from the parametric space to the physical space:
  // This is too much for me to prototype, so let's assume the mapping is identity...
  auto dphi_phys = dphi.batch_expand(nelem).clone();
  // The physical basis gradient tensor has shape (4; 4, 4, 3)
  //                                               |  |  |  |
  //                                        Nelem__|  |  |  |
  //                                           Ndofe__|  |  |
  //                                               Nqp___|  |
  //                                                  Ndim__|

  // The solution vector is a 1D tensor with shape (; 18)
  auto sol = Tensor(at::linspace(0.0, 17.0, 18), 0);

  // First step is to scatter the solution vector to the dof map
  auto sol_scattered = fem_scatter(sol, dof_map);
  REQUIRE(sol_scattered.batch_sizes() == TensorShape{nelem});
  REQUIRE(sol_scattered.base_sizes() == TensorShape{ndofe, nvar});
  auto sol_scattered_correct = Tensor::create({{{0.0, 3.0, 4.0, 1.0}, {9.0, 12.0, 13.0, 10.0}},
                                               {{1.0, 4.0, 5.0, 2.0}, {10.0, 13.0, 14.0, 11.0}},
                                               {{3.0, 6.0, 7.0, 4.0}, {12.0, 15.0, 16.0, 13.0}},
                                               {{4.0, 7.0, 8.0, 5.0}, {13.0, 16.0, 17.0, 14.0}}},
                                              1)
                                   .base_transpose(0, 1);
  REQUIRE(sol_scattered.allclose(sol_scattered_correct));

  // The second step is to interpolate the scattered solution to the quadrature points
  // The interpolated solution will have shape (4; 2, 4)
  //                                            |  |  |
  //                                     Nelem__|  |  |
  //                                         Nvar__|  |
  //                                             Nqp__|
  auto sol_interp = fem_interpolate(sol_scattered, phi);
  REQUIRE(sol_interp.batch_sizes() == TensorShape{nelem});
  REQUIRE(sol_interp.base_sizes() == TensorShape{nvar, nqp});
  auto sol_interp_correct =
      Tensor::create({{{0.8453, 2.5774, 1.4226, 3.1547}, {9.8453, 11.5774, 10.4226, 12.1547}},
                      {{1.8453, 3.5774, 2.4226, 4.1547}, {10.8453, 12.5774, 11.4226, 13.1547}},
                      {{3.8453, 5.5774, 4.4226, 6.1547}, {12.8453, 14.5774, 13.4226, 15.1547}},
                      {{4.8453, 6.5774, 5.4226, 7.1547}, {13.8453, 15.5774, 14.4226, 16.1547}}},
                     1);
  REQUIRE(sol_interp.allclose(sol_interp_correct, 0, 1e-3));

  // The third step is to interpolate the scattered solution to get the variable gradient at the
  // quadrature points
  // The interpolated gradient will have shape (4; 2, 4, 3)
  //                                            |  |  |  |
  //                                     Nelem__|  |  |  |
  //                                         Nvar__|  |  |
  //                                            Nqp___|  |
  //                                               Ndim__|
  // Hint: Since I was lazy, I assumed the mapping is identity, so the gradient in the physical
  // space is the same as the gradient in the parametric space. Moreover, look at the way I defined
  // the nodal solution (the ASCII pictures above), the solution gradient is the same at every
  // quadrature point.
  auto sol_grad_interp = fem_interpolate(sol_scattered, dphi_phys);
  REQUIRE(sol_grad_interp.batch_sizes() == TensorShape{nelem});
  REQUIRE(sol_grad_interp.base_sizes() == TensorShape{nvar, nqp, 3});
  auto sol_grad_interp_correct = Tensor::create({0.5, -1.5, 0.0});
  REQUIRE(sol_grad_interp.allclose(sol_grad_interp_correct));

  // The fourth step is to calculate the strain tensor at the quadrature points
  // The strain tensor will have shape (4, 4; 6)
  //                                    |  |
  //                             Nelem__|  |
  //                                 Nqp___|
  auto grad_ux = Vec(sol_grad_interp.base_index({0}), 2);
  auto grad_uy = Vec(sol_grad_interp.base_index({1}), 2);
  auto grad_uz = Vec::zeros_like(grad_ux);
  auto grad_u = R2(base_stack({grad_ux, grad_uy, grad_uz}, 0));
  auto strain = SR2(grad_u);
  REQUIRE(strain.batch_sizes() == TensorShape{nelem, nqp});
  REQUIRE(strain.base_sizes() == TensorShape{6});

  // The fifth step is to perform the constitutive update to map from strain to stress
  // The stress tensor will have shape (4, 4; 6)
  //                                    |  |
  //                             Nelem__|  |
  //                                 Nqp___|
  auto & model =
      reload_model("models/solid_mechanics/elasticity/LinearIsotropicElasticity.i", "model");
  VariableName strain_name(STATE, "internal", "Ee");
  VariableName stress_name(STATE, "S");
  auto stress = model.value({{strain_name, strain}})[stress_name];
  REQUIRE(stress.batch_sizes() == TensorShape{nelem, nqp});
  REQUIRE(stress.base_sizes() == TensorShape{6});

  // The sixth step is to calulate the residual
  // Assume Galerkin, i.e., test and shape functions are the same.
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
  //   dphi_phys: (4; 4, 4, 3)
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
  dphi_phys = Tensor(dphi_phys.base_transpose(0, 1), 2);
  auto J = Scalar::full(1.0).batch_expand({nelem, nqp});
  auto W = Scalar::full(1.0).batch_expand({nqp});
  auto T = Scalar::full(1.0);
  auto re_qp = neml2::bmm(dphi_phys, R2(SR2(stress))) * J * W * T;
  auto re = batch_sum(re_qp, 1).base_index({indexing::Slice(), indexing::Slice(0, 2)});
  // The element residual has shape (4; 4, 2)
  //                                 |  |  |
  //                          Nelem__|  |  |
  //                             Ndofe__|  |
  //                                 Nvar__|
  REQUIRE(re.batch_sizes() == TensorShape{nelem});
  REQUIRE(re.base_sizes() == TensorShape{ndofe, nvar});

  // Final step, assemble the element residual into the global residual vector
  // The global residual vector has shape (18)
  auto r = fem_assemble(re, dof_map, ndof);
  REQUIRE(!r.batched());
  REQUIRE(r.base_sizes() == TensorShape{ndof});
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
