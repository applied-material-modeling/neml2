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

#include "neml2/tensors/functions/linalg/dsptrf.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/outer.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/R4.h"

namespace neml2::linalg
{
SSR4
dsptrf(const Vec & evals, const R2 & evecs, const Vec & f, const Vec & df)
{
  // I'm not using `auto` here on purpose to make sure the tensor symmetries are correct.

  // Helper lambda to handle the degenerate cases
  auto theta = [&](Size i, Size j) -> Scalar
  {
    return neml2::where(
        evals(i) != evals(j), 0.5 * (f(i) - f(j)) / (evals(i) - evals(j)), 0.25 * (df(i) + df(j)));
  };

  // Eigenvectors
  const Vec v1 = evecs.col(0);
  const Vec v2 = evecs.col(1);
  const Vec v3 = evecs.col(2);

  // Some useful tensor products
  const SR2 M_11(neml2::outer(v1));
  const SR2 M_22(neml2::outer(v2));
  const SR2 M_33(neml2::outer(v3));
  const R2 M_12 = neml2::outer(v1, v2);
  const R2 M_13 = neml2::outer(v1, v3);
  const R2 M_23 = neml2::outer(v2, v3);

  // Derivative contribution from eigenvalues
  const SSR4 P_1 =
      df(0) * neml2::outer(M_11) + df(1) * neml2::outer(M_22) + df(2) * neml2::outer(M_33);

  // Derivative contribution from eigenvectors
  // Note the symmetrization is handled by the constructor of SSR4(R4)
  const Scalar theta_12 = theta(0, 1);
  const Scalar theta_23 = theta(1, 2);
  const Scalar theta_13 = theta(0, 2);
  const SSR4 P_2 = theta_12 * SSR4(neml2::outer(M_12)) + theta_13 * SSR4(neml2::outer(M_13)) +
                   theta_23 * SSR4(neml2::outer(M_23));

  return P_1 + 4 * P_2;
}
} // namespace neml2::linalg
