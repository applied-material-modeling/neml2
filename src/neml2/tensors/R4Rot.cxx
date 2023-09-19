// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/tensors/R4Rot.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/R4.h"

namespace neml2
{

R4Rot
R4Rot::derivative(const Rot & r, const R4 & T)
{
  R2 R = r.to_R2();
  R3 F = r.dR2();

  return (einsum({R, R, R, T, F}, {"jn", "ko", "lp", "mnop", "imt"}, "ijklt") +
          einsum({R, R, R, T, F}, {"im", "ko", "lp", "mnop", "jnt"}, "ijklt") +
          einsum({R, R, R, T, F}, {"im", "jn", "lp", "mnop", "kot"}, "ijklt") +
          einsum({R, R, R, T, F}, {"im", "jn", "ko", "mnop", "lpt"}, "ijklt"));
}

} // namespace neml2
