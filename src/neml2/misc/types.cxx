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

#include <limits>
#include <iostream>

#include "neml2/misc/types.h"
#include "neml2/misc/errors.h"

namespace neml2
{
CScalar
machine_precision(Dtype dtype)
{
  switch (dtype)
  {
    case kFloat64:
      return std::numeric_limits<double>::epsilon();
    case kFloat32:
      return std::numeric_limits<float>::epsilon();
    case kFloat16:
      // Note: C++ has no native float16 type prior to c++23; approximation needed
      return 0.00097656f; // ~2^-10
    default:
      throw NEMLException("Unsupported dtype for machine precision");
  }
}

std::ostream &
operator<<(std::ostream & os, FType f)
{
  if (f == FType::NONE)
    os << "NONE";
  else if (f == FType::INPUT)
    os << "INPUT";
  else if (f == FType::OUTPUT)
    os << "OUTPUT";
  else if (f == FType::PARAMETER)
    os << "PARAMETER";
  else if (f == FType::BUFFER)
    os << "BUFFER";
  else
    throw NEMLException("Unknown FType");

  return os;
}
} // namespace neml2
