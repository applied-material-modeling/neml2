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

// Minimal downstream consumer of an installed NEML2: include a shipped public
// header and call an exported (non-inline) symbol so the link against
// libneml2.so is real. This is built twice in CI -- once via find_package(neml2)
// and once via pkg-config -- to prove a published wheel/install carries usable
// headers + the CMake package config + the pkg-config files.

#include <neml2/csrc/aoti/Exception.h>

#include <iostream>

int
main()
{
  const neml2::aoti::FatalError fatal("downstream link check");
  const neml2::aoti::ConvergenceError recoverable("downstream link check");

  // recoverable() is the exported virtual that lives in libneml2 -- calling it
  // forces the link to resolve against the shared library, not just the headers.
  if (fatal.recoverable() || !recoverable.recoverable())
  {
    std::cerr << "neml2 downstream: unexpected recoverable() semantics\n";
    return 1;
  }

  std::cout << "neml2 downstream consumer OK\n";
  return 0;
}
