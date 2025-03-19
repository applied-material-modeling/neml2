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

#include <catch2/catch_session.hpp>

#include <filesystem>
#include "utils.h"

int
main(int argc, char * argv[])
{
  Catch::Session session;

  // Add path cli arg
  using namespace Catch::Clara;
  std::string working_dir;
  auto cli = session.cli() | Opt(working_dir, ".")["-p"]["--path"]("path to the test input files");
  session.cli(cli);

  // Let Catch2 parse the command line
  auto err = session.applyCommandLine(argc, argv);
  if (err)
    return err;

  // Set the working directory
  auto exec_prefix = std::filesystem::path(argv[0]).parent_path();
  err = guess_test_dir("unit", working_dir, exec_prefix);
  if (err)
    return err;
  std::cout << "Working directory: " << working_dir << std::endl;
  std::filesystem::current_path(working_dir);

  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  return session.run();
}
