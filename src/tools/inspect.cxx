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

#include "neml2/neml2.h"
#include "neml2/models/Model.h"

#include <argparse/argparse.hpp>
#include "utils.h"

int
main(int argc, char * argv[])
{
  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  argparse::ArgumentParser program("inspect");
  program.add_description("Summarize the structure of a model.");
  program.add_argument("input").help("path to the input file");
  program.add_argument("model").help("name of the model in the input file to inspect");
  program.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  try
  {
    // Parse cliargs
    program.parse_args(argc, argv);

    const auto input = program.get<std::string>("input");
    const auto additional_cliargs = get_additional_cliargs(program);
    auto factory = neml2::load_input(input, additional_cliargs);
    const auto modelname = program.get<std::string>("model");
    auto model = factory->get_model(modelname);
    std::cout << *model << std::endl;
  }
  catch (const std::exception & err)
  {
    std::cerr << err.what() << std::endl;
    std::exit(1);
  }

  return 0;
}
