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
#include "neml2/base/Registry.h"
#include "neml2/base/Settings.h"
#include "neml2/base/OptionSet.h"

#include <argparse/argparse.hpp>

int
main(int argc, char * argv[])
{
  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  argparse::ArgumentParser program("syntax");
  program.add_description("Extract object syntax from the registry. By default outputs to stdout.");
  program.add_argument("--yaml")
      .help("redirect output to a YAML file")
      .default_value(std::string("syntax.yml"));

  // Force link dynamic libraries
  ::_neml2_force_link_models();
  ::_neml2_force_link_equation_systems();
  ::_neml2_force_link_solvers();
  ::_neml2_force_link_user_tensors();
  ::_neml2_force_link_drivers();

  // Parse cliargs
  program.parse_args(argc, argv);

  // Create the output stream
  std::ofstream ofs;
  std::ostream * out = &std::cout;

  if (program.is_used("--yaml"))
  {
    ofs.open(program.get<std::string>("yaml"));
    if (!ofs.is_open())
      throw std::runtime_error("Failed to open output file: " + program.get<std::string>("yaml"));
    out = &ofs;
  }

  auto settings = neml2::Settings::expected_options();
  *out << "neml2::Settings:\n";
  *out << settings << '\n';

  for (const auto & [type, info] : neml2::Registry::info())
  {
    *out << info.type_name << ":\n";
    auto options = info.expected_options;
    options.set<std::string>("type") = type;
    *out << options << '\n';
  }

  if (ofs.is_open())
    ofs.close();

  return 0;
}
