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

  argparse::ArgumentParser program("neml2-syntax");
  program.add_description("Extract object syntax from the registry. By default outputs to stdout.");
  program.add_argument("--yaml")
      .help("redirect output to a YAML file")
      .default_value(std::string("syntax.yml"));
  program.add_argument("--section")
      .help("only emit objects whose input-file section matches (e.g. Models, Solvers, Drivers, "
            "Tensors, Schedulers, Data, EquationSystems, Settings)")
      .default_value(std::string(""));
  program.add_argument("--type")
      .help("only emit the object whose registered type matches (e.g. LinearDashpot); "
            "intended for drilling into a single object's full option list")
      .default_value(std::string(""));
  program.add_argument("--summary")
      .help("emit only the type, section, and doc string for each object (omit per-option detail)")
      .flag();

  // Force link dynamic libraries
  neml2::force_link_runtime();

  // Parse cliargs
  program.parse_args(argc, argv);

  // Create the output stream
  std::ofstream ofs;
  std::ostream * out = &std::cout;

  if (program.is_used("--yaml"))
  {
    ofs.open(program.get<std::string>("yaml"));
    if (!ofs.is_open())
    {
      std::cerr << "Failed to open output file: " + program.get<std::string>("yaml") << std::endl;
      return 1;
    }
    out = &ofs;
  }

  const auto section_filter = program.get<std::string>("--section");
  const auto type_filter = program.get<std::string>("--type");
  const bool summary = program.get<bool>("--summary");

  auto emit = [&](const neml2::OptionSet & opts)
  {
    if (!section_filter.empty() && opts.section() != section_filter)
      return;
    if (!type_filter.empty() && opts.type() != type_filter)
      return;
    if (summary)
    {
      *out << opts.type() << ":\n";
      *out << "  section: " << opts.section() << '\n';
      if (opts.doc().empty())
        *out << "  doc:\n";
      else
        *out << "  doc: |-\n    " << opts.doc() << '\n';
    }
    else
    {
      *out << opts << '\n';
    }
  };

  emit(neml2::Settings::expected_options());
  for (const auto & [type, info] : neml2::Registry::info())
    emit(info.expected_options);

  if (ofs.is_open())
    ofs.close();

  return 0;
}
