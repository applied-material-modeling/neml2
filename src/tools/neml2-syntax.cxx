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

#ifdef NEML2_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

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
  program.add_argument("--server")
      .help("run as a long-lived JSON server on stdin/stdout (requires NEML2_JSON)")
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
  const bool server_mode = program.get<bool>("--server");

  if (server_mode && (program.is_used("--yaml") || program.is_used("--section") ||
                      program.is_used("--type") || summary))
  {
    std::cerr << "error: --server is incompatible with --yaml, --section, --type, and --summary\n";
    return 1;
  }

#ifdef NEML2_JSON
  if (server_mode)
  {
    // Collect all OptionSets once; build pointer list after vector is fully populated
    std::vector<neml2::OptionSet> all_opts_storage;
    all_opts_storage.push_back(neml2::Settings::expected_options());
    for (const auto & [type, info] : neml2::Registry::info())
      all_opts_storage.push_back(info.expected_options);
    std::vector<const neml2::OptionSet *> all_opts;
    for (const auto & opts : all_opts_storage)
      all_opts.push_back(&opts);

    auto ftype_str = [](neml2::FType f) -> std::string
    {
      if (f == neml2::FType::INPUT)
        return "INPUT";
      if (f == neml2::FType::OUTPUT)
        return "OUTPUT";
      if (f == neml2::FType::PARAMETER)
        return "PARAMETER";
      if (f == neml2::FType::BUFFER)
        return "BUFFER";
      return "NONE";
    };

    auto opts_to_json = [&](const neml2::OptionSet & opts) -> json
    {
      json j;
      j["type"] = opts.type();
      j["section"] = opts.section();
      j["doc"] = opts.doc();
      json options = json::array();
      for (const auto & [name, opt] : opts)
      {
        if (opt->suppressed())
          continue;
        json o;
        o["name"] = name;
        o["doc"] = opt->doc();
        o["ftype"] = ftype_str(opt->ftype());
        o["required"] = opt->required();
        o["type"] = neml2::user_readable_type(opt->type());
        options.push_back(std::move(o));
      }
      j["options"] = std::move(options);
      return j;
    };

    std::string line;
    while (std::getline(std::cin, line))
    {
      if (line.empty())
        continue;
      json req;
      try
      {
        req = json::parse(line);
      }
      catch (...)
      {
        std::cout << json{{"id", nullptr}, {"error", "parse error"}}.dump() << "\n";
        std::cout.flush();
        continue;
      }

      const auto id = req.value("id", json{});
      const std::string method = req.value("method", "");
      json result;

      if (method == "list_sections")
      {
        std::set<std::string> sections;
        for (const auto * opts : all_opts)
          if (!opts->section().empty())
            sections.insert(opts->section());
        result = json::array();
        for (const auto & s : sections)
          result.push_back(s);
      }
      else if (method == "list_types")
      {
        const std::string section = req.value("section", "");
        result = json::array();
        for (const auto * opts : all_opts)
        {
          if (!section.empty() && opts->section() != section)
            continue;
          if (opts->type().empty())
            continue;
          result.push_back(
              {{"type", opts->type()}, {"section", opts->section()}, {"doc", opts->doc()}});
        }
      }
      else if (method == "get_options")
      {
        const std::string type = req.value("type", "");
        result = nullptr;
        for (const auto * opts : all_opts)
        {
          if (opts->type() == type)
          {
            result = opts_to_json(*opts);
            break;
          }
        }
      }
      else
      {
        std::cout << json{{"id", id}, {"error", "unknown method"}}.dump() << "\n";
        std::cout.flush();
        continue;
      }

      std::cout << json{{"id", id}, {"result", std::move(result)}}.dump() << "\n";
      std::cout.flush();
    }
    return 0;
  }
#else
  if (server_mode)
  {
    std::cerr << "neml2-syntax --server requires NEML2_JSON=ON\n";
    return 1;
  }
#endif

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
