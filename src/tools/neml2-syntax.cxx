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

#include <sstream>

#ifdef NEML2_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace
{
std::string
ftype_str(neml2::FType f)
{
  switch (f)
  {
    case neml2::FType::INPUT:
      return "INPUT";
    case neml2::FType::OUTPUT:
      return "OUTPUT";
    case neml2::FType::PARAMETER:
      return "PARAMETER";
    case neml2::FType::BUFFER:
      return "BUFFER";
    default:
      return "NONE";
  }
}

// Build the JSON record for one OptionSet. When `source_path` / `class_name`
// are empty (e.g. Settings, which isn't registered through the Registry), the
// fields are still emitted as empty strings so the schema is uniform.
json
opts_to_json(const neml2::OptionSet & opts,
             const std::string & source_path,
             const std::string & class_name,
             bool include_options)
{
  json j;
  j["type"] = opts.type();
  j["section"] = opts.section();
  j["doc"] = opts.doc();
  j["source_path"] = source_path;
  j["class_name"] = class_name;
  if (include_options)
  {
    json options = json::array();
    for (const auto & [name, opt] : opts)
    {
      if (opt->suppressed())
        continue;
      std::ostringstream value_ss;
      opt->print(value_ss);
      json o;
      o["name"] = name;
      o["doc"] = opt->doc();
      o["ftype"] = ftype_str(opt->ftype());
      o["required"] = opt->required();
      o["type"] = neml2::user_readable_type(opt->type());
      o["value"] = value_ss.str();
      options.push_back(std::move(o));
    }
    j["options"] = std::move(options);
  }
  return j;
}

// Collect every OptionSet known to the program along with its source_path.
// Settings is included first; Registry-known objects follow in alphabetical
// order of type name (the Registry uses a std::map keyed on type).
struct Record
{
  const neml2::OptionSet * opts;
  std::string source_path;
  std::string class_name;
};

std::vector<Record>
collect_records(std::vector<neml2::OptionSet> & storage)
{
  // Reserve before any push_back so pointers into `storage` remain valid.
  storage.reserve(1 + neml2::Registry::info().size());
  storage.push_back(neml2::Settings::expected_options());
  std::vector<std::string> paths;
  std::vector<std::string> class_names;
  paths.reserve(storage.capacity());
  class_names.reserve(storage.capacity());
  paths.emplace_back();
  class_names.emplace_back();
  for (const auto & [type, info] : neml2::Registry::info())
  {
    storage.push_back(info.expected_options);
    paths.push_back(info.source_path);
    class_names.push_back(info.type_name);
  }
  std::vector<Record> records;
  records.reserve(storage.size());
  for (std::size_t i = 0; i < storage.size(); ++i)
    records.push_back({&storage[i], paths[i], class_names[i]});
  return records;
}
} // namespace
#endif

int
main(int argc, char * argv[])
{
  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  argparse::ArgumentParser program("neml2-syntax");
  program.add_description("Extract object syntax from the registry as JSON. Requires NEML2_JSON.");
  program.add_argument("--json")
      .help("redirect JSON output to a file (use \"-\" for stdout)")
      .default_value(std::string("syntax.json"));
  program.add_argument("--section")
      .help("only emit objects whose input-file section matches (e.g. Models, Solvers, Drivers, "
            "Tensors, Schedulers, Data, EquationSystems, Settings)")
      .default_value(std::string(""));
  program.add_argument("--type")
      .help("only emit the object whose registered type matches (e.g. LinearDashpot); "
            "intended for drilling into a single object's full option list")
      .default_value(std::string(""));
  program.add_argument("--summary")
      .help("emit only the type, section, source_path, and doc string for each object "
            "(omit per-option detail)")
      .flag();
  program.add_argument("--server").help("run as a long-lived JSON server on stdin/stdout").flag();

  // Force link dynamic libraries
  neml2::force_link_runtime();

  // Parse cliargs
  program.parse_args(argc, argv);

#ifndef NEML2_JSON
  (void)argc;
  (void)argv;
  std::cerr << "error: neml2-syntax requires the library to be built with NEML2_JSON=ON\n";
  return 1;
#else
  const auto section_filter = program.get<std::string>("--section");
  const auto type_filter = program.get<std::string>("--type");
  const bool summary = program.get<bool>("--summary");
  const bool server_mode = program.get<bool>("--server");

  if (server_mode && (program.is_used("--json") || program.is_used("--section") ||
                      program.is_used("--type") || summary))
  {
    std::cerr << "error: --server is incompatible with --json, --section, --type, and --summary\n";
    return 1;
  }

  std::vector<neml2::OptionSet> storage;
  const auto records = collect_records(storage);

  if (server_mode)
  {
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
        for (const auto & r : records)
          if (!r.opts->section().empty())
            sections.insert(r.opts->section());
        result = json::array();
        for (const auto & s : sections)
          result.push_back(s);
      }
      else if (method == "list_types")
      {
        const std::string section = req.value("section", "");
        result = json::array();
        for (const auto & r : records)
        {
          if (!section.empty() && r.opts->section() != section)
            continue;
          if (r.opts->type().empty())
            continue;
          result.push_back({{"type", r.opts->type()},
                            {"section", r.opts->section()},
                            {"doc", r.opts->doc()},
                            {"source_path", r.source_path},
                            {"class_name", r.class_name}});
        }
      }
      else if (method == "get_options")
      {
        const std::string type = req.value("type", "");
        result = nullptr;
        for (const auto & r : records)
          if (r.opts->type() == type)
          {
            result = opts_to_json(*r.opts, r.source_path, r.class_name, /*include_options=*/true);
            break;
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

  // File / stdout JSON emission
  const auto json_path = program.get<std::string>("--json");
  std::ofstream ofs;
  std::ostream * out = &std::cout;
  if (json_path != "-")
  {
    ofs.open(json_path);
    if (!ofs.is_open())
    {
      std::cerr << "Failed to open output file: " << json_path << std::endl;
      return 1;
    }
    out = &ofs;
  }

  json result = json::array();
  for (const auto & r : records)
  {
    if (!section_filter.empty() && r.opts->section() != section_filter)
      continue;
    if (!type_filter.empty() && r.opts->type() != type_filter)
      continue;
    result.push_back(
        opts_to_json(*r.opts, r.source_path, r.class_name, /*include_options=*/!summary));
  }

  *out << result.dump(2) << '\n';

  if (ofs.is_open())
    ofs.close();

  return 0;
#endif
}
