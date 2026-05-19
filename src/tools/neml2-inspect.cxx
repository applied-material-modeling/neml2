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
#include "neml2/models/ParameterStore.h"
#include "neml2/models/BufferStore.h"
#include "neml2/misc/string_utils.h"

#include <argparse/argparse.hpp>
#include "utils.h"

#ifdef NEML2_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace
{
std::shared_ptr<neml2::Model>
load_model(const argparse::ArgumentParser & program)
{
  const auto input = program.get<std::string>("input");
  const auto additional_cliargs = get_additional_cliargs(program);
  auto factory = neml2::load_input(input, additional_cliargs);
  const auto modelname = program.get<std::string>("model");
  return factory->get_model(modelname);
}
} // namespace

#ifdef NEML2_JSON
namespace
{
using neml2::utils::stringify;

json
model_to_json(const neml2::Model & model)
{
  json j;
  j["name"] = model.name();
  j["host"] = model.host()->name();

  json inputs = json::array();
  for (auto && [name, var] : model.input_variables())
    inputs.push_back({{"name", name}, {"type", stringify(var->type())}});
  j["inputs"] = std::move(inputs);

  json outputs = json::array();
  for (auto && [name, var] : model.output_variables())
    outputs.push_back({{"name", name}, {"type", stringify(var->type())}});
  j["outputs"] = std::move(outputs);

  const auto * pstore = model.host<neml2::ParameterStore>();
  json parameters = json::array();
  for (auto && [name, param] : pstore->named_parameters())
    parameters.push_back({{"name", name},
                          {"type", stringify(param->type())},
                          {"dtype", stringify(neml2::Tensor(*param).scalar_type())},
                          {"device", stringify(neml2::Tensor(*param).device())}});
  j["parameters"] = std::move(parameters);

  const auto * bstore = model.host<neml2::BufferStore>();
  json buffers = json::array();
  for (auto && [name, buffer] : bstore->named_buffers())
    buffers.push_back({{"name", name},
                       {"type", stringify(buffer->type())},
                       {"dtype", stringify(neml2::Tensor(*buffer).scalar_type())},
                       {"device", stringify(neml2::Tensor(*buffer).device())}});
  j["buffers"] = std::move(buffers);

  return j;
}
} // namespace
#endif

int
main(int argc, char * argv[])
{
  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  argparse::ArgumentParser program("neml2-inspect");
  program.add_description("Summarize the structure of a model.");
  program.add_argument("input").help("path to the input file");
  program.add_argument("model").help("name of the model in the input file to inspect");
  program.add_argument("--json")
      .help("emit a structured JSON description on stdout instead of the human-readable format "
            "(requires NEML2_JSON). Errors are returned as {\"retcode\": <nonzero>, \"error\": "
            "\"...\"}; the process exit code mirrors retcode.")
      .flag();
  program.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  try
  {
    program.parse_args(argc, argv);
  }
  catch (const std::exception & err)
  {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  const bool json_mode = program.get<bool>("--json");

#ifndef NEML2_JSON
  if (json_mode)
  {
    std::cerr << "neml2-inspect --json requires NEML2_JSON=ON\n";
    return 2;
  }
#endif

  try
  {
    auto model = load_model(program);
#ifdef NEML2_JSON
    if (json_mode)
    {
      auto out = model_to_json(*model);
      out["retcode"] = 0;
      std::cout << out.dump() << std::endl;
      return 0;
    }
#endif
    std::cout << *model << std::endl;
  }
  catch (const std::exception & err)
  {
#ifdef NEML2_JSON
    if (json_mode)
    {
      std::cout << json{{"retcode", 1}, {"error", err.what()}}.dump() << std::endl;
      return 1;
    }
#endif
    std::cerr << err.what() << std::endl;
    return 1;
  }

  return 0;
}
