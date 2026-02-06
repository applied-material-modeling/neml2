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

#include <algorithm>

#include <torch/script.h>

#include "VTestVerification.h"
#include "neml2/drivers/TransientDriver.h"
#include "neml2/misc/string_utils.h"
#include "neml2/misc/assertions.h"
#include "neml2/base/Factory.h"

namespace neml2
{
static std::string diff(const torch::jit::named_buffer_list & res,
                        const std::map<std::string, Tensor> & ref_map,
                        double rtol,
                        double atol,
                        std::vector<size_t> & indices);

register_NEML2_object(VTestVerification);

template <typename T>
static void
setup_ref_values(const OptionSet & options,
                 std::map<std::string, Tensor> & ref,
                 const std::string & name_opt,
                 const std::string & value_opt)
{
  const auto vars = options.get<std::vector<std::string>>(name_opt);
  const auto vals = options.get<std::vector<TensorName<T>>>(value_opt);
  neml_assert(vars.size() == vals.size(),
              "Must provide the same number of variables and references.",
              vars.size(),
              " variables provided to ",
              name_opt,
              ", while ",
              vals.size(),
              " references provided to ",
              value_opt,
              ".");
  auto * factory = options.get<Factory *>("_factory");
  for (std::size_t i = 0; i < vars.size(); i++)
    ref[vars[i]] = Tensor(vals[i].resolve(factory));
}

OptionSet
VTestVerification::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("driver");

#define OPTION_VAR_(T)                                                                             \
  options.set<std::vector<std::string>>(#T "_names");                                              \
  options.set(#T "_names").doc() = "Variables of type " #T " to compare";                          \
  options.set<std::vector<TensorName<T>>>(#T "_values");                                           \
  options.set(#T "_values").doc() = "Reference variables of type " #T " to be compared against"
  FOR_ALL_TENSORBASE(OPTION_VAR_);

  options.set<double>("rtol") = 1e-5;
  options.set<double>("atol") = 1e-8;

  options.set<std::vector<size_t>>("time_steps") = {};
  options.set("time_steps").doc() =
      "Time steps provided in the provided reference solutions to verify. If empty, all time steps "
      "must be provided and will be verified.";
  return options;
}

VTestVerification::VTestVerification(const OptionSet & options)
  : Driver(options),
    _driver(get_driver<TransientDriver>("driver")),
    _rtol(options.get<double>("rtol")),
    _atol(options.get<double>("atol")),
    _time_steps(options.get<std::vector<size_t>>("time_steps"))
{
#define SETUP_REF_(T) setup_ref_values<T>(options, _ref, #T "_names", #T "_values")
  FOR_ALL_TENSORBASE(SETUP_REF_);
}

void
VTestVerification::diagnose() const
{
  Driver::diagnose();
  neml2::diagnose(*_driver);

  diagnostic_assert(!_driver->save_as_path().empty(),
                    "The driver does not save any results. Use the save_as option to specify the "
                    "destination file/path.");
}

bool
VTestVerification::run()
{
  _driver->run();

  auto res = torch::jit::load(_driver->save_as_path());
  auto err_msg = diff(res.named_buffers(), _ref, _rtol, _atol, _time_steps);

  neml_assert(err_msg.empty(), err_msg);

  return true;
}

std::string
diff(const torch::jit::named_buffer_list & res,
     const std::map<std::string, Tensor> & ref_map,
     double rtol,
     double atol,
     std::vector<size_t> & indices)
{
  // Sort indices for future use
  std::sort(indices.begin(), indices.end());

  std::map<std::string, ATensor> res_map;
  for (auto item : res)
    res_map.emplace(item.name, item.value);

  std::ostringstream err_msg;

  for (const auto & [name, val] : ref_map)
  {
    const auto tokens = utils::split(name, ".");
    if (tokens.size() < 2)
      err_msg << "Invalid reference variable name " << name << ".\n";
    Size nstep = 0;
    bool found = false;
    for (const auto & [resname, resval] : res_map)
    {
      const auto restokens = utils::split(resname, ".");
      if (restokens.size() != tokens.size() + 1)
        continue;
      if (restokens.front() != tokens.front())
        continue;
      if (!std::equal(tokens.begin() + 1, tokens.end(), restokens.begin() + 2))
        continue;

      try
      {
        const auto step = static_cast<Size>(std::stoull(restokens[1]));
        nstep = std::max(nstep, step + 1);
        found = true;
      }
      catch (const std::exception &)
      {
        continue;
      }
    }
    if (!found)
    {
      err_msg << "Result is missing variable " << name << ".\n";
      continue;
    }
    Size j = 0;
    for (Size i = 0; i < nstep; i++)
    {
      if (!indices.empty() &&
          !std::binary_search(indices.begin(), indices.end(), static_cast<size_t>(i)))
        continue;

      const auto refi = indices.size() == 1 ? at::Tensor(val) : val.index({j}).squeeze();
      j++;

      auto restokens = tokens;
      restokens.insert(restokens.begin() + 1, std::to_string(i));
      const auto resname = utils::join(restokens, ".");

      if (!res_map.count(resname))
      {
        if (!at::allclose(refi, at::zeros_like(refi)))
          err_msg << "Result is missing variable " << resname << ".\n";
        continue;
      }

      const auto resi = res_map[resname].squeeze();
      if (!at::allclose(resi, refi, rtol, atol))
      {
        const auto diff = at::abs(resi - refi) - rtol * at::abs(refi);
        err_msg << "Result has wrong value for variable " << resname
                << ". Maximum mixed difference = " << std::scientific << diff.max().item<double>()
                << " > atol = " << std::scientific << atol << "\n";
        err_msg << "Reference: " << refi << "\n";
        err_msg << "Result: " << resi << "\n";
      }
    }
  }

  return err_msg.str();
}
}
