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

#include <torch/script.h>

#include "VTestVerification.h"
#include "neml2/drivers/TransientDriver.h"
#include "neml2/misc/string_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(VTestVerification);

OptionSet
VTestVerification::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("driver");
  options.set<std::vector<std::string>>("variables");
  options.set<std::vector<TensorName<ATensor>>>("references");
  options.set<Real>("rtol") = 1e-5;
  options.set<Real>("atol") = 1e-8;
  return options;
}

VTestVerification::VTestVerification(const OptionSet & options)
  : Driver(options),
    _driver(Factory::get_object<TransientDriver>("Drivers", options.get<std::string>("driver"))),
    _rtol(options.get<Real>("rtol")),
    _atol(options.get<Real>("atol"))
{
  const auto vars = options.get<std::vector<std::string>>("variables");
  const auto vals = options.get<std::vector<TensorName<ATensor>>>("references");
  neml_assert(vars.size() == vals.size(),
              "Must provide the same number of variables and references. ",
              vars.size(),
              " variables provided, while ",
              vals.size(),
              " references provided.");
  for (std::size_t i = 0; i < vars.size(); i++)
    _ref[vars[i]] = vals[i].resolve();
}

void
VTestVerification::diagnose() const
{
  Driver::diagnose();
  neml2::diagnose(_driver);

  diagnostic_assert(!_driver.save_as_path().empty(),
                    "The driver does not save any results. Use the save_as option to specify the "
                    "destination file/path.");
}

bool
VTestVerification::run()
{
  _driver.run();

  auto res = jit::load(_driver.save_as_path());
  auto err_msg = diff(res.named_buffers(), _ref, _rtol, _atol);

  neml_assert(err_msg.empty(), err_msg);

  return true;
}

std::string
diff(const jit::named_buffer_list & res,
     const std::map<std::string, ATensor> & ref_map,
     Real rtol,
     Real atol)
{
  std::map<std::string, ATensor> res_map;
  for (auto item : res)
    res_map.emplace(item.name, item.value);

  std::ostringstream err_msg;

  for (const auto & [name, val] : ref_map)
  {
    const auto tokens = utils::split(name, ".");
    if (tokens.size() < 2)
      err_msg << "Invalid reference variable name " << name << ".\n";
    const auto nstep = val.size(0);
    for (Size i = 0; i < nstep; i++)
    {
      const auto refi = val.index({i}).squeeze();
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
                << ". Maximum mixed difference = " << std::scientific << diff.max().item<Real>()
                << " > atol = " << std::scientific << atol << "\n";
        err_msg << "Reference: " << refi << "\n";
        err_msg << "Result: " << resi << "\n";
      }
    }
  }

  return err_msg.str();
}
}
