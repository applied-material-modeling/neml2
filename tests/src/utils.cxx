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
#include "neml2/base/Parser.h"

std::unordered_set<neml2::Device> &
set_test_suite_additional_devices()
{
  static std::unordered_set<neml2::Device> _test_suite_additional_devices;
  return _test_suite_additional_devices;
}

const std::unordered_set<neml2::Device> &
get_test_suite_additional_devices()
{
  return set_test_suite_additional_devices();
}

int
test_main(int argc, char * argv[], const std::string & name)
{
  Catch::Session session;

  // Add path cli arg
  using namespace Catch::Clara;
  std::string working_dir;
  std::string additional_devs;
  auto cli = session.cli() | Opt(working_dir, ".")["-p"]["--path"]("path to the test input files") |
             Opt(additional_devs, "")["-d"]["--devices"](
                 "additional (non-CPU) devices to provide to the test suite");
  session.cli(cli);

  // Let Catch2 parse the command line
  auto err = session.applyCommandLine(argc, argv);
  if (err)
    return err;

  // Set the working directory
  auto exec_prefix = std::filesystem::path(argv[0]).parent_path();
  err = guess_test_dir(name, working_dir, exec_prefix);
  if (err)
    return err;
  std::filesystem::current_path(working_dir);

  // Get additional devices
  err = init_test_devices(additional_devs);
  if (err)
  {
    std::cerr << "Failed to parse additional devices: " << additional_devs << ". Reason: ";
    switch (err)
    {
      case 1:
        std::cerr << "Invalid device specification.";
        break;
      case 2:
        std::cerr << "Invalid device type (e.g., contains CPU).";
        break;
      case 3:
        std::cerr << "Duplicate device specification.";
        break;
      default:
        std::cerr << "Unknown error.";
        break;
    }
    std::cerr << std::endl;

    return err;
  }

  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  // Print test suite configuration
  std::cout << "Working directory: " << std::string(std::filesystem::current_path()) << std::endl;
  std::cout << "Additional devices: ";
  for (const auto & device : get_test_suite_additional_devices())
    std::cout << device << " ";
  std::cout << std::endl;

  return session.run();
}

int
try_hint(const std::string & stem, std::string & hint)
{
  namespace fs = std::filesystem;

  const auto hint_path = fs::path(hint);
  fs::path candidate;

  candidate = hint_path / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  candidate = hint_path / "tests" / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  candidate = hint_path / "share" / "neml2" / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  return 1;
}

int
guess_test_dir(const std::string & stem, std::string & hint, const std::string & exec_prefix)
{
  namespace fs = std::filesystem;

  // Check if hint is an exact match
  const auto hint_path = fs::path(hint);
  if (fs::is_directory(hint_path) && hint_path.stem() == stem)
  {
    hint = std::string(fs::absolute(hint_path));
    return 0;
  }

  hint = exec_prefix + "/../../../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/../../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix;
  if (try_hint(stem, hint) == 0)
    return 0;

  return 1;
}

int
init_test_devices(const std::string & additional_devs)
{
  auto & devs = set_test_suite_additional_devices();
  auto tokens = neml2::utils::split(additional_devs, ",");
  for (const auto & token : tokens)
  {
    neml2::Device device = neml2::kCPU;
    auto success = neml2::utils::parse_<neml2::Device>(device, token);
    if (!success)
      return 1;

    if (device == neml2::kCPU)
      return 2;

    if (devs.find(device) != devs.end())
      return 3;

    devs.insert(device);
  }
  return 0;
}
