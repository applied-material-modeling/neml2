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

#include "neml2/base/Factory.h"
#include "neml2/base/guards.h"
#include "neml2/drivers/Driver.h"
#include "neml2/models/Model.h"
#include "neml2/misc/errors.h"

#include <argparse/argparse.hpp>

static std::string get_additional_cliargs(const argparse::ArgumentParser & program);

int
main(int argc, char * argv[])
{
  // Set default tensor options
  neml2::set_default_dtype(neml2::kFloat64);

  argparse::ArgumentParser program("runner");

  // sub-commend: run
  argparse::ArgumentParser run_command("run");
  run_command.add_description("Run a driver from an input file.");
  run_command.add_argument("input").help("path to the input file");
  run_command.add_argument("driver").help("name of the driver in the input file");
  run_command.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  // sub-command: diagnose
  argparse::ArgumentParser diagnose_command("diagnose");
  diagnose_command.add_description("Run diagnostics on a driver or a model from an input file.");
  diagnose_command.add_argument("input").help("path to the input file");
  auto & grp = diagnose_command.add_mutually_exclusive_group();
  grp.add_argument("-d", "--driver").help("name of the driver in the input file to diagnose");
  grp.add_argument("-m", "--model").help("name of the model in the input file to diagnose");
  diagnose_command.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  // sub-command: inspect
  argparse::ArgumentParser inspect_command("inspect");
  inspect_command.add_description("Summarize the structure of a model.");
  inspect_command.add_argument("input").help("path to the input file");
  inspect_command.add_argument("model").help("name of the model in the input file to inspect");
  inspect_command.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  // sub-command: time
  argparse::ArgumentParser time_command("time");
  time_command.add_description("Time the execution of a driver from an input file.");
  time_command.add_argument("input").help("path to the input file");
  time_command.add_argument("driver").help("name of the driver in the input file to time");
  time_command.add_argument("-n", "--num-runs")
      .default_value(1)
      .help("number of times to run the driver")
      .scan<'i', int>();
  time_command.add_argument("-w", "--warmup")
      .default_value(0)
      .help("number of warmup runs before actually measuring the model evaluation time")
      .scan<'i', int>();
  time_command.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  // Add sub-commands to the main program
  program.add_subparser(run_command);
  program.add_subparser(diagnose_command);
  program.add_subparser(inspect_command);
  program.add_subparser(time_command);

  try
  {
    // Parse cliargs
    program.parse_args(argc, argv);

    // sub-command: run
    if (program.is_subcommand_used("run"))
    {
      const auto input = run_command.get<std::string>("input");
      const auto additional_cliargs = get_additional_cliargs(run_command);
      auto factory = neml2::load_input(input, additional_cliargs);
      const auto drivername = run_command.get<std::string>("driver");
      auto driver = factory->get_driver(drivername);
      driver->run();
    }

    // sub-command: diagnose
    if (program.is_subcommand_used("diagnose"))
    {
      const auto input = diagnose_command.get<std::string>("input");
      const auto additional_cliargs = get_additional_cliargs(diagnose_command);
      auto factory = neml2::load_input(input, additional_cliargs);

      std::vector<neml2::Diagnosis> diagnoses;

      if (diagnose_command.is_used("--driver"))
      {
        auto drivername = diagnose_command.get<std::string>("--driver");
        auto driver = factory->get_driver(drivername);
        std::cout << "Diagnosing driver '" << drivername << "'...\n";
        diagnoses = neml2::diagnose(*driver);
      }
      else if (diagnose_command.is_used("--model"))
      {
        auto modelname = diagnose_command.get<std::string>("--model");
        auto model = factory->get_model(modelname);
        std::cout << "Diagnosing model '" << modelname << "'...\n";
        diagnoses = neml2::diagnose(*model);
      }
      else
      {
        std::cerr << "You must specify either a driver or a model to diagnose.\n";
        std::exit(1);
      }

      if (!diagnoses.empty())
      {
        std::cout << "Found the following potential issues(s):\n";
        for (const auto & e : diagnoses)
          std::cout << e.what() << std::endl;
        return 1;
      }
      else
        std::cout << "No issue identified :)\n";
    }

    // sub-command: inspect
    if (program.is_subcommand_used("inspect"))
    {
      const auto input = inspect_command.get<std::string>("input");
      const auto additional_cliargs = get_additional_cliargs(inspect_command);
      auto factory = neml2::load_input(input, additional_cliargs);
      const auto modelname = inspect_command.get<std::string>("model");
      auto model = factory->get_model(modelname);
      std::cout << *model << std::endl;
    }

    // sub-command: time
    if (program.is_subcommand_used("time"))
    {
      const auto input = time_command.get<std::string>("input");
      const auto additional_cliargs = get_additional_cliargs(time_command);
      auto factory = neml2::load_input(input, additional_cliargs);
      const auto drivername = time_command.get<std::string>("driver");
      auto driver = factory->get_driver(drivername);

      if (time_command.get<int>("--warmup") > 0)
        std::cout << "Warming up...\n";
      for (int i = 0; i < time_command.get<int>("--warmup"); i++)
        driver->run();

      for (int i = 0; i < time_command.get<int>("--num-runs"); i++)
      {
        neml2::TimedSection ts(drivername, "Driver::run");
        driver->run();
      }

      std::cout << "Elapsed wall time\n";
      for (const auto & [section, object_times] : neml2::timed_sections())
      {
        std::cout << "  " << section << std::endl;
        for (const auto & [object, time] : object_times)
          std::cout << "    " << object << ": " << time << " ms" << std::endl;
      }
    }
  }
  catch (const std::exception & err)
  {
    std::cerr << err.what() << std::endl;
    std::exit(1);
  }

  return 0;
}

std::string
get_additional_cliargs(const argparse::ArgumentParser & program)
{
  std::vector<std::string> args;
  try
  {
    args = program.get<std::vector<std::string>>("additional_args");
  }
  catch (std::logic_error & e)
  {
    // No additional args provided
    return "";
  }
  std::ostringstream args_stream;
  std::copy(args.begin(), args.end(), std::ostream_iterator<std::string>(args_stream, " "));
  return args_stream.str();
}
