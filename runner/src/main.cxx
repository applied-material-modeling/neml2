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

#include "neml2/base/Parser.h"
#include "neml2/base/Factory.h"
#include "neml2/base/guards.h"
#include "neml2/drivers/Driver.h"

#include <argparse/argparse.hpp>
#include <chrono>

int
main(int argc, char * argv[])
{
  argparse::ArgumentParser program("runner");

  // Positional arguments
  program.add_argument("input").help("path to the input file");
  program.add_argument("driver").help("name of the driver in the input file");
  program.add_argument("additional_args")
      .remaining()
      .help("additional command-line arguments to pass to the input file parser");

  // Optional arguments
  program.add_argument("-d", "--diagnose")
      .help("run diagnostics on common problems and exit (without further execution)")
      .flag();
  program.add_argument("-t", "--time")
      .help("output the elapsed wall time during model evaluation")
      .flag();
  program.add_argument("-n", "--num-runs")
      .default_value(1)
      .help("number of times to run the driver")
      .scan<'i', int>();
  program.add_argument("--warmup")
      .default_value(0)
      .help("number of warmup runs before actually measuring the model evaluation time")
      .scan<'i', int>();

  try
  {
    // Parse cliargs
    program.parse_args(argc, argv);
    const auto input = program.get<std::string>("input");
    const auto drivername = program.get<std::string>("driver");

    // Remaining args
    std::vector<std::string> args;
    try
    {
      args = program.get<std::vector<std::string>>("additional_args");
    }
    catch (std::logic_error & e)
    {
    }
    std::ostringstream args_stream;
    std::copy(args.begin(), args.end(), std::ostream_iterator<std::string>(args_stream, " "));
    const auto additional_cliargs = args_stream.str();

    // Run the model
    try
    {
      neml2::load_input(input, additional_cliargs);
      auto & driver = neml2::Factory::get_object<neml2::Driver>("Drivers", drivername);

      if (program["--diagnose"] == true)
      {
        std::cout << "Running diagnostics on input file '" << input << "' driver '" << drivername
                  << "'...\n";

        const auto diagnoses = neml2::diagnose(driver);

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

      if (program["--time"] == true)
      {
        if (program.get<int>("--warmup") > 0)
          std::cout << "Warming up...\n";
        for (int i = 0; i < program.get<int>("--warmup"); i++)
          driver.run();
      }

      for (int i = 0; i < program.get<int>("--num-runs"); i++)
      {
        neml2::TimedSection ts(drivername, "Driver::run");
        driver.run();
      }

      if (program["--time"] == true)
      {
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
      std::cerr << "An exception was raised while running the model:\n";
      std::cerr << err.what() << std::endl;
      std::exit(1);
    }
  }
  catch (const std::exception & err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  return 0;
}
