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

#include "SampleParserTestingModel.h"

using namespace neml2;

register_NEML2_object(SampleParserTestingModel);

OptionSet
SampleParserTestingModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "This model tests the correctness of parsed options.";

  // Types we support:
  //   bool
  //   int
  //   unsigned int
  //   Real
  //   string
  //
  // We also support vector of and vector of vector of each integral type.
  options.set<bool>("bool");
  options.set<std::vector<bool>>("bool_vec");
  options.set<std::vector<std::vector<bool>>>("bool_vec_vec");

  options.set<int>("int");
  options.set<std::vector<int>>("int_vec");
  options.set<std::vector<std::vector<int>>>("int_vec_vec");

  options.set<unsigned int>("uint");
  options.set<std::vector<unsigned int>>("uint_vec");
  options.set<std::vector<std::vector<unsigned int>>>("uint_vec_vec");

  options.set<Real>("Real");
  options.set<std::vector<Real>>("Real_vec");
  options.set<std::vector<std::vector<Real>>>("Real_vec_vec");

  options.set<std::string>("string");
  options.set<std::vector<std::string>>("string_vec");
  options.set<std::vector<std::vector<std::string>>>("string_vec_vec");

  options.set<TensorShape>("shape");
  options.set<std::vector<TensorShape>>("shape_vec");
  options.set<std::vector<std::vector<TensorShape>>>("shape_vec_vec");

  options.set<Device>("device");
  options.set<std::vector<Device>>("device_vec");
  options.set<std::vector<std::vector<Device>>>("device_vec_vec");

  options.set<std::string>("suppressed_option") = "suppress me";
  options.set("suppressed_option").suppressed() = true;

  return options;
}
