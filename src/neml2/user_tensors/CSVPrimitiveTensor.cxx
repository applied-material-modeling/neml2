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

#ifdef NEML2_HAS_CSV

#include "neml2/user_tensors/CSVPrimitiveTensor.h"
#include "csvparser/csv.hpp"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/stack.h"

namespace neml2
{
template <typename T>
OptionSet
CSVPrimitiveTensor<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a " + tensor_type + " from a CSV file.";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Name of CSV file";

  options.set<TensorShape>("batch_shape");
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::string>("tensor_name");
  options.set("tensor_name").doc() = "Name of tensor.";

  return options;
}

template <typename T>
CSVPrimitiveTensor<T>::CSVPrimitiveTensor(const OptionSet & options)
  : T(parse_csv(options.get<std::string>("csv_file"),
                options.get<std::string>("tensor_name"),
                options.get<TensorShape>("batch_shape"))),
    UserTensorBase(options)
{
}

template <typename T>
T
CSVPrimitiveTensor<T>::parse_csv(const std::string & csv_file,
                                 const std::string & tensor_name,
                                 const TensorShape & batch_shape) const
{
  csv::CSVReader reader(csv_file);

  std::vector<double> csv_vals;

  for (const auto & row : reader)
  {
    csv_vals.push_back(row[tensor_name].get<double>());
  }

  auto csv_tensor = T::create(csv_vals).batch_reshape(batch_shape);
  return csv_tensor;
}

template <>
Vec
CSVPrimitiveTensor<Vec>::parse_csv(const std::string & csv_file,
                                   const std::string & tensor_name,
                                   const TensorShape & batch_shape) const
{
  csv::CSVReader reader(csv_file);

  std::vector<Vec> csv_vals;
  std::vector<std::string> suffix = {"_x", "_y", "_z"};

  for (const auto & row : reader)
  {
    std::vector<double> csv_row = {};
    for (auto & j : suffix)
    {
      csv_row.push_back(row[tensor_name + j].get<double>());
    }
    auto csv_row_tens = Vec::create(csv_row);
    csv_vals.push_back(csv_row_tens);
  }

  auto csv_tensor = batch_stack(csv_vals).batch_reshape(batch_shape);
  return csv_tensor;
}

template <>
SR2
CSVPrimitiveTensor<SR2>::parse_csv(const std::string & csv_file,
                                   const std::string & tensor_name,
                                   const TensorShape & batch_shape) const
{
  csv::CSVReader reader(csv_file);

  std::vector<SR2> csv_vals;

  std::vector<std::string> suffix = {"_xx", "_yy", "_zz", "_yz", "_xz", "_xy"};

  for (const auto & row : reader)
  {
    std::vector<double> csv_row = {};
    for (auto & j : suffix)
    {
      csv_row.push_back(row[tensor_name + j].get<double>());
    }
    auto csv_row_tens = SR2::create(csv_row);
    csv_vals.push_back(csv_row_tens);
  }

  auto csv_tensor = batch_stack(csv_vals).batch_reshape(batch_shape);
  return csv_tensor;
}

#define CSVPRIMITIVETENSOR_REGISTER(T)                                                             \
  using CSV##T = CSVPrimitiveTensor<T>;                                                            \
  register_NEML2_object_alias(CSV##T, "CSV" #T)
CSVPRIMITIVETENSOR_REGISTER(Scalar);
CSVPRIMITIVETENSOR_REGISTER(Vec);
CSVPRIMITIVETENSOR_REGISTER(SR2);
} // namespace neml2

#endif