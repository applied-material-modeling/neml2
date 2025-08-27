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

#include "neml2/user_tensors/CSVTensor.h"
#include "csvparser/csv.hpp"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/indexing.h"
#include "neml2/tensors/functions/stack.h"

namespace neml2
{
template <typename T>
OptionSet
CSVTensor<T>::expected_options()
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

  options.set<std::vector<std::string>>("csv_columns") = {};
  options.set("csv_columns").doc() = "Name of CSV columns";

  return options;
}

template <typename T>
CSVTensor<T>::CSVTensor(const OptionSet & options)
  : T(parse_csv(options)),
    UserTensorBase(options)
{
}

template <typename T>
T
CSVTensor<T>::parse_csv(const OptionSet & options) const
{
  std::string csv_file = options.get<std::string>("csv_file");
  TensorShape batch_shape = options.get<TensorShape>("batch_shape");
  std::vector<std::string> csv_columns = options.get<std::vector<std::string>>("csv_columns");

  // csv readers
  csv::CSVReader counter(csv_file);
  csv::CSVReader reader(csv_file);

  // vector of vectors to hold csv values
  std::vector<std::vector<double>> csv_vals;

  // set no of rows for csv_vals = no of columns specified
  if (csv_columns.empty()) // if no columns specified, use whole csv
  {
    csv_columns = counter.get_col_names();
    csv_vals.resize(counter.get_col_names().size());
  }
  else
  {
    csv_vals.resize(csv_columns.size());
  }

  // set no of columns for csv_vals = no of rows of csv
  for (auto & row : counter)
  {
    (void)row;
  }
  for (auto & row : csv_vals)
  {
    row.resize(counter.n_rows());
  }

  // fill csv_vals
  int row_count = 0;
  int col_count = 0;
  for (auto & row : reader)
  {
    for (auto & j : csv_columns)
    {
      csv_vals[row_count][col_count] = row[j].get<double>();
      row_count += 1;
    }
    col_count += 1;
    row_count = 0;
  }

  // convert each row in csv_vals to tensor, then stack and reshape
  std::vector<T> csv_convert;
  for (const auto & row : csv_vals)
  {
    auto csv_test = T::create(row);
    csv_convert.push_back(csv_test);
  }
  auto csv_stack = batch_stack(csv_convert);
  auto csv_tensor = csv_stack.batch_reshape(batch_shape);

  return csv_tensor;
}

#define REGISTER(T)                                                                                \
  using T##CSVTensor = CSVTensor<T>;                                                               \
  register_NEML2_object(T##CSVTensor);
REGISTER(Scalar);
REGISTER(SR2);
} // namespace neml2

#endif