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

  // counting row/column number
  csv::CSVReader counter(csv_file);

  // vector of vectors to hold csv values
  std::vector<std::vector<double>> csv_vals;

  // set no of rows for vector of vectors using counter
  if (csv_columns.empty()) // if no column specified, use whole csv
  {
    csv_columns = counter.get_col_names();
    csv_vals.resize(counter.get_col_names().size());
  }
  else // columns specified
  {
    csv_vals.resize(csv_columns.size());
  }

  // set no of columns for vector of vector using counter
  for (auto & row : counter)
  {
  }
  for (auto & row : csv_vals)
  {
    row.resize(counter.n_rows());
  }

  // read csv values into vector of vectors
  int row_count = 0;
  int col_count = 0;
  csv::CSVReader reader(csv_file);
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

  // flatten vector of vectors
  std::vector<double> csv_vals_2;
  for (const auto & inner_vec : csv_vals)
  {
    for (const auto & element : inner_vec)
    {
      csv_vals_2.push_back(element);
    }
  }

  // convert to tensor and reshape to batch size
  auto csv_tensor_2 = neml2::Tensor::create(csv_vals_2, 1).batch_reshape(batch_shape);
  return csv_tensor_2;
}

#define REGISTER(T)                                                                                \
  using T##CSVTensor = CSVTensor<T>;                                                               \
  register_NEML2_object(T##CSVTensor);
REGISTER(Scalar);
// REGISTER(SR2);
} // namespace neml2

#endif