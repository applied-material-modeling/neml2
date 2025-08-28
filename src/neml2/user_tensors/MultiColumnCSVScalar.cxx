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

#include "neml2/user_tensors/MultiColumnCSVScalar.h"
#include "csvparser/csv.hpp"

namespace neml2
{
register_NEML2_object(MultiColumnCSVScalar);

OptionSet
MultiColumnCSVScalar::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Scalar from a CSV file by column.";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Path to the CSV file";

  options.set<TensorShape>("batch_shape");
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::vector<std::string>>("csv_columns") = {};
  options.set("csv_columns").doc() = "Name of CSV columns. Default is all the columnss";

  return options;
}

MultiColumnCSVScalar::MultiColumnCSVScalar(const OptionSet & options)
  : Scalar(parse_csv(options)),
    UserTensorBase(options)
{
}

Scalar
MultiColumnCSVScalar::parse_csv(const OptionSet & options) const
{
  std::string csv_file = options.get<std::string>("csv_file");
  TensorShape batch_shape = options.get<TensorShape>("batch_shape");
  std::vector<std::string> csv_columns = options.get<std::vector<std::string>>("csv_columns");

  csv::CSVReader counter(csv_file);
  csv::CSVReader reader(csv_file);

  // Vector of vectors to hold CSV values
  std::vector<std::vector<double>> csv_vals;
  if (csv_columns.empty())
    csv_columns = counter.get_col_names();
  csv_vals.resize(csv_columns.size());
  for (auto & row : counter)
    (void)row;
  for (auto & row : csv_vals)
    row.resize(counter.n_rows());

  // Read CSV values
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

  // Convert to Scalar and reshape
  std::vector<double> csv_flatten;
  for (const auto & row : csv_vals)
  {
    for (const auto & value : row)
    {
      csv_flatten.push_back(value);
    }
  }
  auto csv_tensor = Scalar::create(csv_flatten).batch_reshape(batch_shape);

  return csv_tensor;
}
} // namespace neml2

#endif