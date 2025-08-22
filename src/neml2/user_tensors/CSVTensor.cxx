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
#include "neml2/tensors/indexing.h"
#include "csvparser/csv.hpp"

namespace neml2
{
register_NEML2_object(CSVTensor);

OptionSet
CSVTensor::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Scalar Tensor from a CSV file.";

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Name of CSV file";

  return options;
}

CSVTensor::CSVTensor(const OptionSet & options)
  : UserTensorBase(options),
    Scalar(parse_csv(options.get<std::string>("csv_file"), options.get<TensorShape>("batch_shape")))
{
}

Scalar
CSVTensor::parse_csv(const std::string & csv_file, const TensorShape & batch_shape) const
{

  csv::CSVReader counter(csv_file);
  csv::CSVReader reader(csv_file);

  // set up zeros tensor
  int no_row = 0;
  for (csv::CSVRow & row : counter)
  {
    no_row += 1;
  }
  int no_col = counter.get_col_names().size();
  int no_ele = no_row * no_col;
  auto csv_tensor = Scalar::zeros(no_ele);

  // fill zeros tensor with csv values
  int i = 0;
  for (csv::CSVRow & row : reader)
  {
    for (csv::CSVField & field : row)
    {
      csv_tensor.index_put_({i}, field.get<double>());
      i += 1;
    }
  }

  // reshape to batch shape
  csv_tensor = csv_tensor.batch_reshape(batch_shape);

  return csv_tensor;
}
} // namespace neml2

#endif