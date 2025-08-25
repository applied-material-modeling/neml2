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
#include "neml2/tensors/tensors.h"

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

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Name of CSV file";

  return options;
}

template <typename T>
CSVTensor<T>::CSVTensor(const OptionSet & options)
  : T(parse_csv(options.get<std::string>("csv_file"), options.get<TensorShape>("batch_shape"))),
    UserTensorBase(options)
{
}

template <typename T>
T
CSVTensor<T>::parse_csv(const std::string & csv_file, const TensorShape & batch_shape) const
{
  std::vector<double> csv_vals;

  csv::CSVReader reader(csv_file);

  for (csv::CSVRow & row : reader)
  {
    for (csv::CSVField & field : row)
    {
      csv_vals.push_back(field.get<double>());
    }
  }

  auto csv_tensor = neml2::Tensor::create(csv_vals).batch_reshape(batch_shape);

  return csv_tensor;
}

#define REGISTER(T)                                                                                \
  using T##CSVTensor = CSVTensor<T>;                                                               \
  register_NEML2_object(T##CSVTensor);
REGISTER(Scalar);

} // namespace neml2

#endif