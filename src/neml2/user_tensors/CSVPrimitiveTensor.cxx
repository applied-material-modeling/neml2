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

  OptionSet options = CSVTensorBase::expected_options();
  options.doc() =
      "Construct a " + tensor_type + " from a CSV file. Each component of the " + tensor_type +
      " should be provided as a column in the CSV. By default the entire CSV is read in order of "
      "the columns. A subset/different order of columns can be selected using `column_indices` or "
      "`column_names`. The rows of the CSV "
      "correspond to different instances of the " +
      tensor_type +
      ", and by default the number of rows is the batch size. If `batch_shape` is specified, an "
      "additional reshaping is performed";

  return options;
}

template <typename T>
CSVPrimitiveTensor<T>::CSVPrimitiveTensor(const OptionSet & options)
  : CSVTensorBase(options),
    T(parse_csv(options))
{
}

template <typename T>
T
CSVPrimitiveTensor<T>::parse_csv(const OptionSet & options) const
{
  const unsigned int base_size_ele = utils::storage_size(T::const_base_sizes);

  // Parse CSV
  const auto fmt = CSVTensorBase::parse_format();
  csv::CSVReader csv(options.get<std::string>("csv_file"), fmt);
  check_col();
  const auto indices = CSVTensorBase::parse_indices(csv);

  // CSV data
  std::vector<double> vals;
  std::size_t nrow = 0, ncol = 0;

  if (indices.empty())
  {
    CSVTensorBase::read_all(csv, vals, nrow, ncol);
    if (ncol != base_size_ele)
      throw NEMLException("Number of columns provided (" + std::to_string(ncol) +
                          ") does not match the expected number of components in " +
                          utils::demangle(typeid(T).name()).substr(7) + " (" +
                          std::to_string(base_size_ele) + ").");
  }
  else
    CSVTensorBase::read_by_indices(csv, indices, vals, nrow, ncol);

  // Multiplying factor
  multiply_factor(vals, nrow, ncol);

  // Convert to tensor and reshape
  std::vector<std::vector<double>> vals_double(nrow, std::vector<double>(ncol));
  for (std::size_t i = 0; i < nrow; i++)
    for (std::size_t j = 0; j < ncol; j++)
      vals_double[i][j] = vals[i * ncol + j];
  std::vector<T> vals_convert(nrow);
  for (std::size_t i = 0; i < nrow; i++)
    vals_convert[i] = T::create(vals_double[i]);
  auto csv_tensor = batch_stack(vals_convert).batch_reshape(Size(nrow));

  // Reshape if requested
  if (options.user_specified("batch_shape"))
  {
    const auto B = options.get<TensorShape>("batch_shape");
    neml_assert(csv_tensor.numel() == utils::storage_size(B) * base_size_ele,
                "The requested batch_shape ",
                B,
                " is incompatible with the number of values read from the CSV file (",
                csv_tensor.numel(),
                ").");
    csv_tensor = csv_tensor.batch_reshape(B);
  }

  return csv_tensor;
}

template <typename T>
void
CSVPrimitiveTensor<T>::check_col() const
{
  const auto & options = this->input_options();
  const unsigned int base_size = utils::storage_size(T::const_base_sizes);

  if (options.user_specified("column_names"))
  {
    const auto col_names = options.template get<std::vector<std::string>>("column_names");
    if (col_names.size() != base_size)
      throw NEMLException("Number of column_names provided (" + std::to_string(col_names.size()) +
                          ") does not match the expected number of components in " +
                          utils::demangle(typeid(T).name()).substr(7) + " (" +
                          std::to_string(base_size) + ").");
  }

  if (options.user_specified("column_indices"))
  {
    const auto indices = options.template get<std::vector<unsigned int>>("column_indices");
    if (indices.size() != base_size)
      throw NEMLException("Number of column_indices provided (" + std::to_string(indices.size()) +
                          ") does not match the expected number of components in " +
                          utils::demangle(typeid(T).name()).substr(7) + " (" +
                          std::to_string(base_size) + ").");
  }
}

template <typename T>
void
CSVPrimitiveTensor<T>::multiply_factor(std::vector<double> & vals,
                                       std::size_t & nrow,
                                       std::size_t & ncol) const
{
  (void)vals;
  (void)nrow;
  (void)ncol;
}

template <>
void
CSVPrimitiveTensor<SR2>::multiply_factor(std::vector<double> & vals,
                                         std::size_t & nrow,
                                         std::size_t & ncol) const
{
  std::vector<double> factor = {1.0, 1.0, 1.0, std::sqrt(2), std::sqrt(2), std::sqrt(2)};
  for (std::size_t i = 0; i < nrow; i++)
    for (std::size_t j = 0; j < ncol; j++)
      vals[i * ncol + j] *= factor[j];
}

#define CSVPRIMITIVETENSOR_REGISTER(T)                                                             \
  using CSV##T = CSVPrimitiveTensor<T>;                                                            \
  register_NEML2_object_alias(CSV##T, "CSV" #T)
CSVPRIMITIVETENSOR_REGISTER(Scalar);
CSVPRIMITIVETENSOR_REGISTER(Vec);
CSVPRIMITIVETENSOR_REGISTER(SR2);
} // namespace neml2

#endif