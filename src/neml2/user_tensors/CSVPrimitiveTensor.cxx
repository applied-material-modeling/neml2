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

#ifdef NEML2_CSV

#include "neml2/user_tensors/CSVPrimitiveTensor.h"

namespace neml2
{
template <typename T>
OptionSet
CSVPrimitiveTensor<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options += CSVReader::expected_options();
  options.doc() =
      "Construct a " + UserTensorBase<T>::tensor_type() +
      " from a CSV file. Each component of the " + UserTensorBase<T>::tensor_type() +
      " should be provided as a column in the CSV. By default the entire CSV is read in order of "
      "the columns. A subset/different order of columns can be selected using `column_indices` or "
      "`column_names`. Each row of the CSV corresponds to a batch. The default batch shape is the "
      "number of rows in the CSV.";

  return options;
}

template <typename T>
CSVPrimitiveTensor<T>::CSVPrimitiveTensor(const OptionSet & options)
  : UserTensorBase<T>(options),
    CSVReader(this)
{
}

static void
check_ncol(std::size_t ncol,
           std::size_t expected,
           const std::string & tensor_type,
           const std::optional<std::string> & option_name = std::nullopt)
{
  neml_assert(ncol == expected,
              "Number of columns ",
              option_name ? "specified in " + *option_name + " " : "",
              "(",
              ncol,
              ") does not match the expected number of components in ",
              tensor_type,
              " (",
              expected,
              ").");
}

template <typename T>
T
CSVPrimitiveTensor<T>::make() const
{
  const auto & options = this->input_options();

  // Parse CSV format and load CSV
  const auto fmt = parse_format();
  csv::CSVReader csv(options.template get<std::string>("csv_file"), fmt);

  // Check number of columns if specified
  if (options.user_specified("column_names"))
  {
    const auto col_names = options.template get<std::vector<std::string>>("column_names");
    check_ncol(
        col_names.size(), T::const_base_numel, UserTensorBase<T>::tensor_type(), "column_names");
  }
  else if (options.user_specified("column_indices"))
  {
    const auto col_indices = options.template get<std::vector<unsigned int>>("column_indices");
    check_ncol(col_indices.size(),
               T::const_base_numel,
               UserTensorBase<T>::tensor_type(),
               "column_indices");
  }

  // Parse indices
  const auto indices = parse_indices(csv);

  // Read CSV data
  std::vector<double> vals;
  std::size_t nrow = 0, ncol = 0;

  if (indices.empty())
  {
    read_all(csv, vals, nrow, ncol);
    check_ncol(ncol, T::const_base_numel, UserTensorBase<T>::tensor_type());
  }
  else
    read_by_indices(csv, indices, vals, nrow, ncol);

  // Convert to tensor and reshape
  auto csv_data = Scalar::create(vals).dynamic_reshape({Size(nrow), Size(ncol)});

  // Split the data into components
  auto comps = at::split(csv_data, 1, -1);
  std::array<Scalar, T::const_base_numel> scalar_comps;
  for (std::size_t i = 0; i < T::const_base_numel; ++i)
    scalar_comps[i] = Scalar(comps[i].squeeze(-1), 0);

  // Dispatch to the tensor type's fill method
  auto csv_tensor = std::apply(
      [&](auto &&... args) { return T::fill(std::forward<Scalar>(args)...); }, scalar_comps);

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
