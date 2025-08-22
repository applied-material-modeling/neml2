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
#include <sstream>
#include "neml2/misc/assertions.h"

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
  options.doc() =
      "Construct a " + tensor_type + " from a CSV file. Each component of the " + tensor_type +
      " should be provided as a column in the CSV. By default the entire CSV is read in order of "
      "the columns. A subset/different order of columns can be selected using `column_indices` or "
      "`column_names`. The rows of the CSV "
      "correspond to different instances of the " +
      tensor_type +
      ", and by default the number of rows is the batch size. If `batch_shape` is specified, an "
      "additional reshaping is performed";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Path to the CSV file";

  options.set<TensorShape>("batch_shape");
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::vector<std::string>>("column_names");
  options.set("column_names").doc() =
      "Names of CSV columns corresponding to components of " + tensor_type;

  options.set<std::vector<unsigned int>>("column_indices") = {};
  options.set("column_indices").doc() =
      "Indices of CSV columns corresponding to components of " + tensor_type;

  EnumSelection delimiter_selection(
      {"COMMA", "SEMICOLON", "SPACE", "TAB"}, {',', ';', ' ', '\t'}, "COMMA");
  options.set<EnumSelection>("delimiter") = delimiter_selection;
  options.set("delimiter").doc() =
      "Delimiter used to parse the CSV file. Options are " + delimiter_selection.candidates_str();

  options.set<bool>("no_header") = false;
  options.set("no_header").doc() = "Whether the CSV file has a header row.";

  options.set<int>("starting_row") = 0;
  options.set("starting_row").doc() =
      "Starting row of the CSV file (0-indexed). Rows before this row are ignored. This should be "
      "the header row if the CSV file has a header, otherwise it should be the first row of the "
      "data. By default the starting row is the 0th row.";

  return options;
}

template <typename T>
CSVPrimitiveTensor<T>::CSVPrimitiveTensor(const OptionSet & options)
  : UserTensorBase(options),
    T(parse_csv(options))
{
}

template <typename T>
T
CSVPrimitiveTensor<T>::parse_csv(const OptionSet & options) const
{
  // Parse CSV
  const auto fmt = parse_format();
  csv::CSVReader csv(options.get<std::string>("csv_file"), fmt);
  const auto indices = parse_indices(csv);

  // CSV data
  std::vector<T> vals;
  std::size_t nrow = 0;

  if (indices.empty())
    vals = read_all(csv, nrow);
  else
    vals = read_by_indices(csv, indices, nrow);

  // Reshape
  auto csv_tensor = batch_stack(vals).batch_reshape(Size(nrow));
  if (options.user_specified("batch_shape"))
  {
    const auto B = options.get<TensorShape>("batch_shape");
    neml_assert(csv_tensor.numel() ==
                    utils::storage_size(B) * utils::storage_size(T::const_base_sizes),
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
csv::CSVFormat
CSVPrimitiveTensor<T>::parse_format() const
{
  const auto & options = this->input_options();
  csv::CSVFormat fmt;

  // Delimiter
  const auto delim = options.template get<EnumSelection>("delimiter").template as<char>();
  fmt.delimiter(delim);

  if (options.template get<bool>("no_header"))
  {
    // Set 'header row' to one before starting row of data, so that we start reading data from the
    // starting row
    fmt.header_row(options.template get<int>("starting_row") - 1);
    // This is a csvparser bug. It false-positively detects variable columns
    fmt.variable_columns(csv::VariableColumnPolicy::KEEP);
  }
  // no_header=false
  else
  {
    // Use starting_row as header row
    const auto starting_row = options.template get<int>("starting_row");
    neml_assert(starting_row >= 0,
                "starting_row must be non-negative. If your CSV file has no header, set no_header "
                "to true.");
    fmt.header_row(starting_row);
    // Throw if row length (number of columns per row) varies across rows
    fmt.variable_columns(csv::VariableColumnPolicy::THROW);
  }

  return fmt;
}

template <typename T>
std::vector<unsigned int>
CSVPrimitiveTensor<T>::parse_indices(const csv::CSVReader & csv) const
{
  const auto & options = this->input_options();

  // column_names and column_indices are mutually exclusive
  if (options.user_specified("column_names") && options.user_specified("column_indices"))
    throw NEMLException("Only one of column_names or column_indices can be set.");

  // If there is no header row, column_names cannot be used
  if (options.template get<bool>("no_header") && options.user_specified("column_names"))
    throw NEMLException(
        "no_header is set to true, column_names cannot be used. Use column_indices instead.");

  // Column indices to read
  std::vector<unsigned int> indices;

  // If neither column_names nor column_indices is set, use all columns
  if (!options.user_specified("column_names") && !options.user_specified("column_indices"))
    return indices;

  const unsigned int base_size = utils::storage_size(T::const_base_sizes);
  // If column_names is set, get indices from names
  if (options.user_specified("column_names"))
  {
    const auto col_names = options.template get<std::vector<std::string>>("column_names");
    if (col_names.size() != base_size)
      throw NEMLException("Number of column_names provided (" + std::to_string(col_names.size()) +
                          ") does not match the expected number of components in " +
                          utils::demangle(typeid(T).name()).substr(7) + " (" +
                          std::to_string(base_size) + ").");
    const auto all_col_names = csv.get_col_names();

    for (const auto & name : col_names)
    {
      const auto it = std::find(all_col_names.begin(), all_col_names.end(), name);
      if (it != all_col_names.end())
        indices.push_back(std::distance(all_col_names.begin(), it));
      else
      {
        std::stringstream ss;
        for (const auto & col_name : all_col_names)
          ss << col_name << " ";
        throw NEMLException("Column name " + name +
                            " does not exist in CSV file. Available columns are: " + ss.str());
      }
    }
  }

  // If column_indices is set, use them directly
  if (options.user_specified("column_indices"))
  {
    indices = options.template get<std::vector<unsigned int>>("column_indices");
    if (indices.size() != base_size)
      throw NEMLException("Number of column_indices provided (" + std::to_string(indices.size()) +
                          ") does not match the expected number of components in " +
                          utils::demangle(typeid(T).name()).substr(7) + " (" +
                          std::to_string(base_size) + ").");
  }

  return indices;
}

template <typename T>
std::vector<T>
CSVPrimitiveTensor<T>::read_all(csv::CSVReader & csv, std::size_t & nrow) const
{
  const auto no_header = input_options().template get<bool>("no_header");
  std::vector<T> vals;
  for (const auto & row : csv)
  {
    if (csv.n_rows() == 1)
    {
      const unsigned int base_size = utils::storage_size(T::const_base_sizes);
      if (row.size() != base_size)
        throw NEMLException("Number of columns provided (" + std::to_string(row.size()) +
                            ") does not match the expected number of components in " +
                            utils::demangle(typeid(T).name()).substr(7) + " (" +
                            std::to_string(base_size) + ").");
    }
    std::vector<double> csv_row;
    for (unsigned int i = 0; i < row.size(); i++)
    {
      if (no_header)
        neml_assert(row[i].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", in column with index ",
                    i,
                    nrow == 0 ? ". Did you mistakenly set no_header=true?" : ".");
      else
        neml_assert(row[i].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", column ",
                    csv.get_col_names()[i]);
      csv_row.push_back(row[i].get<double>());
    }
    auto csv_row_tens = T::create(csv_row);
    vals.push_back(csv_row_tens);
    nrow++;
  }

  return vals;
}

template <>
std::vector<SR2>
CSVPrimitiveTensor<SR2>::read_all(csv::CSVReader & csv, std::size_t & nrow) const
{
  const auto no_header = input_options().get<bool>("no_header");
  std::vector<double> factor = {1.0, 1.0, 1.0, std::sqrt(2), std::sqrt(2), std::sqrt(2)};
  std::vector<SR2> vals;
  for (const auto & row : csv)
  {
    if (csv.n_rows() == 1)
    {
      if (row.size() != 6)
        throw NEMLException("Number of columns provided (" + std::to_string(row.size()) +
                            ") does not match the expected number of components in SR2 (6).");
    }
    std::vector<double> csv_row;
    for (unsigned int i = 0; i < 6; i++)
    {
      if (no_header)
        neml_assert(row[i].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", in column with index ",
                    i,
                    nrow == 0 ? ". Did you mistakenly set no_header=true?" : ".");
      else
        neml_assert(row[i].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", column ",
                    csv.get_col_names()[i]);
      csv_row.push_back(row[i].get<double>() * factor[i]);
    }
    auto csv_row_tens = SR2::create(csv_row);
    vals.push_back(csv_row_tens);
    nrow++;
  }

  return vals;
}

template <typename T>
std::vector<T>
CSVPrimitiveTensor<T>::read_by_indices(csv::CSVReader & csv,
                                       const std::vector<unsigned int> & indices,
                                       std::size_t & nrow) const
{
  const auto no_header = input_options().template get<bool>("no_header");
  std::vector<T> vals;
  for (const auto & row : csv)
  {
    std::vector<double> csv_row;
    for (const auto & idx : indices)
    {
      if (csv.n_rows() == 1)
      {
        neml_assert(idx < row.size(),
                    "Column index ",
                    idx,
                    " is out of bounds. The CSV file has ",
                    row.size(),
                    " columns.");
      }
      if (no_header)
        neml_assert(row[idx].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", in column with index ",
                    idx,
                    nrow == 0 ? ". Did you mistakenly set no_header=true?" : ".");
      else
        neml_assert(row[idx].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", column ",
                    csv.get_col_names()[idx]);
      csv_row.push_back(row[idx].get<double>());
    }
    auto csv_row_tens = T::create(csv_row);
    vals.push_back(csv_row_tens);
    nrow++;
  }

  return vals;
}

template <>
std::vector<SR2>
CSVPrimitiveTensor<SR2>::read_by_indices(csv::CSVReader & csv,
                                         const std::vector<unsigned int> & indices,
                                         std::size_t & nrow) const
{
  const auto no_header = input_options().get<bool>("no_header");
  std::vector<double> factor = {1.0, 1.0, 1.0, std::sqrt(2), std::sqrt(2), std::sqrt(2)};
  std::vector<SR2> vals;
  for (const auto & row : csv)
  {
    std::vector<double> csv_row;
    for (unsigned int i = 0; i < 6; i++)
    {
      if (csv.n_rows() == 1)
      {
        neml_assert(indices[i] < row.size(),
                    "Column index ",
                    indices[i],
                    " is out of bounds. The CSV file has ",
                    row.size(),
                    " columns.");
      }
      if (no_header)
        neml_assert(row[indices[i]].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", in column with index ",
                    indices[i],
                    nrow == 0 ? ". Did you mistakenly set no_header=true?" : ".");
      else
        neml_assert(row[indices[i]].is_num(),
                    "Non-numeric value found in CSV file at row ",
                    nrow,
                    ", column ",
                    csv.get_col_names()[indices[i]]);
      csv_row.push_back(row[indices[i]].get<double>() * factor[i]);
    }
    auto csv_row_tens = SR2::create(csv_row);
    vals.push_back(csv_row_tens);
    nrow++;
  }

  return vals;
}

#define CSVPRIMITIVETENSOR_REGISTER(T)                                                             \
  using CSV##T = CSVPrimitiveTensor<T>;                                                            \
  register_NEML2_object_alias(CSV##T, "CSV" #T)
CSVPRIMITIVETENSOR_REGISTER(Scalar);
CSVPRIMITIVETENSOR_REGISTER(Vec);
CSVPRIMITIVETENSOR_REGISTER(SR2);
} // namespace neml2

#endif