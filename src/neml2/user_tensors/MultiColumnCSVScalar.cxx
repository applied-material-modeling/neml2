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
  options.doc() =
      "Construct a Scalar from a CSV file by column. One must set either the column_names or "
      "column_indices of the columns to be read, as well as the batch_shape to reshape the read "
      "values to. By "
      "default, it is assumed that there is a header row and the delimiter is a comma. These can "
      "be modified, though the only currently supported delimiter types are comma, semicolon, "
      "space, and tab.";

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Path to the CSV file";

  options.set<TensorShape>("batch_shape");
  options.set("batch_shape").doc() = "Batch shape";

  options.set<std::vector<std::string>>("column_names") = {};
  options.set("column_names").doc() = "Names of CSV columns.";

  options.set<std::vector<unsigned int>>("column_indices") = {};
  options.set("column_indices").doc() = "Indices of CSV columns.";

  EnumSelection delimiter_selection(
      {"COMMA", "SEMICOLON", "SPACE", "TAB"}, {',', ';', ' ', '\t'}, "COMMA");
  options.set<EnumSelection>("delimiter") = delimiter_selection;
  options.set("delimiter").doc() =
      "Delimiter used to parse the CSV file. Options are " + delimiter_selection.candidates_str();

  options.set<bool>("header_row") = true;
  options.set("header_row").doc() = "Whether the CSV file has a header row. The default is true.";

  return options;
}

MultiColumnCSVScalar::MultiColumnCSVScalar(const OptionSet & options)
  : Scalar(parse_csv(options)),
    UserTensorBase(options)
{
}

void
MultiColumnCSVScalar::check_col_options(const OptionSet & options, bool header_row) const
{
  if (options.user_specified("column_names") && options.user_specified("column_indices"))
    throw NEMLException("Only one of column_names or column_indices can be set.");
  else if (options.user_specified("column_names") && !header_row)
    throw NEMLException("If there is no header row, column_names cannot be used. Use "
                        "column_indices intead.");
  else if (!options.user_specified("column_names") && !options.user_specified("column_indices"))
    throw NEMLException("Either column_names or column_indices must be set.");
}

std::vector<double>
MultiColumnCSVScalar::flatten(const std::vector<std::vector<double>> v2d) const
{
  if (v2d.empty())
    return {};

  auto rows = v2d.size();
  auto cols = v2d[0].size();

  std::vector<double> flat(rows * cols);

  for (std::size_t i = 0; i < rows; ++i)
    std::copy(v2d[i].begin(), v2d[i].end(), flat.begin() + i * cols);

  return flat;
}

Scalar
MultiColumnCSVScalar::parse_csv(const OptionSet & options) const
{
  bool header_row = options.get<bool>("header_row");
  check_col_options(options, header_row);

  std::string csv_file = options.get<std::string>("csv_file");
  TensorShape batch_shape = options.get<TensorShape>("batch_shape");
  std::vector<std::string> column_names = options.get<std::vector<std::string>>("column_names");
  std::vector<unsigned int> column_indices =
      options.get<std::vector<unsigned int>>("column_indices");
  auto delim = options.get<EnumSelection>("delimiter").as<char>();

  // Allow for no header row
  csv::CSVFormat format;
  if (!header_row)
  {
    format.delimiter(delim).no_header();
  }
  else
    format.delimiter(delim);

  csv::CSVReader counter(csv_file, format);
  csv::CSVReader reader(csv_file, format);

  // Vector of vectors to hold CSV values
  std::vector<std::vector<double>> csv_vals;

  // Set number of rows in csv_vals to equal number of columns specified by user
  if (options.user_specified("column_names"))
  {
    std::vector<std::string> all_col_names = counter.get_col_names();
    for (auto & j : column_names)
    {
      // Check if column name exists
      auto it = find(all_col_names.begin(), all_col_names.end(), j);
      if (it == all_col_names.end())
        throw NEMLException("Column name " + j +
                            " does not exist in CSV file, or the incorrect delimiter may have been "
                            "chosen to parse the CSV file.");
    }
    csv_vals.resize(column_names.size());
  }
  else if (options.user_specified("column_indices"))
    csv_vals.resize(column_indices.size());

  // Set number of columns in csv_vals to equal number of rows in csv
  for (auto & row : counter)
    (void)row; // counts total number of rows in csv
  for (auto & row : csv_vals)
    row.resize(counter.n_rows());

  // Read CSV values
  int row_count = 0;
  int col_count = 0;
  for (auto & row : reader)
  {
    if (options.user_specified("column_names"))
    {
      for (auto & j : column_names)
      {
        csv_vals[row_count][col_count] = row[j].get<double>();
        row_count += 1;
      }
    }
    else if (options.user_specified("column_indices"))
    {
      for (auto & j : column_indices)
      {
        csv_vals[row_count][col_count] = row[j].get<double>();
        row_count += 1;
      }
    }
    col_count += 1;
    row_count = 0;
  }

  // Convert to Scalar and reshape
  std::vector<double> csv_flatten = flatten(csv_vals);
  auto csv_tensor = Scalar::create(csv_flatten).batch_reshape(batch_shape);

  return csv_tensor;
}
} // namespace neml2

#endif