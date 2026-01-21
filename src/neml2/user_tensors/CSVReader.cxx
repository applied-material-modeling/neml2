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

#include "neml2/user_tensors/CSVReader.h"

#ifdef NEML2_CSV

namespace neml2
{
OptionSet
CSVReader::expected_options()
{
  OptionSet options;

  options.set<std::string>("csv_file");
  options.set("csv_file").doc() = "Path to the CSV file";

  options.set<std::vector<std::string>>("column_names") = {};
  options.set("column_names").doc() = "Names of CSV columns.";

  options.set<std::vector<unsigned int>>("column_indices") = {};
  options.set("column_indices").doc() = "Indices of CSV columns.";

  EnumSelection delimiter_selection(
      {"COMMA", "SEMICOLON", "SPACE", "TAB"}, {',', ';', ' ', '\t'}, "COMMA");
  options.set<EnumSelection>("delimiter") = delimiter_selection;
  options.set("delimiter").doc() =
      "Delimiter used to parse the CSV file. Options are " + delimiter_selection.join();

  options.set<bool>("no_header") = false;
  options.set("no_header").doc() = "Whether the CSV file has a header row.";

  options.set<int>("starting_row") = 0;
  options.set("starting_row").doc() =
      "Starting row of the CSV file (0-indexed). Rows before this row are ignored. This should be "
      "the header row if the CSV file has a header, otherwise it should be the first row of the "
      "data. By default the starting row is the 0th row.";

  return options;
}

CSVReader::CSVReader(const NEML2Object * obj)
  : _obj(obj)
{
}

csv::CSVFormat
CSVReader::parse_format() const
{
  const auto & options = _obj->input_options();
  csv::CSVFormat fmt;

  // Delimiter
  const auto delim = options.get<EnumSelection>("delimiter").as<char>();
  fmt.delimiter(delim);

  // no_header=true
  if (options.get<bool>("no_header"))
  {
    // Set 'header row' to one before starting row of data, so that we start reading data from the
    // starting row
    fmt.header_row(options.get<int>("starting_row") - 1);
    // This is a csvparser bug. It false-positively detects variable columns
    fmt.variable_columns(csv::VariableColumnPolicy::KEEP);
  }
  // no_header=false
  else
  {
    // Use starting_row as header row
    const auto starting_row = options.get<int>("starting_row");
    neml_assert(starting_row >= 0,
                "starting_row must be non-negative. If your CSV file has no header, set no_header "
                "to true.");
    fmt.header_row(starting_row);
    // Throw if row length (number of columns per row) varies across rows
    fmt.variable_columns(csv::VariableColumnPolicy::THROW);
  }

  return fmt;
}

std::vector<unsigned int>
CSVReader::parse_indices(const csv::CSVReader & csv) const
{
  const auto & options = _obj->input_options();

  // column_names and column_indices are mutually exclusive
  if (options.user_specified("column_names") && options.user_specified("column_indices"))
    throw NEMLException("Only one of column_names or column_indices can be set.");

  // If there is no header row, column_names cannot be used
  if (options.get<bool>("no_header") && options.user_specified("column_names"))
    throw NEMLException(
        "no_header is set to true, column_names cannot be used. Use column_indices instead.");

  // Column indices to read
  std::vector<unsigned int> indices;

  // If neither column_names nor column_indices is set, use all columns
  if (!options.user_specified("column_names") && !options.user_specified("column_indices"))
    return indices;

  // If column_names is set, get indices from names
  if (options.user_specified("column_names"))
  {
    const auto col_names = options.get<std::vector<std::string>>("column_names");
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
    indices = options.get<std::vector<unsigned int>>("column_indices");

  return indices;
}

void
CSVReader::read_all(csv::CSVReader & csv,
                    std::vector<double> & vals,
                    std::size_t & nrow,
                    std::size_t & ncol) const
{
  const auto no_header = _obj->input_options().get<bool>("no_header");
  for (const auto & row : csv)
  {
    if (nrow == 0)
      ncol = row.size();
    for (unsigned int i = 0; i < ncol; i++)
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
      vals.push_back(row[i].get<double>());
    }
    nrow++;
  }
}

void
CSVReader::read_by_indices(csv::CSVReader & csv,
                           const std::vector<unsigned int> & indices,
                           std::vector<double> & vals,
                           std::size_t & nrow,
                           std::size_t & ncol) const
{
  const auto no_header = _obj->input_options().get<bool>("no_header");
  ncol = indices.size();
  for (const auto & row : csv)
  {
    for (const auto & idx : indices)
    {
      if (nrow == 0)
        neml_assert(idx < row.size(),
                    "Column index ",
                    idx,
                    " is out of bounds. The CSV file has ",
                    row.size(),
                    " columns.");
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
      vals.push_back(row[idx].get<double>());
    }
    nrow++;
  }
}

} // namespace neml2

#endif
