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

#include "neml2/user_tensors/MultiColumnCSVScalar.h"

#ifdef NEML2_CSV

namespace neml2
{
register_NEML2_object(MultiColumnCSVScalar);

OptionSet
MultiColumnCSVScalar::expected_options()
{
  OptionSet options = UserTensorBase<Scalar>::expected_options();
  options += CSVReader::expected_options();
  options.doc() =
      "Construct a two-dimensional Scalar from a CSV file. A subset of columns can be selected "
      "using `column_indices` or `column_names`. By default, the CSV is interpreted as "
      "column-major, i.e., each column in the CSV corresponds to one row of the 2D Scalar. This "
      "behavior can be altered via the `indexing` option.";

  EnumSelection indexing_selection({"COLUMN_MAJOR", "ROW_MAJOR"}, "COLUMN_MAJOR");
  options.set<EnumSelection>("indexing") = indexing_selection;
  options.set("indexing").doc() =
      "Indexing interpretation. Options are " + indexing_selection.join();

  return options;
}

MultiColumnCSVScalar::MultiColumnCSVScalar(const OptionSet & options)
  : UserTensorBase<Scalar>(options),
    CSVReader(this)
{
}

Scalar
MultiColumnCSVScalar::make() const
{
  const auto & options = input_options();

  // Parse CSV
  const auto fmt = parse_format();
  csv::CSVReader csv(options.get<std::string>("csv_file"), fmt);
  const auto indices = parse_indices(csv);

  // CSV data
  std::vector<double> vals;
  std::size_t nrow = 0, ncol = 0;

  if (indices.empty())
    read_all(csv, vals, nrow, ncol);
  else
    read_by_indices(csv, indices, vals, nrow, ncol);

  // Convert to Scalar and reshape
  auto csv_scalar = Scalar::create(vals).dynamic_reshape({Size(nrow), Size(ncol)});
  if (options.get<EnumSelection>("indexing") == "COLUMN_MAJOR")
    csv_scalar = csv_scalar.dynamic_transpose(0, 1);

  return csv_scalar;
}

} // namespace neml2

#endif // NEML2_CSV
