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

#pragma once

#include "csvparser/csv.hpp"
#include "neml2/user_tensors/UserTensorBase.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
class MultiColumnCSVScalar : public UserTensorBase, public Scalar
{
public:
  static OptionSet expected_options();

  MultiColumnCSVScalar(const OptionSet & options);

private:
  Scalar parse(const OptionSet & options) const;

  /// Helper functions to parse CSV format
  csv::CSVFormat parse_format() const;

  /// Helper function to parse column indices
  std::vector<unsigned int> parse_indices(const csv::CSVReader & csv) const;

  /// Read all values without column indices
  void read_all(csv::CSVReader & csv,
                std::vector<double> & vals,
                std::size_t & nrow,
                std::size_t & ncol) const;

  /// Read values by specified column indices
  void read_by_indices(csv::CSVReader & csv,
                       const std::vector<unsigned int> & indices,
                       std::vector<double> & vals,
                       std::size_t & nrow,
                       std::size_t & ncol) const;
};
} // namespace neml2

#endif
