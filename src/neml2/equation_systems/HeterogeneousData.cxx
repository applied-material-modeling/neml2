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

#include "neml2/equation_systems/HeterogeneousData.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/assertions.h"

namespace neml2
{

HeterogeneousData::HeterogeneousData(std::vector<Tensor> v, Preference p)
  : _disassembled_data(std::move(v)),
    _preference(p)
{
  neml_assert_dbg(!_disassembled_data.empty(), "HeterogeneousData: empty data vector.");
}

HeterogeneousData::HeterogeneousData(Tensor assembled, Preference p)
  : _assembled_data(std::move(assembled)),
    _preference(p)
{
  neml_assert_dbg(_assembled_data.defined(), "HeterogeneousData: undefined assembled data.");
}

bool
HeterogeneousData::disassembled() const
{
  return !_disassembled_data.empty();
}

bool
HeterogeneousData::assembled() const
{
  return _assembled_data.defined();
}

const Tensor &
HeterogeneousData::get_assembled() const
{
  if (!assembled())
    assemble();
  return _assembled_data;
}

const std::vector<Tensor> &
HeterogeneousData::get_disassembled() const
{
  if (!disassembled())
    disassemble();
  return _disassembled_data;
}

bool
HeterogeneousData::requires_grad() const
{
  if (assembled())
    return _assembled_data.requires_grad();
  for (const auto & vi : _disassembled_data)
    if (vi.defined() && vi.requires_grad())
      return true;
  return false;
}

TensorOptions
HeterogeneousData::options() const
{
  if (assembled())
    return _assembled_data.options();
  for (const auto & vi : _disassembled_data)
    if (vi.defined())
      return vi.options();
  throw NEMLException("unable to determine HVector/Matrix tensor options.");
}

} // namespace neml2
