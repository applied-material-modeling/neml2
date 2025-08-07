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

#include "neml2/base/LabeledAxis.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
LabeledAxis::LabeledAxis(LabeledAxisAccessor prefix)
  : _prefix(std::move(prefix))
{
}

LabeledAxisAccessor
LabeledAxis::qualify(const LabeledAxisAccessor & accessor) const
{
  return accessor.prepend(_prefix);
}

LabeledAxis &
LabeledAxis::add_subaxis(const std::string & name)
{
  neml_assert(!_setup, "Cannot modify a sub-axis after the axis has been set up.");
  neml_assert(
      _variables.count(name) == 0, "Cannot add a subaxis with the same name as a variable: ", name);
  auto [subaxis, success] =
      _subaxes.emplace(name, std::make_shared<LabeledAxis>(_prefix.append(name)));
  if (success)
    cache_reserved_subaxis(name);
  return *(subaxis->second);
}

void
LabeledAxis::add_variable(const LabeledAxisAccessor & name,
                          TensorShapeRef base_sizes,
                          TensorShapeRef lbatch_sizes)
{
  neml_assert(!_setup, "Cannot modify a sub-axis after the axis has been set up.");
  neml_assert(!name.empty(), "Cannot add a variable with empty name.");

  if (name.size() == 1)
  {
    neml_assert(_variables.count(name[0]) == 0 && _subaxes.count(name[0]) == 0,
                "Cannot add a variable with the same name as an existing variable or a sub-axis: '",
                name[0],
                "'");
    _variables.emplace(name[0], std::make_pair(base_sizes, lbatch_sizes));
  }
  else
    add_subaxis(name[0]).add_variable(name.slice(1), base_sizes, lbatch_sizes);
}

template <typename T>
void
LabeledAxis::add_variable(const LabeledAxisAccessor & name, TensorShapeRef lbatch_sizes)
{
  add_variable(name, T::const_base_sizes, lbatch_sizes);
}
#define INSTANTIATE_ADD_VARIABLE(T)                                                                \
  template void LabeledAxis::add_variable<T>(const LabeledAxisAccessor &, TensorShapeRef)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_ADD_VARIABLE);

static Size
calculate_size(TensorShapeRef base_sizes, TensorShapeRef lbatch_sizes)
{
  return utils::storage_size(base_sizes) * utils::storage_size(lbatch_sizes);
}

void
LabeledAxis::setup_layout()
{
  // Clear internal data that may have been constructed from previous setup_layout calls
  _size = 0;

  _variable_to_id_map.clear();
  _id_to_variable_map.clear();
  _id_to_variable_size_map.clear();
  _id_to_variable_slice_map.clear();
  _id_to_base_sizes_map.clear();
  _id_to_lbatch_sizes_map.clear();

  _sorted_subaxes.clear();
  _subaxis_to_id_map.clear();
  _id_to_subaxis_map.clear();
  _id_to_subaxis_size_map.clear();
  _id_to_subaxis_slice_map.clear();

  // Set up variable assembly IDs and slicing indices
  for (const auto & [name, sizes] : _variables)
  {
    const auto & [base_sizes, lbatch_sizes] = sizes;
    const auto sz = calculate_size(base_sizes, lbatch_sizes);
    _variable_to_id_map.emplace(name, _variable_to_id_map.size());
    _id_to_variable_map.emplace_back(name);
    _id_to_variable_size_map.push_back(sz);
    _id_to_variable_slice_map.emplace_back(_size, _size + sz);
    _id_to_base_sizes_map.push_back(base_sizes);
    _id_to_lbatch_sizes_map.push_back(lbatch_sizes);
    _size += sz;
  }

  // Set up subaxes
  for (auto & [name, axis] : _subaxes)
  {
    axis->setup_layout();
    auto sz = axis->size();
    _sorted_subaxes.push_back(axis.get());
    _subaxis_to_id_map.emplace(name, _subaxis_to_id_map.size());
    _id_to_subaxis_map.push_back(name);
    _id_to_subaxis_size_map.push_back(sz);
    _id_to_subaxis_slice_map.emplace_back(_size, _size + sz);
    _size += sz;

    // Merge variable maps
    for (const auto & var_name : axis->_id_to_variable_map)
    {
      auto var_id = axis->_variable_to_id_map.at(var_name);
      auto full_name = var_name.prepend(name);
      _variable_to_id_map.emplace(full_name, _variable_to_id_map.size());
      _id_to_variable_map.push_back(full_name);
      _id_to_variable_size_map.push_back(axis->_id_to_variable_size_map[var_id]);
      _id_to_base_sizes_map.push_back(axis->_id_to_base_sizes_map[var_id]);
      _id_to_lbatch_sizes_map.push_back(axis->_id_to_lbatch_sizes_map[var_id]);

      // Slice is relative to the sub-axis, so we need to shift it
      const auto & slice = axis->_id_to_variable_slice_map[var_id];
      auto offset = _id_to_subaxis_slice_map.back().first;
      auto new_slice = std::pair<Size, Size>{slice.first + offset, slice.second + offset};
      _id_to_variable_slice_map.push_back(new_slice);
    }
  }

  // Finished set up
  _setup = true;
}

Size
LabeledAxis::size() const
{
  // If the axis has been set up, return the cached size
  if (_setup)
    return _size;

  // Otherwise, calculate the size
  Size sz = 0;
  for (const auto & [name, sizes] : _variables)
    sz += calculate_size(sizes.first, sizes.second);
  for (const auto & [name, axis] : _subaxes)
    sz += axis->size();
  return sz;
}

Size
LabeledAxis::size(const LabeledAxisAccessor & name) const
{
  neml_assert(!name.empty(), "Cannot get the size of an item with an empty name.");

  // If the name has length 1, it must be a variable or a local sub-axis
  if (name.size() == 1)
  {

    const auto var = _variables.find(name[0]);
    if (var != _variables.end())
    {
      if (_setup)
      {
        const auto id = variable_id(name);
        return _id_to_variable_size_map[id];
      }
      else
      {
        const auto & [base_sizes, lbatch_sizes] = var->second;
        return calculate_size(base_sizes, lbatch_sizes);
      }
    }

    const auto subaxis = _subaxes.find(name[0]);
    neml_assert(subaxis != _subaxes.end(),
                "Item named '",
                name,
                "' is neither a variable nor a local sub-axis on axis:\n",
                *this);
    if (_setup)
    {
      const auto id = subaxis_id(name[0]);
      return _id_to_subaxis_size_map[id];
    }
    return subaxis->second->size();
  }

  // Otherwise, the item must be on a sub-axis
  const auto subaxis = _subaxes.find(name[0]);
  neml_assert(subaxis != _subaxes.end(),
              "Item named '",
              name,
              "' is neither a variable nor a sub-axis on axis:\n",
              *this);
  return subaxis->second->size(name.slice(1));
}

indexing::Slice
LabeledAxis::slice(const LabeledAxisAccessor & name) const
{
  ensure_setup_dbg();
  neml_assert(!name.empty(), "Cannot get the slice of an item with an empty name.");

  // If the name is a variable
  if (has_variable(name))
  {
    auto s = variable_slice(name);
    return {s.first, s.second};
  }

  // Otherwise, the name must be a sub-axis
  neml_assert_dbg(has_subaxis(name[0]),
                  "Item named '",
                  name,
                  "' is neither a variable nor a sub-axis on axis:\n",
                  *this);
  auto s = subaxis_slice(name);
  return {s.first, s.second};
}

std::size_t
LabeledAxis::nvariable() const
{
  // If axis has been set up, return the cached number of variables
  if (_setup)
    return _id_to_variable_map.size();

  // Otherwise, calculate the number of variables
  std::size_t nvar = _variables.size();
  for (const auto & [name, axis] : _subaxes)
    nvar += axis->nvariable();
  return nvar;
}

bool
LabeledAxis::has_variable(const LabeledAxisAccessor & name) const
{
  neml_assert(!name.empty(), "Variable name cannot be empty.");

  // If axis has been set up, return the cached existence
  if (_setup)
    return std::find(_id_to_variable_map.begin(), _id_to_variable_map.end(), name) !=
           _id_to_variable_map.end();

  // Otherwise, check the existence of the variable
  if (name.size() == 1)
    return _variables.find(name[0]) != _variables.end();

  const auto subaxis = _subaxes.find(name[0]);
  return subaxis != _subaxes.end() && subaxis->second->has_variable(name.slice(1));
}

std::size_t
LabeledAxis::variable_id(const LabeledAxisAccessor & name) const
{
  ensure_setup_dbg();
  neml_assert(!name.empty(), "Cannot get the ID of a variable with an empty name.");
  const auto id = _variable_to_id_map.find(name);
  neml_assert(id != _variable_to_id_map.end(),
              "Variable named '",
              name,
              "' does not exist on axis:\n",
              *this);
  return id->second;
}

const std::vector<LabeledAxisAccessor> &
LabeledAxis::variable_names() const
{
  ensure_setup_dbg();
  return _id_to_variable_map;
}

const std::vector<std::pair<Size, Size>> &
LabeledAxis::variable_slices() const
{
  ensure_setup_dbg();
  return _id_to_variable_slice_map;
}

const std::pair<Size, Size> &
LabeledAxis::variable_slice(const LabeledAxisAccessor & name) const
{
  ensure_setup_dbg();
  return _id_to_variable_slice_map.at(variable_id(name));
}

const std::vector<Size> &
LabeledAxis::variable_sizes() const
{
  ensure_setup_dbg();
  return _id_to_variable_size_map;
}

Size
LabeledAxis::variable_size(const LabeledAxisAccessor & name) const
{
  // If axis has been set up, return the cached variable size
  if (_setup)
    return _id_to_variable_size_map[variable_id(name)];

  // Otherwise, calculate the variable size
  if (name.size() == 1)
  {
    const auto var = _variables.find(name[0]);
    neml_assert(
        var != _variables.end(), "Variable named '", name, "' does not exist on axis:\n", *this);
    const auto & [base_sizes, lbatch_sizes] = var->second;
    return calculate_size(base_sizes, lbatch_sizes);
  }

  const auto subaxis = _subaxes.find(name[0]);
  neml_assert(
      subaxis != _subaxes.end(), "Variable named '", name, "' does not exist on axis:\n", *this);
  return subaxis->second->variable_size(name.slice(1));
}

const std::vector<TensorShape> &
LabeledAxis::variable_base_sizes() const
{
  ensure_setup_dbg();
  return _id_to_base_sizes_map;
}

const std::vector<TensorShape> &
LabeledAxis::variable_lbatch_sizes() const
{
  ensure_setup_dbg();
  return _id_to_lbatch_sizes_map;
}

std::size_t
LabeledAxis::nsubaxis() const
{
  return _subaxes.size();
}

bool
LabeledAxis::has_subaxis(const LabeledAxisAccessor & name) const
{
  neml_assert(!name.empty(), "Sub-axis name cannot be empty.");

  const auto subaxis = _subaxes.find(name[0]);

  if (name.size() == 1)
    return subaxis != _subaxes.end();

  return subaxis->second->has_subaxis(name.slice(1));
}

std::size_t
LabeledAxis::subaxis_id(const std::string & name) const
{
  ensure_setup_dbg();
  const auto id = _subaxis_to_id_map.find(name);
  neml_assert(id != _subaxis_to_id_map.end(),
              "Sub-axis named '",
              name,
              "' does not exist on axis:\n",
              *this);
  return id->second;
}

const std::vector<const LabeledAxis *> &
LabeledAxis::subaxes() const
{
  ensure_setup_dbg();
  return _sorted_subaxes;
}

const LabeledAxis &
LabeledAxis::subaxis(const LabeledAxisAccessor & name) const
{
  neml_assert(!name.empty(), "Sub-axis name cannot be empty.");

  const auto subaxis = _subaxes.find(name[0]);
  neml_assert(
      subaxis != _subaxes.end(), "Sub-axis named '", name, "' does not exist on axis:\n", *this);

  if (name.size() == 1)
    return *subaxis->second;

  return subaxis->second->subaxis(name.slice(1));
}

LabeledAxis &
LabeledAxis::subaxis(const LabeledAxisAccessor & name)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<LabeledAxis &>(std::as_const(*this).subaxis(name));
}

const std::vector<std::string> &
LabeledAxis::subaxis_names() const
{
  ensure_setup_dbg();
  return _id_to_subaxis_map;
}

const std::vector<std::pair<Size, Size>> &
LabeledAxis::subaxis_slices() const
{
  ensure_setup_dbg();
  return _id_to_subaxis_slice_map;
}

std::pair<Size, Size>
LabeledAxis::subaxis_slice(const LabeledAxisAccessor & name) const
{
  ensure_setup_dbg();

  // If the name has length 1, it must be a local sub-axis
  if (name.size() == 1)
    return _id_to_subaxis_slice_map[subaxis_id(name[0])];

  // Otherwise, the name must be on a sub-axis
  const auto subaxis = _subaxes.find(name[0]);
  neml_assert(
      subaxis != _subaxes.end(), "Sub-axis named '", name, "' does not exist on axis:\n", *this);
  const auto & slice = subaxis->second->subaxis_slice(name.slice(1));
  auto offset = _id_to_subaxis_slice_map[subaxis_id(name[0])].first;
  return {slice.first + offset, slice.second + offset};
}

const std::vector<Size> &
LabeledAxis::subaxis_sizes() const
{
  ensure_setup_dbg();
  return _id_to_subaxis_size_map;
}

Size
LabeledAxis::subaxis_size(const LabeledAxisAccessor & name) const
{
  const auto subaxis = _subaxes.find(name[0]);
  neml_assert(
      subaxis != _subaxes.end(), "Sub-axis named '", name, "' does not exist on axis:\n", *this);

  if (name.size() == 1)
    return subaxis->second->size();

  return subaxis->second->subaxis_size(name.slice(1));
}

bool
LabeledAxis::equals(const LabeledAxis & other) const
{
  // They must have the same set of variables (with the same storage sizes)
  if (_variables != other._variables)
    return false;

  // They must have the same number of subaxes
  if (_subaxes.size() != other._subaxes.size())
    return false;

  // Compare each subaxis
  for (const auto & [name, axis] : _subaxes)
  {
    if (other._subaxes.count(name) == 0)
      return false;

    if (*other._subaxes.at(name) != *axis)
      return false;
  }

  return true;
}

void
LabeledAxis::cache_reserved_subaxis(const std::string & axis_name)
{
  if (axis_name == STATE)
    _has_state = true;
  else if (axis_name == OLD_STATE)
    _has_old_state = true;
  else if (axis_name == FORCES)
    _has_forces = true;
  else if (axis_name == OLD_FORCES)
    _has_old_forces = true;
  else if (axis_name == RESIDUAL)
    _has_residual = true;
  else if (axis_name == PARAMETERS)
    _has_parameters = true;
}

void
LabeledAxis::ensure_setup_dbg() const
{
  neml_assert_dbg(_setup, "The axis has not been setup yet.");
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxis & axis)
{
  // Get unqualified variable names
  const auto var_names = axis.variable_names();

  // Find the maximum variable name length
  size_t max_var_name_length = 0;
  for (const auto & var_name : var_names)
  {
    const auto var_name_str = utils::stringify(var_name);
    if (var_name_str.size() > max_var_name_length)
      max_var_name_length = var_name_str.size();
  }

  // Print variables with right alignment
  for (auto var = var_names.begin(); var != var_names.end(); var++)
  {
    if (axis._setup)
      os << std::setw(3) << std::right << axis.variable_id(*var) << ": ";
    os << std::setw(int(max_var_name_length)) << std::left << utils::stringify(*var);
    if (axis._setup)
      os << " [" << axis.variable_slice(*var) << "]";
    if (std::next(var) != var_names.end())
      os << std::endl;
  }

  return os;
}

bool
operator==(const LabeledAxis & a, const LabeledAxis & b)
{
  return a.equals(b);
}

bool
operator!=(const LabeledAxis & a, const LabeledAxis & b)
{
  return !a.equals(b);
}
} // namespace neml2
