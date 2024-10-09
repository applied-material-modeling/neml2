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

#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "neml2/misc/error.h"

namespace neml2
{
/**
 * Storage container that stores a vector of unique pointers of T, but represents most of the public
 * facing accessors (iterators, operator[]).
 *
 * That is, these accessors dereference the underlying storage. More importantly, if data is not
 * properly initialized using set_pointer(), this dereferencing will either lead to an assertion or
 * a nullptr dereference.
 */
template <typename I, typename T>
class Storage
{
public:
  Storage() = default;
  Storage(Storage &&) = default;
  Storage(const Storage &) = delete;
  Storage & operator=(const Storage &) = delete;

  /**
   * Iterator that adds an additional dereference to BaseIterator.
   */
  template <class BaseIterator>
  struct DereferenceIterator : public BaseIterator
  {
    DereferenceIterator(const BaseIterator & it)
      : BaseIterator(it)
    {
    }

    using key_type = typename BaseIterator::value_type::first_type;
    using value_type = typename BaseIterator::value_type::second_type::element_type;

    std::pair<key_type, value_type &> operator*() const
    {
      auto & [key, val] = BaseIterator::operator*();
      neml_assert_dbg(val.get(),
                      "Trying to dereference a null object. Make sure the storage was properly "
                      "initialized using set_pointer().");
      return {key, *val};
    }
    std::pair<key_type, value_type *> operator->() const
    {
      auto & [key, val] = BaseIterator::operator*();
      neml_assert_dbg(val.get(),
                      "Trying to dereference a null object. Make sure the storage was properly "
                      "initialized using set_pointer().");
      return {key, val.get()};
    }
  };

  using values_type = typename std::map<I, std::unique_ptr<T>>;
  using iterator = DereferenceIterator<typename values_type::iterator>;
  using const_iterator = DereferenceIterator<typename values_type::const_iterator>;

  /**
   * @name Iterators
   * Begin and end iterators to the underlying data.
   *
   * Note that dereferencing these iterators may lead to an assertion
   * or the dereference of a nullptr whether or not the underlying data
   * is initialized.
   */
  ///@{
  iterator begin() { return iterator(_values.begin()); }
  iterator end() { return iterator(_values.end()); }
  const_iterator begin() const { return const_iterator(_values.begin()); }
  const_iterator end() const { return const_iterator(_values.end()); }
  ///@}

  ///@{
  /**
   * @returns A reference to the underlying data at index \p i.
   *
   * Note that the underlying data may not necessarily be initialized,
   * in which case this will throw an assertion or dereference a nullptr.
   *
   * You can check whether or not the underlying data is intialized
   * with has_key(i).
   */
  T & operator[](const I & i) const
  {
    neml_assert_dbg(has_key(i),
                    "Trying to access a null object. Make sure the storage was properly "
                    "initialized using set_pointer().");
    return *pointer_value(i);
  }
  T & operator[](const I & i) { return std::as_const(*this)[i]; }
  ///@}

  /**
   * @returns The size of the underlying storage.
   *
   * Note that this is not necessarily the size of _constructed_ objects,
   * as underlying objects could be uninitialized
   */
  std::size_t size() const { return _values.size(); }

  /**
   * @returns Whether or not the underlying storage is empty.
   */
  bool empty() const { return _values.empty(); }

  /**
   * @returns whether or not the underlying object at index \p i is initialized
   */
  bool has_key(const I & i) const { return _values.count(i); }

  ///@{
  /**
   * @returns A pointer to the underlying data at index \p i
   *
   * The pointer will be nullptr if !has_key(i), that is, if the
   * unique_ptr at index \p i is not initialized
   */
  const T * query_value(const I & i) const { return has_key(i) ? pointer_value(i).get() : nullptr; }
  T * query_value(const I & i)
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return has_key(i) ? const_cast<T *>(std::as_const(*this).query_value(i)) : nullptr;
  }
  ///@}

  /**
   * Sets the underlying unique_ptr at index \p i to \p ptr.
   *
   * This can be used to construct objects in the storage, i.e.,
   * set_pointer(0, std::make_unique<T>(...));
   *
   * This is the only method that allows for the modification of
   * ownership in the underlying vector. Protect it wisely.
   */
  T * set_pointer(const I & i, std::unique_ptr<T> && ptr)
  {
    _values[i] = std::move(ptr);
    return _values[i].get();
  }

private:
  /**
   * Returns a read-only reference to the underlying unique pointer
   * at index \p i.
   *
   * We hope to only expose the underlying unique_ptr to this API,
   * and not in derived classes. Hopefully it can stay that way.
   */
  const std::unique_ptr<T> & pointer_value(const I & i) const { return _values.at(i); }

  /// The underlying data
  values_type _values;
};
} // namespace neml2
