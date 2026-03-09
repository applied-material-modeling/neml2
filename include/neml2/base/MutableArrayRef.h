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

#include <cstddef>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <vector>

#include "neml2/misc/assertions.h"

namespace neml2
{
/**
 * @brief A mutable, non-owning reference to a contiguous array.
 *
 * This type mirrors c10::ArrayRef but allows modifying referenced elements.
 * The memory is owned elsewhere, and must outlive this reference.
 */
template <typename T>
class MutableArrayRef final
{
public:
  using iterator = T *;
  using const_iterator = const T *;
  using size_type = std::size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
  T * _data;
  size_type _length;

  void debug_check_nullptr_invariant() const
  {
    neml_assert_dbg(_data != nullptr || _length == 0,
                    "created MutableArrayRef with nullptr and non-zero length!");
  }

public:
  /// Construct an empty MutableArrayRef.
  constexpr MutableArrayRef()
    : _data(nullptr),
      _length(0)
  {
  }

  /// Construct a MutableArrayRef from a single element.
  constexpr MutableArrayRef(T & one_elt)
    : _data(&one_elt),
      _length(1)
  {
  }

  /// Construct from pointer and length.
  constexpr MutableArrayRef(T * data, size_t length)
    : _data(data),
      _length(length)
  {
    debug_check_nullptr_invariant();
  }

  /// Construct from range [begin, end).
  constexpr MutableArrayRef(T * begin, T * end)
    : _data(begin),
      _length(end - begin)
  {
    debug_check_nullptr_invariant();
  }

  template <typename Container,
            typename U = decltype(std::declval<Container>().data()),
            typename = std::enable_if_t<std::is_same_v<U, T *>>>
  MutableArrayRef(Container & container)
    : _data(container.data()),
      _length(container.size())
  {
    debug_check_nullptr_invariant();
  }

  constexpr iterator begin() const { return _data; }
  constexpr iterator end() const { return _data + _length; }
  constexpr const_iterator cbegin() const { return _data; }
  constexpr const_iterator cend() const { return _data + _length; }

  constexpr reverse_iterator rbegin() const { return reverse_iterator(end()); }
  constexpr reverse_iterator rend() const { return reverse_iterator(begin()); }
  constexpr const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
  constexpr const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

  constexpr bool empty() const { return _length == 0; }
  constexpr T * data() const { return _data; }
  constexpr size_t size() const { return _length; }

  constexpr T & front() const
  {
    neml_assert(!empty(), "MutableArrayRef: attempted to access front() of empty list");
    return _data[0];
  }

  constexpr T & back() const
  {
    neml_assert(!empty(), "MutableArrayRef: attempted to access back() of empty list");
    return _data[_length - 1];
  }

  constexpr MutableArrayRef<T> slice(size_t n, size_t m) const
  {
    neml_assert(n + m <= size(),
                "MutableArrayRef: invalid slice, n = ",
                n,
                "; m = ",
                m,
                "; size = ",
                size());
    return MutableArrayRef<T>(data() + n, m);
  }

  constexpr MutableArrayRef<T> slice(size_t n) const
  {
    neml_assert(n <= size(), "MutableArrayRef: invalid slice, n = ", n, "; size = ", size());
    return slice(n, size() - n);
  }

  constexpr T & operator[](size_t index) const { return _data[index]; }

  constexpr T & at(size_t index) const
  {
    neml_assert(
        index < _length, "MutableArrayRef: invalid index, index = ", index, "; length = ", _length);
    return _data[index];
  }

  /// Disallow accidental assignment from a temporary.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MutableArrayRef<T>> & operator=(U && temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MutableArrayRef<T>> &
  operator=(std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(_data, _data + _length); }
};

template <typename T>
std::ostream &
operator<<(std::ostream & out, MutableArrayRef<T> list)
{
  int i = 0;
  out << "[";
  for (const auto & e : list)
  {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}
} // namespace neml2
