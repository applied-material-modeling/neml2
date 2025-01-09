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

#include <utility>
#include <cstddef>

namespace neml2
{
template <typename T>
class WorkGenerator
{
public:
  /**
   * @brief Generate the next \p n batches of work
   *
   * Note that in the case of insufficient remaining work, it is possible that the number of batches
   * generated is less than \p n.
   *
   * This is the public interface to the generator. Derived classes should implement
   * WorkGenerator::generate.
   *
   * @param n Number of batches to generator
   * @return std::pair<std::size_t, T> Number of batches generated (\p m) and the next \p m batches
   * of work
   */
  std::pair<std::size_t, T> next(std::size_t n)
  {
    auto && [m, work] = generate(n);
    _offset += m;
    return std::make_pair(m, work);
  }

  /// @brief Return the current offset, i.e., the number of batches that have been generated
  std::size_t offset() const { return _offset; }

  /// @brief Whether the generator has more work to generate
  virtual bool has_more() const = 0;

protected:
  /**
   * @brief Generate the next \p n batches of work
   *
   * Note that in the case of insufficient remaining work, it is possible that the number of batches
   * generated is less than \p n.
   *
   * @param n Number of batches to generate
   * @return std::pair<std::size_t, T> Number of batches generated (\p m) and the next \p m batches
   * of work
   */
  virtual std::pair<std::size_t, T> generate(std::size_t n) = 0;

private:
  std::size_t _offset = 0;
};
} // namespace neml2
