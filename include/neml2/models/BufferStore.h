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

#include "neml2/base/OptionSet.h"
#include "neml2/jit/types.h"

namespace neml2
{
class NEML2Object;
struct TensorName;
class TensorValueBase;
template <typename T>
class TensorBase;

/// Interface for object which can store buffers
class BufferStore
{
public:
  BufferStore(OptionSet options, NEML2Object * object);

  BufferStore(const BufferStore &) = delete;
  BufferStore(BufferStore &&) = delete;
  BufferStore & operator=(const BufferStore &) = delete;
  BufferStore & operator=(BufferStore &&) = delete;
  virtual ~BufferStore() = default;

  ///@{
  /// @returns the buffer storage
  const std::map<std::string, std::unique_ptr<TensorValueBase>> & named_buffers() const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<BufferStore *>(this)->named_buffers();
  }
  std::map<std::string, std::unique_ptr<TensorValueBase>> & named_buffers();
  ///}@

  /// Get a read-only reference of a buffer
  const TensorValueBase & get_buffer(const std::string & name) const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<BufferStore *>(this)->get_buffer(name);
  }
  /// Get a writable reference of a buffer
  TensorValueBase & get_buffer(const std::string & name);

protected:
  /**
   * @brief Send all buffers to \p options
   *
   * @param options The target options
   */
  virtual void send_buffers_to(const TensorOptions & options);

  /**
   * @brief Declare a buffer.
   *
   * Note that all buffers are stored in the host (the object exposed to users). An object may be
   * used multiple times in the host, and the same buffer may be declared multiple times. That is
   * allowed, but only the first call to declare_buffer constructs the buffer value, and subsequent
   * calls only returns a reference to the existing buffer.
   *
   * @tparam T Buffer type. See @ref statically-shaped-tensor for supported types.
   * @param name Buffer name
   * @param rawval Buffer value
   * @return T The value of the registered buffer.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const T & rawval);

  /**
   * @brief Declare a buffer.
   *
   * Similar to the previous method, but additionally handles the resolution of cross-referenced
   * parameters.
   *
   * @tparam T Buffer type. See @ref statically-shaped-tensor for supported types.
   * @param name Buffer name.
   * @param tensorname The cross-ref'ed "string" that defines the value of the buffer.
   * @return T The value of the registered buffer.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const TensorName & tensorname);

  /**
   * @brief Declare a buffer.
   *
   * Similar to the previous methods, but this method takes care of the high-level logic to directly
   * construct a buffer from the input option.
   *
   * @tparam T Buffer type. See @ref statically-shaped-tensor for supported types.
   * @param name Buffer name
   * @param input_option_name Name of the input option that defines the value of the model
   * buffer.
   * @return T Reference to buffer
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const std::string & input_option_name);

  /// Assign stack to buffers
  void assign_buffer_stack(jit::Stack & stack);

  /// Collect stack from buffers
  jit::Stack collect_buffer_stack() const;

private:
  NEML2Object * _object;

  /**
   * @brief Parsed input file options for this object.

   * These options are useful for example when we declare a variable using an input option name.
   *
   */
  const OptionSet _object_options;

  /// The actual storage for all the buffers
  std::map<std::string, std::unique_ptr<TensorValueBase>> _buffer_values;
};

} // namespace neml2
