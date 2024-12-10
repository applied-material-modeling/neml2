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

#include "neml2/base/NEML2Object.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/TensorValue.h"

// The following are not directly used by BufferStore itself.
// We put them here so that derived classes can add expected options of these types.
#include "neml2/base/CrossRef.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
/// Interface for object which can store buffers
class BufferStore
{
public:
  BufferStore(const OptionSet & options, NEML2Object * object);

  ///@{
  /// @returns the buffer storage
  const Storage<std::string, TensorValueBase> & named_buffers() const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<BufferStore *>(this)->named_buffers();
  }
  Storage<std::string, TensorValueBase> & named_buffers();
  ///}@

  /// Get a writable reference of a buffer
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  TensorValue<T> & get_buffer(const std::string & name);

protected:
  /**
   * @brief Send all buffers to \p options
   *
   * @param options The target options
   */
  virtual void send_buffers_to(const torch::TensorOptions & options);

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
   * @return Reference to buffer
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const T & rawval);

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
   * @param input_option_name Name of the input option that defines the value of the model
   * buffer.
   * @return T Reference to buffer
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const std::string & input_option_name);

private:
  NEML2Object * _object;

  /**
   * @brief Parsed input file options for this object.

   * These options are useful for example when we declare a variable using an input option name.
   *
   */
  const OptionSet _object_options;

  /// The actual storage for all the buffers
  Storage<std::string, TensorValueBase> _buffer_values;
};

template <typename T, typename>
TensorValue<T> &
BufferStore::get_buffer(const std::string & name)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");

  auto base_ptr = _buffer_values.query_value(name);
  neml_assert(base_ptr, "Buffer named ", name, " does not exist.");
  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert_dbg(ptr, "Internal error: Failed to cast buffer to a concrete type.");
  return *ptr;
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const T & rawval)
{
  if (_object->host() != _object)
    return _object->host<BufferStore>()->declare_buffer(
        _object->name() + buffer_name_separator() + name, rawval);

  // If the buffer already exists, return its reference
  if (_buffer_values.has_key(name))
    return get_buffer<T>(name).value();

  auto val = std::make_unique<TensorValue<T>>(rawval);
  auto base_ptr = _buffer_values.set_pointer(name, std::move(val));
  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert(ptr, "Internal error: Failed to cast buffer to a concrete type.");
  return ptr->value();
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const std::string & input_option_name)
{
  if (_object_options.contains<T>(input_option_name))
    return declare_buffer(name, _object_options.get<T>(input_option_name));
  else if (_object_options.contains<CrossRef<T>>(input_option_name))
    return declare_buffer(name, T(_object_options.get<CrossRef<T>>(input_option_name)));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type. Note that "
      "the buffer type can either be a plain type, a cross-reference, or an interpolator.");
}

} // namespace neml2
