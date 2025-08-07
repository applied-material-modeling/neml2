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

#include "neml2/models/BufferStore.h"
#include "neml2/misc/assertions.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/Settings.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/TensorValue.h"

namespace neml2
{
BufferStore::BufferStore(NEML2Object * object)
  : _object(object)
{
}

std::map<std::string, std::unique_ptr<TensorValueBase>> &
BufferStore::named_buffers()
{
  neml_assert(_object->host() == _object,
              "named_buffers() should only be called on the host model.");
  return _buffer_values;
}

TensorValueBase &
BufferStore::get_buffer(const std::string & name)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");
  neml_assert(_buffer_values.count(name), "Buffer named ", name, " does not exist.");
  return *_buffer_values[name];
}

void
BufferStore::send_buffers_to(const TensorOptions & options)
{
  for (auto && [name, buffer] : _buffer_values)
    buffer->to_(options);
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const T & rawval)
{
  if (_object->host() != _object)
    return _object->host<BufferStore>()->declare_buffer(
        _object->name() + _object->settings().buffer_name_separator() + name, rawval);

  TensorValueBase * base_ptr = nullptr;

  // If the buffer already exists, return its reference
  if (_buffer_values.count(name))
    base_ptr = &get_buffer(name);
  else
  {
    auto val = std::make_unique<TensorValue<T>>(rawval);
    auto [it, success] = _buffer_values.emplace(name, std::move(val));
    base_ptr = it->second.get();
  }

  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert(ptr, "Internal error: Failed to cast buffer to a concrete type.");
  return (*ptr)();
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const TensorName<T> & tensorname)
{
  auto * factory = _object->factory();
  neml_assert(factory, "Internal error: factory != nullptr");
  return declare_buffer(name, tensorname.resolve(factory));
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const std::string & input_option_name)
{
  if (_object->input_options().contains(input_option_name))
    return declare_buffer<T>(name, _object->input_options().get<TensorName<T>>(input_option_name));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type.");
}

#define BUFFERSTORE_INTANTIATE_TENSORBASE(T)                                                       \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const T &);               \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const TensorName<T> &);   \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const std::string &)
FOR_ALL_TENSORBASE(BUFFERSTORE_INTANTIATE_TENSORBASE);

void
BufferStore::assign_buffer_stack(jit::Stack & stack)
{
  const auto & buffers = _object->host<BufferStore>()->named_buffers();

  neml_assert_dbg(stack.size() >= buffers.size(),
                  "Stack size (",
                  stack.size(),
                  ") is smaller than the number of buffers in the model (",
                  buffers.size(),
                  ").");

  // Last n tensors in the stack are the buffers
  std::size_t i = stack.size() - buffers.size();
  for (auto && [name, buffer] : buffers)
  {
    const auto tensor = stack[i++].toTensor();
    buffer->assign(tensor, TracerPrivilege{});
  }

  // Drop the input variables from the stack
  jit::drop(stack, buffers.size());
}

jit::Stack
BufferStore::collect_buffer_stack() const
{
  const auto & buffers = _object->host<BufferStore>()->named_buffers();
  jit::Stack stack;
  stack.reserve(buffers.size());
  for (auto && [name, buffer] : buffers)
    stack.emplace_back(Tensor(*buffer));
  return stack;
}
} // namespace neml2
