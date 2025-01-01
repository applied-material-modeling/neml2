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

namespace neml2
{
BufferStore::BufferStore(OptionSet options, NEML2Object * object)
  : _object(object),
    _object_options(std::move(options))
{
}

Storage<std::string, TensorValueBase> &
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
  auto * base_ptr = _buffer_values.query_value(name);
  neml_assert(base_ptr, "Buffer named ", name, " does not exist.");
  return *base_ptr;
}

const TensorValueBase &
BufferStore::get_buffer(const std::string & name) const
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");
  const auto * base_ptr = _buffer_values.query_value(name);
  neml_assert(base_ptr, "Buffer named ", name, " does not exist.");
  return *base_ptr;
}

void
BufferStore::send_buffers_to(const torch::TensorOptions & options)
{
  for (auto && [name, buffer] : _buffer_values)
    buffer.to_(options);
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const T & rawval)
{
  if (_object->host() != _object)
    return _object->host<BufferStore>()->declare_buffer(
        _object->name() + buffer_name_separator() + name, rawval);

  TensorValueBase * base_ptr = nullptr;

  // If the buffer already exists, return its reference
  if (_buffer_values.has_key(name))
    base_ptr = &get_buffer(name);
  else
  {
    auto val = std::make_unique<TensorValue<T>>(rawval);
    base_ptr = _buffer_values.set_pointer(name, std::move(val));
  }

  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert(ptr, "Internal error: Failed to cast buffer to a concrete type.");
  return ptr->value();
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const CrossRef<T> & crossref)
{
  return declare_buffer(name, T(crossref));
}

template <typename T, typename>
const T &
BufferStore::declare_buffer(const std::string & name, const std::string & input_option_name)
{
  if (_object_options.contains<T>(input_option_name))
    return declare_buffer(name, _object_options.get<T>(input_option_name));

  if (_object_options.contains<CrossRef<T>>(input_option_name))
    return declare_buffer(name, T(_object_options.get<CrossRef<T>>(input_option_name)));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type. Note that "
      "the buffer type can either be a plain type or a cross-reference.");
}

#define BUFFERSTORE_INTANTIATE_TENSORBASE(T)                                                       \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const T &);               \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const CrossRef<T> &);     \
  template const T & BufferStore::declare_buffer<T>(const std::string &, const std::string &)
FOR_ALL_TENSORBASE(BUFFERSTORE_INTANTIATE_TENSORBASE);

void
BufferStore::assign_buffer_stack(torch::jit::Stack & stack)
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
    buffer = Tensor(tensor, tensor.dim() - Tensor(buffer).base_dim());
  }

  // Drop the input variables from the stack
  torch::jit::drop(stack, buffers.size());
}

torch::jit::Stack
BufferStore::collect_buffer_stack() const
{
  const auto & buffers = _object->host<BufferStore>()->named_buffers();
  torch::jit::Stack stack;
  stack.reserve(buffers.size());
  for (auto && [name, buffer] : buffers)
    stack.push_back(Tensor(buffer));
  return stack;
}
} // namespace neml2
