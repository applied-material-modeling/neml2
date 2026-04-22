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

#include <map>
#include <string>
#include <memory>

#include "neml2/base/Option.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/string_utils.h"
#include "neml2/misc/types.h"

namespace neml2
{
// Forward decl
class OptionSet;
template <typename T>
struct TensorName;

bool options_compatible(const OptionSet & opts, const OptionSet & additional_opts);

// Streaming operators
std::ostream & operator<<(std::ostream & os, const OptionSet & p);

/**
 * @brief A custom map-like data structure. The keys are strings, and the values can be
 * nonhomogeneously typed.
 *
 */
class OptionSet
{
public:
  OptionSet() = default;

  OptionSet(const OptionSet &);
  OptionSet(OptionSet &&) noexcept;
  OptionSet & operator=(const OptionSet &);
  OptionSet & operator=(OptionSet &&) noexcept;
  virtual ~OptionSet() = default;

  ///@{
  /**
   * Addition/Assignment operator.  Inserts copies of all options
   * from \p source.  Any options of the same name already in \p
   * this are replaced.
   *
   * @note This operator does not modify the metadata of the option set.
   */
  void operator+=(const OptionSet & source);
  void operator+=(OptionSet && source);
  ///@}

  /// A readonly reference to the option set's name
  const std::string & name() const { return _metadata.name; }
  /// A writable reference to the option set's name
  std::string & name() { return _metadata.name; }
  /// A readonly reference to the option set's type
  const std::string & type() const { return _metadata.type; }
  /// A writable reference to the option set's type
  std::string & type() { return _metadata.type; }
  /// A readonly reference to the option set's path
  const std::string & path() const { return _metadata.path; }
  /// A writable reference to the option set's path
  std::string & path() { return _metadata.path; }
  /// A readonly reference to the option set's docstring
  const std::string & doc() const { return _metadata.doc; }
  /// A writable reference to the option set's docstring
  std::string & doc() { return _metadata.doc; }
  /// A readonly reference to the option set's section
  const std::string & section() const { return _metadata.section; }
  /// A writable reference to the option set's section
  std::string & section() { return _metadata.section; }

  /// @return true if an option with a specified name exists, false otherwise.
  bool contains(const std::string &) const;

  /// @return true if an option is specified by the user (i.e., from the input file), false otherwise.
  bool user_specified(const std::string & name) const;

  /// @return true if an option is defined (i.e., value has been set either from the input file or programmatically), false otherwise.
  bool defined(const std::string & name) const;

  /// Suppress an option.
  void suppress(const std::string & name);

  /// @return The total number of options
  std::size_t size() const { return _values.size(); }

  /// Clear internal data structures & frees any allocated memory.
  void clear();

  /// Print the contents.
  std::string to_str() const;

  /**
   * @return A copy of the specified option value.  Requires, of course, that the
   * option exists.
   */
  template <typename T>
  T get(const std::string &) const;

  /// @brief Get a const reference to the specified option value.
  const OptionBase & get(const std::string &) const;

  /**
   * @brief Get two options and bind them to form a map
   *
   * The two options are expected to be of type std::vector<K> and std::vector<V>, respectively.
   * Keys shall be unique. Otherwise, an exception is thrown.
   *
   * @tparam K Key type
   * @tparam V Value type
   * @return std::map<K, V>
   */
  template <typename K, typename V>
  std::map<K, V> get_map(const std::string &, const std::string &) const;

  /**
   * @brief Create an option with its default value and its docstring.
   *
   * Throws an exception if the option already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - required: false
   *  - suppressed: false
   *  - user_specified: false
   *  - defined: true
   *
   * Note that later on when a Parser parses the input file, if the user has specified a value for
   * this option, the option value will be overwritten and the user_specified flag will be set to
   * true. If the user has not specified a value for this option, the default value will be used.
   *
   * @tparam T Type of the option value
   * @tparam f Option type (e.g., parameter, buffer, input, output)
   * @param default_value The default value to set
   * @param doc The docstring for the option
   */
  template <typename T, FType f = FType::NONE>
  void add(const std::string &, const T & default_value, std::string doc);

  /**
   * @brief Create a required option with its docstring, without a default value. User must specify
   * a value for this option from the input file.
   *
   * Throws an exception if the option already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - required: true
   *  - suppressed: false
   *  - user_specified: false
   *  - defined: false
   *
   * Note that later on when a Parser parses the input file, if the user has specified a value for
   * this option, the option value will be overwritten and both the user_specified flag and the
   * defined flag will be set to true. The Parser should consider it an error if the user has not
   * specified a value for this option.
   *
   * @tparam T Type of the option value
   * @tparam f Option type (e.g., parameter, buffer, input, output)
   * @param doc The docstring for the option
   */
  template <typename T, FType f = FType::NONE>
  void add(const std::string &, std::string doc);

  /**
   * @brief Create an optional option with its docstring, without a default value.
   *
   * Throws an exception if the option already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - required: false
   *  - suppressed: false
   *  - user_specified: false
   *  - defined: false
   *
   * Note that later on when a Parser parses the input file, if the user has specified a value for
   * this option, the option value will be overwritten and both the user_specified flag and the
   * defined flag will be set to true. If the user has not specified a value for this option, the
   * option is considered undefined and an exception will be thrown when attempting to retrieve the
   * option value.
   *
   * @tparam T Type of the option value
   * @tparam f Option type (e.g., parameter, buffer, input, output)
   * @param doc The docstring for the option
   */
  template <typename T, FType f = FType::NONE>
  void add_optional(const std::string &, std::string doc);

  /**
   * @brief Create a private option with its default value.
   *
   * Throws an exception if the option already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - required: false
   *  - suppressed: true
   *  - user_specified: false
   *  - defined: true
   *
   * @tparam T Type of the option value
   * @param default_value The default value to set
   */
  template <typename T>
  void add_private(const std::string &, const T & default_value);

  /**
   * @brief Set an option.
   *
   * The option must already exist. Otherwise, an exception is thrown. The option value is
   * overridden if it already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - user_specified: false
   *  - defined: true
   *
   * @tparam T Type of the option value
   * @param value The value to set
   */
  template <typename T>
  void set(const std::string &, const T & value);

  /**
   * @brief Set a private option.
   *
   * The option must already exist. Otherwise, an exception is thrown. The option value is
   * overridden if it already exists.
   *
   * Calling this method sets the following metadata for the option:
   *  - user_specified: false
   *  - defined: true
   *
   * @tparam T Type of the option value
   * @param value The value to set
   */
  template <typename T>
  void set_private(const std::string &, const T & value);

  ///@{
  /// Convenient methods to add input variable
  void add_input(const std::string &, const VariableName &, std::string);
  void add_input(const std::string &, std::string);
  void add_optional_input(const std::string &, std::string);
  /// Convenient methods to add output variable
  void add_output(const std::string &, const VariableName &, std::string);
  void add_output(const std::string &, std::string);
  void add_optional_output(const std::string &, std::string);
  /// Convenient methods to add parameter
  template <typename T>
  void add_parameter(const std::string &, const TensorName<T> &, std::string);
  template <typename T>
  void add_parameter(const std::string &, std::string);
  /// Convenient methods to add buffer
  template <typename T>
  void add_buffer(const std::string &, const TensorName<T> &, std::string);
  template <typename T>
  void add_buffer(const std::string &, std::string);
  ///@}

  /// The type of the map that we store internally
  using map_type = std::map<std::string, std::unique_ptr<OptionBase>, std::less<>>;
  /// Option map iterator
  using iterator = map_type::iterator;
  /// Constant option map iterator
  using const_iterator = map_type::const_iterator;

  /// Iterator pointing to the beginning of the set of options
  iterator begin();
  /// Iterator pointing to the beginning of the set of options
  const_iterator begin() const;
  /// Iterator pointing to the end of the set of options
  iterator end();
  /// Iterator pointing to the end of the set of options
  const_iterator end() const;

protected:
  /**
   * Metadata associated with this option set
   */
  struct Metadata
  {
    /**
     * @brief Name of the option set
     *
     * For example, in a HIT input file, this is the subsection name that appears inside the
     * square brackets
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   type = SomeModel
     *   bar = 123
     * []
     * ~~~~~~~~~~~~~~~~~
     * where "foo" is the name of the option set
     */
    std::string name = "";
    /**
     * @brief Type of the option set
     *
     * This is the type of the object that this option set represents. For example, in a HIT input
     * file, this is the value of the "type" field.
     */
    std::string type = "";
    /**
     * @brief Path to the option set
     *
     * The path to an option set describes its hierarchy inside the syntax tree parsed by the
     * parser. For example, in a HIT input file, this is the full path to the current option set
     * (excluding its local path contribution)
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   [bar]
     *     [baz]
     *       type = SomeModel
     *       goo = 123
     *     []
     *   []
     * []
     * ~~~~~~~~~~~~~~~~~
     * The option set with name "baz" has path "foo/bar".
     */
    std::string path = "";
    /**
     * @brief Option set's doc string
     *
     * When we build the documentation for NEML2, we automatically extract the syntax and convert
     * it to a markdown file. The syntax of NEML2 is just the collection of expected options of
     * all the registered objects. Doxygen will then render the markdown syntax to the target
     * output format, e.g., html, tex, etc. This implies that the docstring can contain anything
     * that the Doxygen's markdown renderer can understand. For more information, see
     * https://www.doxygen.nl/manual/markdown.html
     */
    std::string doc = "";
    /**
     * @brief Which NEML2 input file section this object belongs to
     *
     * NEML2 supports first class systems such as [Tensors], [Models], [Drivers], [Solvers], etc.
     * This field denotes which section, i.e. which of the first class system, this object belongs
     * to.
     */
    std::string section = "";
  } _metadata;

  /// Data structure to map names with values
  map_type _values;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
T
OptionSet::get(const std::string & name) const
{
  if (!this->contains(name))
    throw NEMLException("ERROR: no option named \"" + name + "\" found.\n\nKnown options:\n" +
                        to_str());

  auto * opt_base = _values.at(name).get();
  if (!opt_base->defined())
    throw NEMLException("ERROR: option named \"" + name +
                        "\" is being accessed before it is defined.");

  auto ptr = dynamic_cast<Option<T> *>(opt_base);
  if (!ptr)
    throw NEMLException("ERROR: option named \"" + name +
                        "\" is not of the requested type: " + opt_base->type());
  return ptr->get();
}

template <typename K, typename V>
std::map<K, V>
OptionSet::get_map(const std::string & key_option, const std::string & value_option) const
{
  const auto keys = this->get<std::vector<K>>(key_option);
  const auto values = this->get<std::vector<V>>(value_option);
  if (keys.size() != values.size())
    throw NEMLException("Trying to build a map from '" + key_option + "' and '" + value_option +
                        "' with " + std::to_string(keys.size()) + " keys and " +
                        std::to_string(values.size()) + " values.");
  std::map<K, V> result;
  for (size_t i = 0; i < keys.size(); i++)
  {
    if (result.find(keys[i]) != result.end())
      throw NEMLException("Trying to build a map from '" + key_option + "' and '" + value_option +
                          "' with duplicate key: " + utils::stringify(keys[i]));
    result[keys[i]] = values[i];
  }
  return result;
}

template <typename T, FType F>
void
OptionSet::add(const std::string & name, const T & default_value, std::string doc)
{
  if (this->contains(name))
    throw NEMLException("Trying to add option '" + name +
                        "', but an option with the same name already exists.");
  _values[name] = std::make_unique<Option<T>>(name);
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  // docstring and Ftype are always needed
  ptr->doc() = std::move(doc);
  ptr->ftype() = F;

  // default value
  ptr->set() = default_value;

  // metadata
  ptr->required() = false;
  ptr->suppressed() = false;
  ptr->user_specified() = false;
  ptr->defined() = true;
}

template <typename T, FType F>
void
OptionSet::add(const std::string & name, std::string doc)
{
  if (this->contains(name))
    throw NEMLException("Trying to add option '" + name +
                        "', but an option with the same name already exists.");
  _values[name] = std::make_unique<Option<T>>(name);
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  // docstring and Ftype are always needed
  ptr->doc() = std::move(doc);
  ptr->ftype() = F;

  // metadata
  ptr->required() = true;
  ptr->suppressed() = false;
  ptr->user_specified() = false;
  ptr->defined() = false;
}

template <typename T, FType F>
void
OptionSet::add_optional(const std::string & name, std::string doc)
{
  if (this->contains(name))
    throw NEMLException("Trying to add option '" + name +
                        "', but an option with the same name already exists.");
  _values[name] = std::make_unique<Option<T>>(name);
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  // docstring and Ftype are always needed
  ptr->doc() = std::move(doc);
  ptr->ftype() = F;

  // metadata
  ptr->required() = false;
  ptr->suppressed() = false;
  ptr->user_specified() = false;
  ptr->defined() = false;
}

template <typename T>
void
OptionSet::add_private(const std::string & name, const T & default_value)
{
  if (this->contains(name))
    throw NEMLException("Trying to add a private option '" + name +
                        "', but an option with the same name already exists.");
  _values[name] = std::make_unique<Option<T>>(name);
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  // default value
  ptr->set() = default_value;

  // metadata
  ptr->required() = false;
  ptr->suppressed() = true;
  ptr->user_specified() = false;
  ptr->defined() = true;
}

template <typename T>
void
OptionSet::set(const std::string & name, const T & value)
{
  if (!this->contains(name))
    throw NEMLException("Trying to set option '" + name + "', but it does not exist.");

  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  if (ptr->suppressed())
    throw NEMLException("Trying to set private option '" + name +
                        "', which is not allowed. Use set_private() instead.");

  // value
  ptr->set() = value;

  // metadata
  ptr->user_specified() = false;
  ptr->defined() = true;
}

template <typename T>
void
OptionSet::set_private(const std::string & name, const T & value)
{
  if (!this->contains(name))
    throw NEMLException("Trying to set private option '" + name + "', but it does not exist.");

  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());

  if (!ptr->suppressed())
    throw NEMLException("Trying to set private option '" + name +
                        "', but it is not marked as private.");

  // value
  ptr->set() = value;

  // metadata
  ptr->user_specified() = false;
  ptr->defined() = true;
}

template <typename T>
void
OptionSet::add_parameter(const std::string & name,
                         const TensorName<T> & tensor_name,
                         std::string doc)
{
  add<TensorName<T>, FType::PARAMETER>(name, tensor_name, std::move(doc));
}

template <typename T>
void
OptionSet::add_parameter(const std::string & name, std::string doc)
{
  add<TensorName<T>, FType::PARAMETER>(name, std::move(doc));
}

template <typename T>
void
OptionSet::add_buffer(const std::string & name, const TensorName<T> & tensor_name, std::string doc)
{
  add<TensorName<T>, FType::BUFFER>(name, tensor_name, std::move(doc));
}

template <typename T>
void
OptionSet::add_buffer(const std::string & name, std::string doc)
{
  add<TensorName<T>, FType::BUFFER>(name, std::move(doc));
}

} // namespace neml2
