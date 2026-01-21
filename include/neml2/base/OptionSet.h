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

namespace neml2
{
// Forward decl
class OptionSet;
class LabeledAxisAccessor;

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

  /// @return \p true if an option with a specified name exists, \p false otherwise.
  bool contains(const std::string &) const;

  /// @return \p true if an option is specified by the user (i.e., from the input file), \p false otherwise.
  bool user_specified(const std::string & name) const;

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

  /// Get two options and bind them to find a map
  ///
  /// @tparam K Key type
  /// @tparam V Value type

  /**
   * @brief Get two options and bind them to find a map
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

  ///@{
  /**
   * @return A writable reference to the specified option value. This method will create the option
   * if it does not exist, so it can be used to define options which will later be accessed with the
   * \p get() member.
   */
  template <typename T, FType f = FType::NONE>
  T & set(const std::string &);
  OptionBase & set(const std::string &);
  ///@}

  /// @name Convenient methods to request an input variable
  LabeledAxisAccessor & set_input(const std::string &);
  /// @name Convenient methods to request an output variable
  LabeledAxisAccessor & set_output(const std::string &);
  /// Convenient method to request a parameter
  template <typename T>
  T & set_parameter(const std::string &);
  /// Convenient method to request a buffer
  template <typename T>
  T & set_buffer(const std::string &);

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
     * For example, in a HIT input file, a special field is reserved for the type of the option
     * set
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   type = SomeModel
     *   bar = 123
     * []
     * ~~~~~~~~~~~~~~~~~
     * where "SomeModel" is the option name. The type is registered to the Registry using
     * register_NEML2_object and its variants.
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
T &
OptionSet::set(const std::string & name)
{
  if (!this->contains(name))
    _values[name] = std::make_unique<Option<T>>(name);
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());
  ptr->ftype() = F;
  return ptr->set();
}

template <typename T>
T &
OptionSet::set_parameter(const std::string & name)
{
  return set<T, FType::PARAMETER>(name);
}

template <typename T>
T &
OptionSet::set_buffer(const std::string & name)
{
  return set<T, FType::BUFFER>(name);
}
} // namespace neml2
