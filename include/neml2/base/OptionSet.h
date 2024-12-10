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

#include "neml2/misc/utils.h"

#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <memory>

namespace neml2
{
// Forward decl
class OptionSet;
class LabeledAxisAccessor;

namespace details
{
/**
 * @name Helper functions for printing
 *
 * Helper functions for printing scalar, vector, vector of vector. Called from
 * OptionSet::Option<T>::print(...).
 *
 * The leading underscore does NOT conform to our naming convention. It is there to avoid ambiguity
 * with customers who try to inject a function with the same name into our namespace, which is
 * unusual but possible with a `using namespace` in the global scope.
 */
///@{
template <typename P>
void _print_helper(std::ostream & os, const P *);
template <typename P>
void _print_helper(std::ostream & os, const std::vector<P> *);
/// The evil vector of bool :/
template <>
void _print_helper(std::ostream & os, const std::vector<bool> *);
template <typename P>
void _print_helper(std::ostream & os, const std::vector<std::vector<P>> *);
/// Specialization so that we don't print out unprintable characters
template <>
void _print_helper(std::ostream & os, const char *);
/// Specialization so that we don't print out unprintable characters
template <>
void _print_helper(std::ostream & os, const unsigned char *);
///@}
}

bool options_compatible(const OptionSet & opts, const OptionSet & additional_opts);

// Streaming operators
std::ostream & operator<<(std::ostream & os, FType);
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

  /// Copy constructor. Deep copy
  OptionSet(const OptionSet &);

  virtual ~OptionSet() = default;

  /// Assignment operator. Deep copy
  OptionSet & operator=(const OptionSet & source);

  /**
   * Addition/Assignment operator.  Inserts copies of all options
   * from \p source.  Any options of the same name already in \p
   * this are replaced.
   */
  void operator+=(const OptionSet & source);

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

  /**
   * \returns \p true if an option of type \p T
   * with a specified name exists, \p false otherwise.
   */
  template <typename T>
  bool contains(const std::string &) const;

  /**
   * \returns \p true if an option with a specified name exists, \p false otherwise.
   *
   * We return \p true if an option of specified name exists regardless of its type.
   */
  bool contains(const std::string &) const;

  /// \returns The total number of options
  std::size_t size() const { return _values.size(); }

  /// Clear internal data structures & frees any allocated memory.
  virtual void clear();

  /// Print the contents.
  void print(std::ostream & os = std::cout) const;

  /**
   * Abstract definition of an option.
   */
  class OptionBase
  {
  public:
    virtual ~OptionBase() = default;

    /// Test for option equality
    virtual bool operator==(const OptionBase & other) const = 0;

    /// Test for option inequality
    virtual bool operator!=(const OptionBase & other) const = 0;

    /// A readonly reference to the option's name
    const std::string & name() const { return _metadata.name; }

    /// A readonly reference to the option's type
    const std::string & type() const { return _metadata.type; }

    /// A readonly reference to the option's ftype
    const FType & ftype() const { return _metadata.ftype; }

    /// A writable reference to the option's ftype
    FType & ftype() { return _metadata.ftype; }

    /// A readonly reference to the option's docstring
    const std::string & doc() const { return _metadata.doc; }

    /// A writable reference to the option's docstring
    std::string & doc() { return _metadata.doc; }

    /// A readonly reference to the option's suppression status
    const bool & suppressed() const { return _metadata.suppressed; }

    /// A writable reference to the option's suppression status
    bool & suppressed() { return _metadata.suppressed; }

    /// A readonly reference to the option's user_specified status
    const bool & user_specified() const { return _metadata.user_specified; }

    /// A writable reference to the option's user_specified status
    bool & user_specified() { return _metadata.user_specified; }

    /**
     * Prints the option value to the specified stream.
     * Must be reimplemented in derived classes.
     */
    virtual void print(std::ostream &) const = 0;

    /**
     * Clone this value.  Useful in copy-construction.
     * Must be reimplemented in derived classes.
     */
    virtual std::unique_ptr<OptionBase> clone() const = 0;

  protected:
    /**
     * Metadata associated with this option
     */
    struct Metadata
    {
      /**
       * @brief Name of the option
       *
       * For example, in a HIT input file, this is the field name that appears on the left-hand side
       * ~~~~~~~~~~~~~~~~~python
       * [foo]
       *   type = SomeModel
       *   bar = 123
       * []
       * ~~~~~~~~~~~~~~~~~
       * where "bar" is the option name
       */
      std::string name = "";
      /**
       * @brief Type of the option
       *
       * We use RTTI to determine the type of the option. Most importantly, two options are
       * considered different if they have different types, even if they have the same name. For
       * example, if you specify an option of name "foo" of type `int` as an expected option, later
       * if you attempt to retrieve an option of name "foo" but of type `string`, an exception will
       * be thrown saying that the option does not exist.
       */
      std::string type = "";
      /**
       * @brief Option's role in defining the function
       *
       * Since the syntax documentation is automatically extracted from options defined by
       * neml2::NEML2Object::expected_options, there is no way for us to tell, at the time of syntax
       * extraction, whether a variable name is used for the model's input variable, output
       * variable. This metadata information defines such missing information. See neml2::FType for
       * enum values.
       */
      FType ftype = FType::NONE;
      /**
       * @brief Option's doc string
       *
       * When we build the documentation for NEML2, we automatically extract the syntax and
       * convert it to a markdown file. The syntax of NEML2 is just the collection of expected
       * options of all the registered objects. Doxygen will then render the markdown syntax to
       * the target output format, e.g., html, tex, etc. This implies that the docstring can
       * contain anything that the Doxygen's markdown renderer can understand. For more
       * information, see https://www.doxygen.nl/manual/markdown.html
       */
      std::string doc = "";
      /**
       * @brief Whether this option is suppressed
       *
       * By default an option is not suppressed. However, it is sometimes desirable for a derived
       * object to suppress certain option. A suppressed option cannot be modified by the user. It
       * is up to the specific Parser to decide what happens when a user attempts to set a
       * suppressed option, e.g., the parser can choose to throw an exception, print a warning and
       * accept it, or print a warning and ignores it.
       */
      bool suppressed = false;
      /**
       * @brief Whether this option has been specified by the user from the input file
       *
       * In occasions, options are optional. This field is used to determine whether the user has
       * specified the option. If the user has not specified the option, the default (sometimes
       * undefined) value is used. It is therefore important to check this flag before retrieving
       * optional options.
       */
      bool user_specified = false;

      bool operator==(const Metadata & other) const
      {
        return name == other.name && type == other.type && ftype == other.ftype &&
               doc == other.doc && suppressed == other.suppressed &&
               user_specified == other.user_specified;
      }

      bool operator!=(const Metadata & other) const { return !(*this == other); }

    } _metadata;
  };

  /**
   * Concrete definition of an option value
   * for a specified type
   */
  template <typename T>
  class Option : public OptionBase
  {
  public:
    Option(const std::string & name)
      : _value()
    {
      _metadata.name = name;
      _metadata.type = utils::demangle(typeid(T).name());
    }

    virtual bool operator==(const OptionBase & other) const override;

    virtual bool operator!=(const OptionBase & other) const override;

    /**
     * \returns A read-only reference to the option value
     */
    const T & get() const { return _value; }

    /**
     * \returns A writable reference to the option value
     */
    T & set() { return _value; }

    virtual void print(std::ostream &) const override;

    virtual std::unique_ptr<OptionBase> clone() const override;

  private:
    /// Stored option value
    T _value;
  };

  /**
   * \returns A constant reference to the specified option value.  Requires, of course, that the
   * option exists.
   */
  template <typename T>
  const T & get(const std::string &) const;

  const OptionBase & get(const std::string &) const;

  ///@{
  /**
   * \returns A writable reference to the specified option value. This method will create the option
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
  typedef std::map<std::string, std::unique_ptr<OptionBase>, std::less<>> map_type;

  /// Option map iterator
  typedef map_type::iterator iterator;

  /// Constant option map iterator
  typedef map_type::const_iterator const_iterator;

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

template <typename T>
bool
OptionSet::Option<T>::operator==(const OptionBase & other) const
{
  const auto other_ptr = dynamic_cast<const Option<T> *>(&other);
  if (!other_ptr)
    return false;

  return (_metadata == other_ptr->_metadata) && (other_ptr->get() == this->get());
}

template <typename T>
bool
OptionSet::Option<T>::operator!=(const OptionBase & other) const
{
  return !(*this == other);
}

// LCOV_EXCL_START
template <typename T>
void
OptionSet::Option<T>::print(std::ostream & os) const
{
  details::_print_helper(os, static_cast<const T *>(&_value));
}
// LCOV_EXCL_STOP

template <typename T>
std::unique_ptr<OptionSet::OptionBase>
OptionSet::Option<T>::clone() const
{
  auto copy = std::make_unique<Option<T>>(this->name());
  copy->_value = this->_value;
  copy->_metadata = this->_metadata;
  return copy;
}

template <typename T>
bool
OptionSet::contains(const std::string & name) const
{
  OptionSet::const_iterator it = _values.find(name);
  if (it != _values.end())
    if (dynamic_cast<const Option<T> *>(it->second.get()))
      return true;
  return false;
}

template <typename T>
const T &
OptionSet::get(const std::string & name) const
{
  neml_assert(this->contains<T>(name),
              "ERROR: no option named \"",
              name,
              "\" found.\n\nKnown options:\n",
              *this);

  auto ptr = dynamic_cast<Option<T> *>(_values.at(name).get());
  return ptr->get();
}

template <typename T, FType F>
T &
OptionSet::set(const std::string & name)
{
  if (!this->contains<T>(name))
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

namespace details
{
// LCOV_EXCL_START
template <typename P>
void
_print_helper(std::ostream & os, const P * option)
{
  os << *option;
}

template <>
inline void
_print_helper(std::ostream & os, const char * option)
{
  os << static_cast<int>(*option);
}

template <>
inline void
_print_helper(std::ostream & os, const unsigned char * option)
{
  os << static_cast<int>(*option);
}

template <typename P>
void
_print_helper(std::ostream & os, const std::vector<P> * option)
{
  for (const auto & p : *option)
    os << p << " ";
}

template <>
inline void
_print_helper(std::ostream & os, const std::vector<bool> * option)
{
  for (const auto p : *option)
    os << static_cast<bool>(p) << " ";
}

template <typename P>
void
_print_helper(std::ostream & os, const std::vector<std::vector<P>> * option)
{
  for (const auto & pv : *option)
    _print_helper(os, &pv);
}
} // namespace details
// LCOV_EXCL_STOP
} // namespace neml2
