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

#include <string>
#include <memory>
#include <iosfwd>

#include "neml2/misc/types.h"

namespace neml2
{
/**
 * Abstract definition of an option.
 */
class OptionBase
{
public:
  OptionBase() = default;

  OptionBase(OptionBase &&) = delete;
  OptionBase(const OptionBase &) = delete;
  OptionBase & operator=(const OptionBase &) = delete;
  OptionBase & operator=(OptionBase &&) = delete;
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
      return name == other.name && type == other.type && ftype == other.ftype && doc == other.doc &&
             suppressed == other.suppressed && user_specified == other.user_specified;
    }

    bool operator!=(const Metadata & other) const { return !(*this == other); }

  } _metadata;
};
} // namespace neml2
