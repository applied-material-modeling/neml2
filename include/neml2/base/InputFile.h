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

#include "neml2/base/OptionSet.h"
#include "neml2/base/Settings.h"

namespace neml2
{
class Settings;

/**
 * @brief A data structure that holds options of multiple objects.
 */
class InputFile
{
public:
  InputFile(const OptionSet & settings);

  /// Get global settings
  const std::shared_ptr<Settings> & settings() const { return _settings; }

  /// Get all the object options under a specific section.
  std::map<std::string, OptionSet> & operator[](const std::string & section);

  /// Get all the object options under a specific section.
  const std::map<std::string, OptionSet> & operator[](const std::string & section) const;

  /**
   * A two layer map, where the first layer key is the section name, e.g.Models, Tensors, Drivers,
   * etc., and the second layer key is the object name.
   */
  const std::map<std::string, std::map<std::string, OptionSet>> & data() const { return _data; }

private:
  /// Global settings specified under the [Settings] section
  const std::shared_ptr<Settings> _settings;

  /// Collection of options for all manufacturable objects
  std::map<std::string, std::map<std::string, OptionSet>> _data;
};

std::ostream & operator<<(std::ostream & os, const InputFile & p);
} // namespace neml2
