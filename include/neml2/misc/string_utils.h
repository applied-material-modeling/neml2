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
#include <vector>
#include <sstream>

#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"

namespace neml2::utils
{
template <typename T, typename... Args>
void stream_all(std::ostringstream & ss, T && val, Args &&... args);

/// Demangle a piece of cxx abi type information
std::string demangle(const char * name);

template <typename T>
std::string stringify(const T & t);

std::string join(const std::vector<std::string> & strs, const std::string & delim);

std::vector<std::string> split(const std::string & str, const std::string & delims);

std::string trim(const std::string & str, const std::string & white_space = " \t\n\v\f\r");

bool start_with(std::string_view str, std::string_view prefix);

bool end_with(std::string_view str, std::string_view suffix);
} // namespace neml2::utils

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2::utils
{
template <typename T, typename... Args>
void
stream_all(std::ostringstream & ss, T && val, Args &&... args)
{
  ss << val;
  if constexpr (sizeof...(args) > 0)
    stream_all(ss, std::forward<Args>(args)...);
}

template <typename T>
std::string
stringify(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}

template <>
inline std::string
stringify(const bool & t)
{
  return t ? "true" : "false";
}
}
