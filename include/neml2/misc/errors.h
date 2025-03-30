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

#include <exception>
#include <string>
#include <utility>

namespace neml2
{
class NEMLException : public std::exception
{
public:
  NEMLException() = default;

  NEMLException(std::string msg)
    : _msg(std::move(msg))
  {
  }

  const char * what() const noexcept override;

protected:
  std::string _msg;
};

class SetupException : public std::exception
{
public:
  SetupException(std::string msg)
    : _msg(std::move(msg))
  {
  }

  const char * what() const noexcept override;

private:
  std::string _msg;
};

class ParserException : public SetupException
{
public:
  ParserException(std::string msg)
    : SetupException(std::move(msg))
  {
  }
};

class FactoryException : public SetupException
{
public:
  FactoryException(std::string msg)
    : SetupException(std::move(msg))
  {
  }
};

/// Exception type reserved for diagnostics, so as to not conceptually clash with other exceptions
class Diagnosis : public NEMLException
{
public:
  using NEMLException::NEMLException;
};
} // namespace neml2
