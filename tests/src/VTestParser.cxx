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

#include <fstream>

#include "neml2/misc/defaults.h"
#include "neml2/misc/assertions.h"
#include "VTestParser.h"

using namespace neml2;

VTestParser::VTestParser(const std::string & name)
  : _name(name)
{
  parse();
}

void
VTestParser::parse()
{
  std::ifstream file(_name);
  neml_assert(file.is_open(), "Unable to open file ", _name);

  std::string buffer;
  std::vector<std::vector<double>> scalar_buffers;

  // Simple state machine to figure out where we are in the file
  int state = 0;

  while (std::getline(file, buffer))
  {
    auto tokens = utils::split(buffer, " \t\n\v\f\r");

    // Comments only allowed in the first position I guess
    if (!tokens.empty() && tokens[0] == "#")
      continue;

    // Meta
    if (state == 0)
    {
      _meta = tokens;
      state++;
    }
    // Description line
    else if (state == 1)
    {
      _description = buffer;
      state++;
    }
    // Column headers
    else if (state == 2)
    {
      _headers = tokens;
      scalar_buffers.resize(_headers.size());
      state++;
    }
    // Data
    else
      for (size_t i = 0; i < tokens.size(); i++)
        scalar_buffers[i].push_back(std::stod(tokens[i]));
  }

  // Convert buffers to tensors
  for (size_t i = 0; i < _headers.size(); i++)
    _data[_headers[i]] = Tensor::create(scalar_buffers[i], 1, default_tensor_options());
}
