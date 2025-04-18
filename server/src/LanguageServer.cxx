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

#include "hit/parse.h"
#include <wasphit/HITInterpreter.h>
#include "LanguageServer.h"

namespace neml2
{
static std::string
replaceAll(std::string str, const std::string & from, const std::string & to)
{
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos)
  {
    str.replace(pos, from.length(), to);
    pos += to.length(); // Handles cases where 'to' contains 'from'
  }
  return str;
}

static std::string
canonicalize(const std::string & str)
{
  // replace tabs with spaces
  std::string formatted = str;
  std::replace(formatted.begin(), formatted.end(), '\t', ' ');

  // deduplicate spaces
  auto adjacent_spaces = [](char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); };
  auto new_end = std::unique(formatted.begin(), formatted.end(), adjacent_spaces);
  formatted.erase(new_end, formatted.end());

  return formatted;
}

LanguageServer::LanguageServer()
  : _connection(std::make_shared<wasp::lsp::IOStreamConnection>(this))
{
  // set server capabilities to receive full input text when changed
  server_capabilities[wasp::lsp::m_text_doc_sync] = wasp::DataObject();
  server_capabilities[wasp::lsp::m_text_doc_sync][wasp::lsp::m_open_close] = true;
  server_capabilities[wasp::lsp::m_text_doc_sync][wasp::lsp::m_change] = wasp::lsp::m_change_full;

  // notify completion, symbol, formatting, definition capabilities support
  server_capabilities[wasp::lsp::m_completion_provider] = wasp::DataObject();
  server_capabilities[wasp::lsp::m_completion_provider][wasp::lsp::m_resolve_provider] = false;
  // server_capabilities[wasp::lsp::m_doc_symbol_provider] = true;
  server_capabilities[wasp::lsp::m_doc_format_provider] = true;
  // server_capabilities[wasp::lsp::m_definition_provider] = true;
  // server_capabilities[wasp::lsp::m_references_provider] = true;
  server_capabilities[wasp::lsp::m_hover_provider] = true;
}

bool
LanguageServer::parseDocumentForDiagnostics(wasp::DataArray & /*diagnostics*/)
{
  // add updated document text to map associated with current document path
  _path_to_text[document_path] = document_text;
  return true;
}

bool
LanguageServer::updateDocumentTextChanges(const std::string & replacement_text,
                                          int /* start_line */,
                                          int /* start_character */,
                                          int /* end_line */,
                                          int /* end_character*/,
                                          int /* range_length*/)
{
  // replacement text swaps full document as indicated in server capabilities
  document_text = replacement_text;
  return true;
}

bool
LanguageServer::gatherDocumentCompletionItems(wasp::DataArray & /*completionItems*/,
                                              bool & /*is_incomplete*/,
                                              int /*line*/,
                                              int /*character*/)
{
  return true;
}

bool
LanguageServer::gatherDocumentDefinitionLocations(wasp::DataArray & /*definitionLocations*/,
                                                  int /*line*/,
                                                  int /*character*/)
{
  return true;
}

bool
LanguageServer::getHoverDisplayText(std::string & display_text, int line, int character)
{
  // Everything we know about registered objects in the registry
  const auto & info = Registry::info();

  // input check expanded any brace expressions in cached tree so reprocess
  std::stringstream input_errors, input_stream(getDocumentText());
  wasp::DefaultHITInterpreter interpreter(input_errors);

  // return without any hover text if input parsing fails
  if (!interpreter.parseStream(input_stream, document_path) || interpreter.root().is_null())
    return true;

  // find hit node for zero based request line and column number from input
  auto request_context = wasp::findNodeUnderLineColumn(interpreter.root(), line + 1, character + 1);

  // request must have a parent
  if (!request_context.has_parent())
    return true;

  // get LHS and RHS of the request context
  auto key = request_context.parent().name();
  auto val = request_context.last_as_string();
  writeDebugMessage(std::string("Hover request for ") + key + " = " + val);

  // if request is the RHS of 'type = xxx', we can look up the object description
  if (request_context.type() == wasp::VALUE && key == std::string("type"))
  {
    const auto object_info = info.find(val);
    if (object_info == info.end())
      return true;
    display_text = object_info->second.expected_options.doc();
    display_text = replaceAll(display_text, "\\f", "");
  }

  // if request is the LHS of 'foo = xxx', we can look up the description for 'foo'
  else if (request_context.type() == wasp::DECL)
  {
    // find the type of the object
    const auto type = request_context.parent().parent().child_by_name("type");
    if (type.size() != 1)
      return true;
    const auto type_key = type[0].last_as_string();
    writeDebugMessage(std::string("Hover request for DECL ") + key + ", parent " + type_key);

    const auto object_info = info.find(type_key);
    if (object_info == info.end())
      return true;
    const auto & options = object_info->second.expected_options;
    if (!options.contains(key))
      return true;
    display_text = options.get(key).doc();
    display_text = replaceAll(display_text, "\\f", "");
  }

  return true;
}

bool
LanguageServer::gatherDocumentReferencesLocations(wasp::DataArray & /*referencesLocations*/,
                                                  int /*line*/,
                                                  int /*character*/,
                                                  bool /*include_declaration*/)
{
  return true;
}

bool
LanguageServer::gatherDocumentFormattingTextEdits(wasp::DataArray & edits,
                                                  int tab_size,
                                                  bool /* insert_spaces */)
{
  // input check expanded any brace expressions in cached tree so reprocess
  std::stringstream input_errors, input_stream(getDocumentText());
  wasp::DefaultHITInterpreter interpreter(input_errors);

  // return without adding any formatting text edits if input parsing fails
  if (!interpreter.parseStream(input_stream, document_path))
    return true;

  // return without adding any formatting text edits if parser root is null
  if (interpreter.root().is_null())
    return true;

  // get input root node line and column range to represent entire document
  auto view_root = interpreter.root();
  auto document_start_line = view_root.line() - 1;
  auto document_start_char = view_root.column() - 1;
  auto document_last_line = view_root.last_line() - 1;
  auto document_last_char = view_root.last_column();

  // set number of spaces for indentation and build formatted document text
  _tab_size = tab_size;
  auto starting_line = view_root.line() - 1;
  auto document_format = formatDocument(view_root, starting_line, 0);

  // add formatted text with whole line and column range to formatting list
  edits.push_back(wasp::DataObject());
  wasp::DataObject * item = edits.back().to_object();
  bool pass = wasp::lsp::buildTextEditObject(*item,
                                             errors,
                                             int(document_start_line),
                                             int(document_start_char),
                                             int(document_last_line),
                                             int(document_last_char),
                                             document_format);
  return pass;
}

bool
LanguageServer::gatherDocumentSymbols(wasp::DataArray & /*documentSymbols*/)
{
  return true;
}

const std::string &
LanguageServer::getDocumentText() const
{
  neml_assert(_path_to_text.count(document_path), "No text for path ", document_path);
  return _path_to_text.at(document_path);
}

std::string
LanguageServer::formatDocument(const wasp::HITNodeView & parent,
                               std::size_t & prev_line,
                               std::size_t level)
{
  // build string of newline and indentation spaces from level and tab size
  std::string newline_indent = "\n" + std::string(level * _tab_size, ' ');

  // formatted string that will be built recursively by appending each call
  std::string format_string;

  // walk over all children of this node context and build formatted string
  for (std::size_t i = 0; i < parent.child_count(); i++)
  {
    // walk must be index based to catch file include and skip its children
    wasp::HITNodeView child = parent.child_at(i);

    // add blank line if necessary after previous line and before this line
    std::string blank = child.line() > prev_line + 1 ? "\n" : "";

    // format include directive with indentation and collapse extra spacing
    if (child.type() == wasp::FILE)
      format_string += blank + newline_indent + utils::trim(canonicalize(child.data()));

    // format normal comment with indentation and inline comment with space
    else if (child.type() == wasp::COMMENT)
      format_string +=
          (child.line() == prev_line ? " " : blank + newline_indent) + utils::trim(child.data());

    // format object recursively with indentation
    else if (child.type() == wasp::OBJECT)
      format_string += blank + newline_indent + "[" + child.name() + "]" +
                       formatDocument(child, prev_line, level + 1) + newline_indent + "[]";

    // format keyed value with indentation and calling reusable hit methods
    else if (child.type() == wasp::KEYED_VALUE || child.type() == wasp::ARRAY)
    {
      const std::string prefix = newline_indent + child.name() + " = ";
      const std::string render_val = hit::extractValue(child.data());
      std::size_t val_column = child.child_count() > 2 ? child.child_at(2).column() : 0;
      std::size_t prefix_len = prefix.size() - 1;
      format_string += blank + prefix + hit::formatValue(render_val, val_column, prefix_len);
    }

    // set previous line reference used for blank lines and inline comments
    prev_line = child.last_line();
  }

  // remove leading newline if this is level zero returning entire document
  return level != 0 ? format_string : format_string.substr(1);
}

void
LanguageServer::writeDebugMessage(const std::string & message)
{
  wasp::DataObject obj, params;
  params[wasp::lsp::m_message] = message;
  obj[wasp::lsp::m_method] = "neml2/debug";
  obj[wasp::lsp::m_params] = params;
  connectionWrite(obj);
}
} // namespace neml2
