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

#include "LanguageServer.h"

namespace neml2
{
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
  server_capabilities[wasp::lsp::m_doc_symbol_provider] = true;
  server_capabilities[wasp::lsp::m_doc_format_provider] = true;
  server_capabilities[wasp::lsp::m_definition_provider] = true;
  server_capabilities[wasp::lsp::m_references_provider] = true;
  server_capabilities[wasp::lsp::m_hover_provider] = true;
}

bool
LanguageServer::parseDocumentForDiagnostics(wasp::DataArray & /*diagnosticsList*/)
{
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
LanguageServer::getHoverDisplayText(std::string & /*display_text*/, int /*line*/, int /*character*/)
{
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
LanguageServer::gatherDocumentFormattingTextEdits(wasp::DataArray & /*formattingTextEdits*/,
                                                  int tab_size,
                                                  bool /* insert_spaces */)
{
  _tab_size = tab_size;
  return true;
}

bool
LanguageServer::gatherDocumentSymbols(wasp::DataArray & /*documentSymbols*/)
{
  return true;
}
} // namespace neml2
