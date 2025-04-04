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
#include <set>
#include <wasphit/HITInterpreter.h>

#include "wasplsp/ServerImpl.h"
#include "wasplsp/Connection.h"
#include "wasplsp/IOStreamConnection.h"
#include "waspcore/Object.h"
#include "wasphit/HITNodeView.h"

namespace neml2
{
// Function naming conventions here are different from our NEML2 main codebase to match the wasplsp
// spec.  For example, we use camelCase for method names instead of snake_case.

/**
 * @brief NEML2-specific Language Server implementation
 *
 * The base wasp::lsp::ServerImpl handles a lot of the boilerplate for us, but we need to implement
 * a few methods to handle the specifics of our server.
 *
 */
class LanguageServer final : public wasp::lsp::ServerImpl
{
public:
  LanguageServer();
  virtual ~LanguageServer() = default;

  LanguageServer(const LanguageServer &) = delete;
  LanguageServer(LanguageServer &&) = delete;
  LanguageServer & operator=(const LanguageServer &) = delete;
  LanguageServer & operator=(LanguageServer &&) = delete;

  /**
   * get this server's connection - to be implemented on derived servers
   * @return - shared pointer to the server's read / write connection
   */
  std::shared_ptr<wasp::lsp::Connection> getConnection() override { return _connection; }

private:
  /**
   * SortedLocationNodes - type alias for set of nodes sorted by location
   */
  using SortedLocationNodes =
      std::set<wasp::HITNodeView,
               std::function<bool(const wasp::HITNodeView &, const wasp::HITNodeView &)>>;

  /**
   * Parse document for diagnostics
   * @param diagnostics - data array of diagnostics data objects to fill
   * @return - true if completed successfully - does not indicate parse fail
   */
  bool parseDocumentForDiagnostics(wasp::DataArray & diagnostics) override;

  /**
   * Update document text changes
   * @param replacement_text - text to be replaced over the provided range
   * @param start_line - starting replace line number ( zero-based )
   * @param start_character - starting replace column number ( zero-based )
   * @param end_line - ending replace line number ( zero-based )
   * @param end_character - ending replace column number ( zero-based )
   * @param range_length - length of replace range - server specific
   * @return - true if the document text was updated successfully
   */
  bool updateDocumentTextChanges(const std::string & replacement_text,
                                 int start_line,
                                 int start_character,
                                 int end_line,
                                 int end_character,
                                 int range_length) override;

  /**
   * Gather document completion items
   * @param completionItems - data array of completion item objects to fill
   * @param is_incomplete - flag indicating if the completions are complete
   * @param line - line to be used for completions gathering logic
   * @param character - column to be used for completions gathering logic
   * @return - true if the gathering of items completed successfully
   */
  bool gatherDocumentCompletionItems(wasp::DataArray & completionItems,
                                     bool & is_incomplete,
                                     int line,
                                     int character) override;

  /**
   * Gather definition locations
   * @param definitionLocations - data array of locations objects to fill
   * @param line - line to be used for locations gathering logic
   * @param character - column to be used for locations gathering logic
   * @return - true if the gathering of locations completed successfully
   */
  bool gatherDocumentDefinitionLocations(wasp::DataArray & definitionLocations,
                                         int line,
                                         int character) override;

  /**
   * Get hover display text
   * @param display_text - string reference to add hover text for display
   * @param line - zero-based line to use for finding node and hover text
   * @param character - zero-based column for finding node and hover text
   * @return - true if display text was added or left empty without error
   */
  bool getHoverDisplayText(std::string & display_text, int line, int character) override;

  /**
   * Gather references locations
   * @param referencesLocations - data array of locations objects to fill
   * @param line - line to be used for locations gathering logic
   * @param character - column to be used for locations gathering logic
   * @param include_declaration - flag indicating declaration inclusion
   * @return - true if the gathering of locations completed successfully
   */
  bool gatherDocumentReferencesLocations(wasp::DataArray & referencesLocations,
                                         int line,
                                         int character,
                                         bool include_declaration) override;

  /**
   * Gather formatting text edits
   * @param edits - data array of text edit objects to fill
   * @param tab_size - value of the size of a tab in spaces for formatting
   * @param insert_spaces - flag indicating whether to use spaces for tabs
   * @return - true if the gathering of text edits completed successfully
   */
  bool gatherDocumentFormattingTextEdits(wasp::DataArray & edits,
                                         int tab_size,
                                         bool insert_spaces) override;

  /**
   * Gather document symbols
   * @param documentSymbols - data array of symbols data objects to fill
   * @return - true if the gathering of symbols completed successfully
   */
  bool gatherDocumentSymbols(wasp::DataArray & documentSymbols) override;

  /**
   * Read from connection into object
   * @param object - reference to object to be read into
   * @return - true if the read from the connection completed successfully
   */
  bool connectionRead(wasp::DataObject & object) override
  {
    return _connection->read(object, errors);
  }

  /**
   * Write object json to connection
   * @param object - reference to object with contents to write to connection
   * @return - true if the write to the connection completed successfully
   */
  bool connectionWrite(wasp::DataObject & object) override
  {
    return _connection->write(object, errors);
  }

  /**
   * @brief getDocumentText - get the current text of the document
   * @return - string reference to the current text of the document
   */
  const std::string & getDocumentText() const;

  /**
   * Recursively walk down whole nodeview tree while formatting document.
   * @param parent - nodeview for recursive tree traversal starting point
   * @param prev_line - line of last print for blanks and inline comments
   * @param level - current level in document tree to use for indentation
   * @return - formatted string that gets appended to each recursive call
   */
  std::string
  formatDocument(const wasp::HITNodeView & parent, std::size_t & prev_line, std::size_t level);

  /// Send a debug message to the client
  void writeDebugMessage(const std::string & message);

  /// Tab size
  int _tab_size = 0;

  /// Shared pointer to this server's read / write iostream
  std::shared_ptr<wasp::lsp::IOStreamConnection> _connection;

  /// Map of document paths to current text strings
  std::map<std::string, std::string> _path_to_text;
};
} // namespace neml2
