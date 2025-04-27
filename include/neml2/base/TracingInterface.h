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

#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "neml2/base/OptionSet.h"

namespace neml2
{
using json = nlohmann::json;

struct TraceWriter
{
  /// Constructor
  TraceWriter(const std::filesystem::path & file);
  /// Destructor
  ~TraceWriter();

  /// Special methods
  TraceWriter(const TraceWriter &) = delete;
  TraceWriter(TraceWriter &&) = delete;
  TraceWriter & operator=(const TraceWriter &) = delete;
  TraceWriter & operator=(TraceWriter &&) = delete;

  /// File name
  const std::string filename;

  /// Output stream for the trace file
  std::ofstream out;

  void trace_duration_begin(const std::string & name,
                            const std::string & category,
                            const json & args = {},
                            unsigned int pid = 0);
  void trace_duration_end(const std::string & name,
                          const std::string & category,
                          const json & args = {},
                          unsigned int pid = 0);
  void trace_instant(const std::string & name,
                     const std::string & category,
                     const json & args = {},
                     const std::string & scope = "t",
                     unsigned int pid = 0);

private:
  /// Define common event fields
  void write_event_common(json & event,
                          const std::string & name,
                          const std::string & category,
                          const json & args,
                          const std::string & phase,
                          unsigned int pid);
  /// Write the event to the trace file
  void dump_event(const json & event, bool last = false);

  /// Trace epoch
  const std::chrono::high_resolution_clock::time_point _epoch;

  /// Mutex to protect the trace file
  std::mutex _mtx;
};

/// All trace writers
std::unordered_map<std::filesystem::path, std::unique_ptr<TraceWriter>> & event_trace_writers();

/**
 * Interface for classes that support event tracing
 *
 * This class provides a common interface for classes that support event tracing. It
 * provides a mechanism for writing trace data to a file in the Chrome tracing
 * format.
 * @see https://www.chromium.org/developers/how-tos/trace-event-profiling-tool
 * @see https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
 */
class TracingInterface
{
public:
  static OptionSet expected_options();

  TracingInterface(std::string);
  TracingInterface(const OptionSet &);
  virtual ~TracingInterface() = default;

  TracingInterface(const TracingInterface &) = default;
  TracingInterface(TracingInterface &&) = default;
  TracingInterface & operator=(const TracingInterface &) = delete;
  TracingInterface & operator=(TracingInterface &&) = delete;

  bool event_tracing_enabled() const { return _enabled; }

  /// Get the event trace writer
  TraceWriter & event_trace_writer() const;

private:
  TraceWriter & init_writer(std::string);

  /// Whether event tracing is enabled
  bool _enabled;

  /// The trace file to write to
  TraceWriter * _writer;
};
} // namespace neml2
