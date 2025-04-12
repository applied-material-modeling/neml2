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

#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

#include "neml2/base/TracingInterface.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
namespace fs = std::filesystem;

std::unordered_map<fs::path, std::unique_ptr<TraceWriter>> &
event_trace_writers()
{
  static std::unordered_map<fs::path, std::unique_ptr<TraceWriter>> trace_writers;
  return trace_writers;
}

TraceWriter::TraceWriter(const fs::path & file)
  : filename(file.string()),
    _epoch(std::chrono::high_resolution_clock::now())
{
  {
    std::lock_guard<std::mutex> lock(_mtx);
    out.open(file);
    neml_assert(out.is_open(), "Failed to open trace file: " + file.string());
    // Write an opening bracket (because the chrome tracing format is represented as a JSON array)
    // We eagerly write this in case the program crashes before the closing bracket is written.
    // Note that the chrome tracing format supports this behavior (i.e. it can handle a partial JSON
    // array without a closing bracket).
    out << "[\n";
  }

  trace_duration_begin("trace writer", "TraceWriter", {{"file", file.string()}});
}

TraceWriter::~TraceWriter()
{
  json event;
  write_event_common(event, "trace writer", "TraceWriter", {{"file", filename}}, "E", 0);
  dump_event(event, true);
  std::lock_guard<std::mutex> lock(_mtx);
  out << "]\n";
  out.close();
}

void
TraceWriter::write_event_common(json & event,
                                const std::string & name,
                                const std::string & category,
                                const json & args,
                                const std::string & phase,
                                unsigned int pid)
{
  event["name"] = name;
  event["cat"] = category;
  event["ph"] = phase;
  event["ts"] = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - _epoch)
                    .count();
  event["pid"] = pid;
  event["tid"] =
      static_cast<unsigned short>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  event["args"] = args;
}

void
TraceWriter::dump_event(const json & event, bool last)
{
  std::lock_guard<std::mutex> lock(_mtx);
  out << event.dump();
  if (!last)
    out << ',';
  out << '\n';
}

void
TraceWriter::trace_duration_begin(const std::string & name,
                                  const std::string & category,
                                  const json & args,
                                  unsigned int pid)
{
  json event;
  write_event_common(event, name, category, args, "B", pid);
  dump_event(event);
}

void
TraceWriter::trace_duration_end(const std::string & name,
                                const std::string & category,
                                const json & args,
                                unsigned int pid)
{
  json event;
  write_event_common(event, name, category, args, "E", pid);
  dump_event(event);
}

void
TraceWriter::trace_instant(const std::string & name,
                           const std::string & category,
                           const json & args,
                           const std::string & scope,
                           unsigned int pid)
{
  json event;
  write_event_common(event, name, category, args, "i", pid);
  event["s"] = scope;
  dump_event(event);
}

OptionSet
TracingInterface::expected_options()
{
  OptionSet options;
  options.set<std::string>("trace_file") = "";
  options.set("trace_file").doc() =
      "The file to write the trace to. If not set, tracing will be disabled.";
  return options;
}

TracingInterface::TracingInterface(std::string trace_file)
  : _enabled(!trace_file.empty()),
    _writer(_enabled ? &init_writer(std::move(trace_file)) : nullptr)
{
}

TracingInterface::TracingInterface(const OptionSet & options)
  : TracingInterface(options.get<std::string>("trace_file"))
{
}

TraceWriter &
TracingInterface::init_writer(std::string filename)
{
  static std::mutex trace_writer_mutex;
  std::lock_guard<std::mutex> lock(trace_writer_mutex);

  auto file = fs::absolute(std::move(filename));
  auto & trace_writers = event_trace_writers();
  auto it = trace_writers.find(file);
  if (it != trace_writers.end())
    return *it->second;

  // Create a new TraceWriter
  auto writer = std::make_unique<TraceWriter>(file);
  auto [it2, success] = trace_writers.emplace(file, std::move(writer));
  neml_assert(success, "Internal error: Trace writer already exists: ", file.string());
  return *it2->second;
}

TraceWriter &
TracingInterface::event_trace_writer() const
{
  neml_assert(_enabled, "Event tracing is not enabled");
  neml_assert(_writer != nullptr, "Event trace writer is not initialized");
  return *_writer;
}
} // namespace neml2
