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

#include "neml2/csrc/aoti/log.h"
#include "neml2/csrc/aoti/Exception.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>

namespace neml2::aoti::log
{
namespace
{
constexpr int NCHAN = 6;

int
idx(Channel c)
{
  return static_cast<int>(c);
}

// The default sink: split by severity so normal output is pipeable on stdout and
// diagnostics/warnings stay out of the way on stderr. One flushed line each so a
// nested (linear) residual never lags the Newton iteration it belongs to.
void
default_sink(Level lvl, const std::string & line)
{
  if (lvl == Level::Info)
    std::cout << line << std::endl;
  else
    std::cerr << line << std::endl;
}

// Process-global config. The two layers are `std::optional` per channel plus an
// `all` baseline; effective resolution walks env[c] -> env_all -> prog[c] ->
// prog_all -> built-in default.
struct State
{
  std::optional<Level> env[NCHAN];
  std::optional<Level> env_all;
  std::optional<Level> prog[NCHAN];
  std::optional<Level> prog_all;
  Sink sink = &default_sink;
  bool env_applied = false;
  std::mutex mtx;
};

State &
state()
{
  static State s;
  return s;
}

std::string
trim_lower(const std::string & s)
{
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])))
    ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
    --b;
  std::string out = s.substr(a, b - a);
  for (auto & ch : out)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

// Clear + re-read the env layer from NEML2_LOGS. Caller holds the lock.
void
apply_env_locked(State & s)
{
  for (auto & e : s.env)
    e.reset();
  s.env_all.reset();
  s.env_applied = true;

  const char * raw = std::getenv("NEML2_LOGS");
  if (!raw)
    return;

  const std::string spec(raw);
  std::size_t pos = 0;
  while (pos <= spec.size())
  {
    const std::size_t comma = spec.find(',', pos);
    const std::string entry =
        spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;

    const std::size_t eq = entry.find('=');
    if (eq == std::string::npos)
      continue;
    const std::string chan = trim_lower(entry.substr(0, eq));
    const std::string lvl = trim_lower(entry.substr(eq + 1));
    Level L;
    if (!parse_level(lvl, L))
      throw FatalError("NEML2_LOGS: unknown level '" + lvl + "' in entry '" + trim_lower(entry) +
                       "' (expected silent|warning|info|debug or 0..3)");
    if (chan == "all")
      s.env_all = L;
    else
    {
      Channel C;
      if (!parse_channel(chan, C))
        throw FatalError("NEML2_LOGS: unknown channel '" + chan +
                         "' (expected all|newton|linear|substep|model|tensor|driver)");
      s.env[idx(C)] = L;
    }
  }
}

Level
effective_locked(State & s, Channel c)
{
  if (!s.env_applied)
    apply_env_locked(s);
  const int i = idx(c);
  if (s.env[i])
    return *s.env[i];
  if (s.env_all)
    return *s.env_all;
  if (s.prog[i])
    return *s.prog[i];
  if (s.prog_all)
    return *s.prog_all;
  return Level::Warning;
}
} // namespace

bool
enabled(Channel channel, Level level)
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  return static_cast<int>(effective_locked(s, channel)) >= static_cast<int>(level);
}

Level
effective_level(Channel channel)
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  return effective_locked(s, channel);
}

std::string
format(Channel channel, const std::string & message)
{
  // Pad the channel to the width of the longest name ("substep") so the prefix
  // is column-aligned across channels: "[neml2:newton ] ..." / "[neml2:substep] ...".
  constexpr std::size_t field = 7;
  std::string name = channel_name(channel);
  if (name.size() < field)
    name.append(field - name.size(), ' ');
  return "[neml2:" + name + "] " + message;
}

void
emit(Channel channel, Level level, const std::string & message)
{
  // Copy the sink out under the lock, then call it unlocked: a custom sink may
  // take the GIL (the Python-forwarding sink) or do I/O, and holding our mutex
  // across that would risk a lock-ordering deadlock with the GIL.
  Sink sink_copy;
  bool on = false;
  {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mtx);
    on = static_cast<int>(effective_locked(s, channel)) >= static_cast<int>(level);
    if (on)
      sink_copy = s.sink;
  }
  if (on && sink_copy)
    sink_copy(level, format(channel, message));
}

void
begin_solve(Channel channel, const std::string & label)
{
  emit(channel, Level::Info, "---- begin " + label + " ----");
}

void
end_solve(Channel channel, const std::string & label)
{
  emit(channel, Level::Info, "---- end " + label + " ----");
}

void
set_default_level(Channel channel, Level level)
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.prog[idx(channel)] = level;
}

void
set_default_level(Level level)
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.prog_all = level;
}

void
reset_defaults()
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  for (auto & p : s.prog)
    p.reset();
  s.prog_all.reset();
}

void
set_sink(Sink sink)
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.sink = sink ? std::move(sink) : Sink(&default_sink);
}

void
reset_sink()
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  s.sink = &default_sink;
}

void
apply_env()
{
  auto & s = state();
  std::lock_guard<std::mutex> lock(s.mtx);
  apply_env_locked(s);
}

const char *
channel_name(Channel channel)
{
  switch (channel)
  {
    case Channel::Newton:
      return "newton";
    case Channel::Linear:
      return "linear";
    case Channel::Substep:
      return "substep";
    case Channel::Model:
      return "model";
    case Channel::Tensor:
      return "tensor";
    case Channel::Driver:
      return "driver";
  }
  throw FatalError("neml2::aoti::log::channel_name: unrecognized Channel enum value");
}

const char *
level_name(Level level)
{
  switch (level)
  {
    case Level::Silent:
      return "silent";
    case Level::Warning:
      return "warning";
    case Level::Info:
      return "info";
    case Level::Debug:
      return "debug";
  }
  throw FatalError("neml2::aoti::log::level_name: unrecognized Level enum value");
}

bool
parse_channel(const std::string & s, Channel & out)
{
  if (s == "newton")
    out = Channel::Newton;
  else if (s == "linear")
    out = Channel::Linear;
  else if (s == "substep")
    out = Channel::Substep;
  else if (s == "model")
    out = Channel::Model;
  else if (s == "tensor")
    out = Channel::Tensor;
  else if (s == "driver")
    out = Channel::Driver;
  else
    return false;
  return true;
}

bool
parse_level(const std::string & s, Level & out)
{
  if (s == "silent" || s == "off" || s == "none" || s == "0")
    out = Level::Silent;
  else if (s == "warning" || s == "warn" || s == "1")
    out = Level::Warning;
  else if (s == "info" || s == "2")
    out = Level::Info;
  else if (s == "debug" || s == "3")
    out = Level::Debug;
  else
    return false;
  return true;
}
} // namespace neml2::aoti::log
