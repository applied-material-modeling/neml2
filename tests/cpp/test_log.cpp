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

// Unit test for the central verbosity/logging store (log.h). Validates the
// NEML2_LOGS grammar, the env > programmatic > default precedence, the level
// gating, and the settable sink -- the C++ half of the parity with neml2.log.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "neml2/csrc/aoti/log.h"

#include "test_util.h"

namespace L = neml2::aoti::log;

namespace
{
// Set (or clear) NEML2_LOGS and re-parse the env layer.
void
set_env(const char * v)
{
  if (v)
    setenv("NEML2_LOGS", v, 1);
  else
    unsetenv("NEML2_LOGS");
  L::apply_env();
}
} // namespace

int
main()
{
  // Clean slate.
  set_env(nullptr);
  L::reset_defaults();
  L::reset_sink();

  // Built-in default is `warning`.
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Warning);
  NEML2_CHECK(L::enabled(L::Channel::Model, L::Level::Warning));
  NEML2_CHECK(!L::enabled(L::Channel::Model, L::Level::Info));
  NEML2_CHECK(!L::enabled(L::Channel::Newton, L::Level::Debug));

  // Programmatic defaults: per-channel, `all` baseline, explicit beats all.
  L::set_default_level(L::Channel::Newton, L::Level::Debug);
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Debug);
  NEML2_CHECK(L::effective_level(L::Channel::Substep) == L::Level::Warning);
  L::set_default_level(L::Level::Info); // all
  NEML2_CHECK(L::effective_level(L::Channel::Substep) == L::Level::Info);
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Debug);
  L::reset_defaults();
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Warning);

  // Env: `all` baseline + explicit channel override.
  set_env("all=info,newton=silent");
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Silent);
  NEML2_CHECK(L::effective_level(L::Channel::Linear) == L::Level::Info);

  // Env beats programmatic.
  L::set_default_level(L::Channel::Newton, L::Level::Debug);
  set_env("newton=silent");
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Silent);
  L::reset_defaults();

  // Integer + alias level synonyms.
  set_env("newton=2,substep=3,model=off,driver=warn");
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Info);
  NEML2_CHECK(L::effective_level(L::Channel::Substep) == L::Level::Debug);
  NEML2_CHECK(L::effective_level(L::Channel::Model) == L::Level::Silent);
  NEML2_CHECK(L::effective_level(L::Channel::Driver) == L::Level::Warning);

  // A malformed spec is a hard error (unknown level, then unknown channel).
  setenv("NEML2_LOGS", "newton=bogus", 1);
  NEML2_CHECK_THROWS(L::apply_env());
  setenv("NEML2_LOGS", "nosuchchannel=info", 1);
  NEML2_CHECK_THROWS(L::apply_env());
  set_env(nullptr);

  // Custom sink: receives the level + fully-formatted line, gated by level.
  std::vector<std::pair<L::Level, std::string>> events;
  L::set_sink([&](L::Level lv, const std::string & line) { events.emplace_back(lv, line); });
  L::set_default_level(L::Level::Debug); // all=debug
  L::emit(L::Channel::Newton, L::Level::Info, "hi");
  L::emit(L::Channel::Model, L::Level::Warning, "warn");
  L::reset_defaults();
  L::set_default_level(L::Channel::Linear, L::Level::Silent);
  L::emit(L::Channel::Linear, L::Level::Debug, "should-not-emit");
  L::reset_sink();
  NEML2_CHECK(events.size() == 2);
  NEML2_CHECK(events[0].first == L::Level::Info);
  // Channel padded to the longest name ("substep", 7) + level tag padded to 4.
  NEML2_CHECK(events[0].second == "[neml2:newton ][info] hi");
  NEML2_CHECK(events[1].second == "[neml2:model  ][warn] warn");

  // Format prefix: channel + level tag columns, both space-padded (covers every
  // level-tag arm: info, dbg, warn, slnt).
  NEML2_CHECK(L::format(L::Channel::Substep, L::Level::Info, "x") == "[neml2:substep][info] x");
  NEML2_CHECK(L::format(L::Channel::Model, L::Level::Debug, "x") == "[neml2:model  ][dbg ] x");
  NEML2_CHECK(L::format(L::Channel::Newton, L::Level::Warning, "x") == "[neml2:newton ][warn] x");
  NEML2_CHECK(L::format(L::Channel::Newton, L::Level::Silent, "x") == "[neml2:newton ][slnt] x");

  // Parse helpers: `all` is NOT a channel here (handled by the env parser).
  L::Channel c;
  L::Level lv;
  NEML2_CHECK(L::parse_channel("linear", c) && c == L::Channel::Linear);
  NEML2_CHECK(!L::parse_channel("all", c));
  NEML2_CHECK(L::parse_level("debug", lv) && lv == L::Level::Debug);
  NEML2_CHECK(!L::parse_level("bogus", lv));

  // Name lookups round-trip (must not return a sentinel / throw for valid enums).
  NEML2_CHECK(std::string(L::channel_name(L::Channel::Driver)) == "driver");
  NEML2_CHECK(std::string(L::level_name(L::Level::Info)) == "info");

  // Default sink (stdout for info, stderr for warning/debug) + the solve banners:
  // exercise every branch so the writer + banner helpers are covered. Output goes
  // to the test process's own streams (harmless).
  L::reset_sink();
  L::set_default_level(L::Level::Debug);
  L::begin_solve(L::Channel::Newton, "unit solve");
  L::emit(L::Channel::Newton, L::Level::Info, "info-line");    // -> stdout
  L::emit(L::Channel::Newton, L::Level::Warning, "warn-line"); // -> stderr
  L::emit(L::Channel::Newton, L::Level::Debug, "debug-line");  // -> stderr
  L::end_solve(L::Channel::Newton, "unit solve");
  L::reset_defaults();

  // Name + parse round-trip for every channel and every level (covers all switch
  // arms of channel_name/level_name and parse_channel/parse_level).
  const L::Channel all_chans[] = {L::Channel::Newton,
                                  L::Channel::Linear,
                                  L::Channel::Substep,
                                  L::Channel::Model,
                                  L::Channel::Tensor,
                                  L::Channel::Driver};
  for (const auto ch : all_chans)
  {
    L::Channel back;
    NEML2_CHECK(L::parse_channel(L::channel_name(ch), back) && back == ch);
    NEML2_CHECK(!L::format(ch, L::Level::Info, "m").empty());
  }
  const L::Level all_levels[] = {
      L::Level::Silent, L::Level::Warning, L::Level::Info, L::Level::Debug};
  for (const auto lv : all_levels)
  {
    L::Level back;
    NEML2_CHECK(L::parse_level(L::level_name(lv), back) && back == lv);
  }
  // Integer + alias level spellings.
  L::Level lv2;
  NEML2_CHECK(L::parse_level("0", lv2) && lv2 == L::Level::Silent);
  NEML2_CHECK(L::parse_level("off", lv2) && lv2 == L::Level::Silent);
  NEML2_CHECK(L::parse_level("warn", lv2) && lv2 == L::Level::Warning);
  NEML2_CHECK(L::parse_level("3", lv2) && lv2 == L::Level::Debug);

  // Empty / whitespace entries in the env spec are skipped, not errors.
  setenv("NEML2_LOGS", ",newton=info, ,", 1);
  L::apply_env();
  NEML2_CHECK(L::effective_level(L::Channel::Newton) == L::Level::Info);

  L::reset_defaults();
  set_env(nullptr);
  std::printf("test_log: all checks passed\n");
  return 0;
}
