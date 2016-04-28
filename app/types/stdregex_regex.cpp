/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <regex>

namespace LFL {
Regex::~Regex() { if (auto compiled = static_cast<std::regex*>(impl)) delete compiled; }
Regex::Regex(const Regex &x) : impl(x.impl ? new std::regex(*static_cast<std::regex*>(x.impl)) : nullptr) {}
Regex::Regex(const string &patternstr) {
  try {
    unique_ptr<std::regex> compiled = make_unique<std::regex>(patternstr);
    impl = compiled.release();
  } catch(std::regex_error e) { ERROR("std::regex ", e.what()); }
}

Regex::Result Regex::MatchOne(const StringPiece &text) {
  if (!impl) return Regex::Result();
  auto compiled = static_cast<std::regex*>(impl);
  std::match_results<const char*> matches;
  if (!std::regex_search(text.begin(), text.end(), matches, *compiled) || matches.size() < 2) return Regex::Result();
  return Regex::Result(matches[1].first - text.begin(), matches[1].second - text.begin());
}

}; // namespace LFL
