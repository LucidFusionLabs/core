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

#include <re2/re2.h>

namespace LFL {
Regex::~Regex() { if (auto compiled = static_cast<RE2*>(impl)) delete compiled; }
Regex::Regex(const Regex &x) : impl(nullptr) { FATAL("RE2 copy not implemented"); }
Regex::Regex(const string &patternstr) {
  if (patternstr.empty()) return;
  unique_ptr<RE2> compiled = make_unique<RE2>(patternstr);
  impl = compiled.release();
}

Regex::Result Regex::MatchOne(const StringPiece &text) {
  if (!impl || !text.len) return Regex::Result();
  auto compiled = static_cast<RE2*>(impl);
  re2::StringPiece match;
  if (!RE2::PartialMatch(re2::StringPiece(text.data(), text.size()), *compiled, &match) ||
      !match.size()) return Regex::Result();
  size_t offset = match.data() - text.data();
  return Regex::Result(offset, offset + match.size());
}

Regex::Result Regex::MatchOne(const String16Piece &text) {
  ERROR("not implemented");
  return Regex::Result();
}

}; // namespace LFL
