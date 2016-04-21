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
Regex::Regex(const string &patternstr) {
  try {
    unique_ptr<std::regex> compiled = make_unique<std::regex>(patternstr);
    impl = compiled.release();
  } catch(std::regex_error e) { ERROR("std::regex ", e.what()); }
}

int Regex::Match(const StringPiece &text, vector<Regex::Result> *out) {
  if (!impl) return 0;
  auto compiled = static_cast<std::regex*>(impl);
  std::match_results<const char*> matches;
  if (!std::regex_search(text.begin(), text.end(), matches, *compiled) || matches.size() < 2) return 0;
  if (out) for (int i=1, l=matches.size(); i!=l; i++)
    out->emplace_back(matches[i].first - text.begin(), matches[i].second - text.begin());
  return 1;
}

int Regex::MatchAll(const StringPiece &text, vector<Regex::Result> *out) {
  out->clear();
  int last_out_size = out->size();
  for (const char *b = text.begin(), *i = b; Match(StringPiece(i, text.end()-i), out) > 0;
       i = b + out->back().end, last_out_size = out->size()) {
    for (auto o = out->begin() + last_out_size, e = out->end(); o != e; ++o) *o += i - b;
  }
  return out->size();
}

}; // namespace LFL
