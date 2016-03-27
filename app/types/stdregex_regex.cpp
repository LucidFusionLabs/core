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
  unique_ptr<std::regex> compiled = make_unique<std::regex>(patternstr);
  impl = compiled.release();
}

int Regex::Match(const string &text, vector<Regex::Result> *out) {
  if (!impl) return -1;
  if (out) out->clear();
  auto compiled = static_cast<std::regex*>(impl);
  std::smatch matches;
  if (!std::regex_search(text, matches, *compiled) || matches.size() < 1) return 0;
  if (out) for (int i=0, l=matches.size(); i!=l; i++)
    out->emplace_back(matches[i].first - text.begin(), matches[i].second - text.begin());
  return 1;
}

}; // namespace LFL
