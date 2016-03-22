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

#include "regexp.h"

namespace LFL {
Regex::~Regex() { re_free(static_cast<regexp*>(impl)); }
Regex::Regex(const string &patternstr) {
  regexp* compiled = 0;
  if (!re_comp(&compiled, patternstr.c_str())) impl = compiled;
}

int Regex::Match(const string &text, vector<Regex::Result> *out) {
  if (!impl) return -1;
  regexp* compiled = static_cast<regexp*>(impl);
  vector<regmatch> matches(re_nsubexp(compiled));
  int retval = re_exec(compiled, text.c_str(), matches.size(), &matches[0]);
  if (retval < 1) return retval;
  if (out) for (auto i : matches) out->emplace_back(i.begin, i.end);
  return 1;
}

}; // namespace LFL
