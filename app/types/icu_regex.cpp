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

#include <unicode/regex.h>

namespace LFL {
Regex::~Regex() { if (auto matcher = static_cast<icu::RegexMatcher*>(impl)) delete matcher; }
Regex::Regex(const Regex &x) {
  auto matcher = static_cast<icu::RegexMatcher*>(x.impl);
  if (!matcher) return;
  string patternstr;
  matcher->pattern().pattern().toUTF8String(patternstr);
  if (patternstr.empty()) return;
  UErrorCode status = U_ZERO_ERROR;
  unique_ptr<icu::RegexMatcher> compiled = make_unique<icu::RegexMatcher>(patternstr.c_str(), 0, status);
  if (!U_FAILURE(status)) impl = compiled.release();
}

Regex::Regex(const string &patternstr) {
  if (patternstr.empty()) return;
  UErrorCode status = U_ZERO_ERROR;
  unique_ptr<icu::RegexMatcher> compiled = make_unique<icu::RegexMatcher>(patternstr.c_str(), 0, status);
  if (!U_FAILURE(status)) impl = compiled.release();
}

Regex::Result Regex::MatchOne(const StringPiece &text) {
  if (!impl || !text.len) return Regex::Result();
  auto matcher = static_cast<icu::RegexMatcher*>(impl);
  matcher->reset(icu::UnicodeString(text.data(), text.size()));
  if (!matcher->find()) return Regex::Result();
  UErrorCode start_status = U_ZERO_ERROR, end_status = U_ZERO_ERROR;
  Regex::Result ret(matcher->start(start_status), matcher->end(end_status));
  return (U_FAILURE(start_status) || U_FAILURE(end_status)) ? Regex::Result() : ret;
}

Regex::Result Regex::MatchOne(const String16Piece &text) {
  static_assert(sizeof(UChar) == sizeof(char16_t), "unexpected sizeof(UChar)");
  if (!impl || !text.len) return Regex::Result();
  auto matcher = static_cast<icu::RegexMatcher*>(impl);
  matcher->reset(icu::UnicodeString(reinterpret_cast<const UChar*>(text.data()), text.size()));
  if (!matcher->find()) return Regex::Result();
  UErrorCode start_status = U_ZERO_ERROR, end_status = U_ZERO_ERROR;
  Regex::Result ret(matcher->start(start_status), matcher->end(end_status));
  return (U_FAILURE(start_status) || U_FAILURE(end_status)) ? Regex::Result() : ret;
}

}; // namespace LFL
