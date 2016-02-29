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

#include "core/app/app.h"

namespace LFL {
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
  if (!strcmp(from, to)) { String::Copy(in, out); return in.len; }
#ifdef WIN32
  if (!strcmp(from, "UTF-16LE") && !strcmp(to, "UTF-8")) {
    out->resize(WideCharToMultiByte(CP_UTF8, 0, (wchar_t*)in.data(), in.size(), NULL, 0, NULL, NULL));
    WideCharToMultiByte(CP_UTF8, 0, (wchar_t*)in.data(), in.size(), (char*)&(*out)[0], out->size(), NULL, NULL);
    return in.len;
  }
#endif
  ONCE(ERROR("conversion from ", from, " to ", to, " not supported.  copying.  #define LFL_ICONV"));
  String::Copy(in, out);
  return in.len;    
}

template int String::Convert<char,     char    >(const StringPiece  &, string  *, const char*, const char*);
template int String::Convert<char,     char16_t>(const StringPiece  &, String16*, const char*, const char*);
template int String::Convert<char16_t, char    >(const String16Piece&, string  *, const char*, const char*);
template int String::Convert<char16_t, char16_t>(const String16Piece&, String16*, const char*, const char*);

}; // namespace LFL
