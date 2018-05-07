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

#include <iconv.h>

namespace LFL {
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
  iconv_t cd = iconv_open(to, from);
  if (cd == (iconv_t)-1) { ERROR("failed convert ", from, " to ", to); out->clear(); return 0; }

  out->resize(in.len*4/sizeof(Y)+4);
  char *inp = reinterpret_cast<char*>(const_cast<X*>(in.buf)), *top = reinterpret_cast<char*>(&(*out)[0]);
  size_t in_remaining = in.len*sizeof(X), to_remaining = out->size()*sizeof(Y);
  if (iconv(cd, &inp, &in_remaining, &top, &to_remaining) == -1)
  { ERROR("failed convert ", from, " to ", to); iconv_close(cd); out->clear(); return 0; }
  out->resize(out->size() - to_remaining/sizeof(Y));
  iconv_close(cd);

  return in.len - in_remaining/sizeof(X);
}

template int String::Convert<char,     char    >(const StringPiece  &, string  *, const char*, const char*);
template int String::Convert<char,     char16_t>(const StringPiece  &, String16*, const char*, const char*);
template int String::Convert<char16_t, char    >(const String16Piece&, string  *, const char*, const char*);
template int String::Convert<char16_t, char16_t>(const String16Piece&, String16*, const char*, const char*);

}; // namespace LFL
