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

#include "core/app/crypto.h"

extern "C" {
#define DEF_WEAK(x)
#define SHA512_DIGEST_LENGTH 64
#define explicit_bzero(b, s) memset(b, 0, s)
#define SHA2_CTX LFL::Crypto::Digest
#define SHA512Init(ctx) (*(ctx) = LFL::Crypto::DigestOpen(LFL::Crypto::DigestAlgos::SHA512()))
#define SHA512Update(ctx, b, s) LFL::Crypto::DigestUpdate(*(ctx), LFL::StringPiece(LFL::MakeSigned(b), s))
#define SHA512Final(out, ctx) { std::string v=LFL::Crypto::DigestFinish(*(ctx)); memcpy(out, v.data(), v.size()); }

#include "core/imports/bcrypt_pbkdf/blf.h"
#include "core/imports/bcrypt_pbkdf/blowfish.c"
#include "core/imports/bcrypt_pbkdf/bcrypt_pbkdf.c"
};

namespace LFL {
string Crypto::BCryptPBKDF(const StringPiece &pw, const StringPiece &salt, int size, int rounds) {
  string ret(size, 0);
  if (bcrypt_pbkdf(pw.data(), pw.size(), reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
                   reinterpret_cast<uint8_t*>(&ret[0]), ret.size(), rounds)) return "";
  return ret;
}

}; // namespace LFL
