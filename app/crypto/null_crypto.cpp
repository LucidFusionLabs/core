/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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
#include "core/app/crypto.h"

namespace LFL {
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES256_CBC()   { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::TripDES_CBC()  { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::Blowfish_CBC() { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::RC4()          { FATAL("not implemented"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA1()         { FATAL("not implemented"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA256()       { FATAL("not implemented"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA384()       { FATAL("not implemented"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA512()       { FATAL("not implemented"); }
Crypto::DigestAlgo Crypto::DigestAlgos::MD5()          { FATAL("not implemented"); }
Crypto::MACAlgo    Crypto::   MACAlgos::MD5()          { FATAL("not implemented"); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA1()         { FATAL("not implemented"); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA256()       { FATAL("not implemented"); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA512()       { FATAL("not implemented"); }
int         Crypto::CipherAlgos::KeySize (CipherAlgo v) { return 0; }
int         Crypto::DigestAlgos::HashSize(DigestAlgo v) { return 0; }
int         Crypto::   MACAlgos::HashSize(MACAlgo v) { return 0; }
const char *Crypto::DigestAlgos::Name(DigestAlgo v) { return "none"; }
const char *Crypto::CipherAlgos::Name(CipherAlgo v) { return "none"; }
const char *Crypto::MACAlgos   ::Name(MACAlgo    v) { return "none"; }
Crypto::Cipher Crypto::CipherInit() { FATAL("not implemented"); }
void Crypto::CipherFree(Cipher c) { FATAL("not implemented"); }
int Crypto::CipherGetBlockSize(Cipher c) { FATAL("not implemented"); }
int Crypto::CipherOpen(Cipher c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) {  FATAL("not implemented"); }
int Crypto::CipherUpdate(Cipher c, const StringPiece &in, char *out, int outlen) { FATAL("not implemented"); }
int Crypto::DigestGetHashSize(Digest d) { FATAL("not implemented"); }
Crypto::Digest Crypto::DigestOpen(DigestAlgo algo) { FATAL("not implemented"); }
void Crypto::DigestUpdate(Digest d, const StringPiece &in) { FATAL("not implemented"); }
string Crypto::DigestFinish(Digest d) { FATAL("not implemented"); }
Crypto::MAC Crypto::MACOpen(MACAlgo algo, const StringPiece &k) { FATAL("not implemented"); }
void Crypto::MACUpdate(MAC m, const StringPiece &in) { FATAL("not implemented"); }
int Crypto::MACFinish(MAC m, char *out, int outlen) { FATAL("not implemented"); }
string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }

}; // namespace LFL
