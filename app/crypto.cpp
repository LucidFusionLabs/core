/*
 * $Id: crypto.cpp 1335 2014-12-02 04:13:46Z justin $
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

namespace LFL {
string ECPointGetData(ECGroup g, ECPoint p, BigNumContext ctx) {
  string ret(ECPointDataSize(g, p, ctx), 0);
  ECPointGetData(g, p, &ret[0], ret.size(), ctx);
  return ret;
}

string Crypto::MD5   (const string &in) { return ComputeDigest(DigestAlgos::MD5   (), in); }
string Crypto::SHA1  (const string &in) { return ComputeDigest(DigestAlgos::SHA1  (), in); }
string Crypto::SHA256(const string &in) { return ComputeDigest(DigestAlgos::SHA256(), in); }
string Crypto::SHA512(const string &in) { return ComputeDigest(DigestAlgos::SHA512(), in); }

string Crypto::ComputeDigest(DigestAlgo algo, const string &in) {
  Digest d = DigestOpen(algo);
  DigestUpdate(d, in);
  return DigestFinish(d);
}

bool Crypto::GenerateKey(const string &algo, int bits, const string &pw, Crypto::CipherAlgo enc,
                         const string &comment, string *pubkeyout, string *privkeyout) {
  if (algo == "RSA") {
    RSAKey key = NewRSAPubKey();
    if (1 != RSAGeneratePair(key, bits)) { RSAKeyFree(key); return ERRORv(false, "gen rsa key"); }
    if (pubkeyout)  *pubkeyout  = RSAPublicKeyPEM (key);
    if (privkeyout) *privkeyout = RSAPrivateKeyPEM(key, pw, enc);
    RSAKeyFree(key);
  } else if (algo == "Ed25519") {
    std::mt19937 rand_eng;
    Ed25519Pair key;
    if (1 != Ed25519GeneratePair(&key, rand_eng)) return ERRORv(false, "gen ed25519 key");
    if (pubkeyout)  *pubkeyout  = Ed25519PublicKeyPEM (key, comment);
    if (privkeyout) *privkeyout = Ed25519PrivateKeyPEM(key, pw, enc, comment, Rand<int>());
  } else if (algo == "ECDSA") {
    ECPair key;
    if      (bits == 256) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP256(), true);
    else if (bits == 384) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP384(), true);
    else if (bits == 521) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP521(), true);
    else return ERRORv(false, "ecdsa bits ", bits);
    if (pubkeyout)  *pubkeyout  = ECDSAPublicKeyPEM (key);
    if (privkeyout) *privkeyout = ECDSAPrivateKeyPEM(key, pw, enc);
    FreeECPair(key);
  } else return ERRORv(false, "unknown algo ", algo);
  return true;
}

}; // namespace LFL
