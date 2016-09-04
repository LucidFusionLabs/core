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

bool Crypto::GenerateKey(const string &algo, int bits, const string &pw, const string &comment,
                         string *pubkeyout, string *privkeyout) {
  if (algo == "RSA") {
    RSAKey key = NewRSAPubKey();
    if (1 != RSAGeneratePair(key, bits)) { RSAKeyFree(key); return ERRORv(false, "gen rsa key"); }
    if (pubkeyout)  *pubkeyout  = RSAOpenSSHPublicKey(key, comment);
    if (privkeyout) *privkeyout = RSAPEMPrivateKey(key, pw);
    RSAKeyFree(key);
  } else if (algo == "Ed25519") {
    std::mt19937 rand_eng;
    Ed25519Pair key;
    if (1 != Ed25519GeneratePair(&key, rand_eng)) return ERRORv(false, "gen ed25519 key");
    if (pubkeyout)  *pubkeyout  = Ed25519OpenSSHPublicKey(key, comment);
    if (privkeyout) *privkeyout = Ed25519PEMPrivateKey(key, pw, comment, Rand<int>());
  } else if (algo == "ECDSA") {
    ECPair key;
    if      (bits == 256) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP256(), true);
    else if (bits == 384) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP384(), true);
    else if (bits == 521) key = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP521(), true);
    else return ERRORv(false, "ecdsa bits ", bits);
    if (pubkeyout)  *pubkeyout  = ECDSAOpenSSHPublicKey(key, comment);
    if (privkeyout) *privkeyout = ECDSAPEMPrivateKey(key, pw);
    FreeECPair(key);
  } else return ERRORv(false, "unknown algo ", algo);
  return true;
}

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) {
  char iv[8] = {0,0,0,0,0,0,0,0};
  Cipher cipher = CipherInit();
  CHECK_EQ(1, CipherOpen(cipher, CipherAlgos::Blowfish_CBC(),
                         encrypt_or_decrypt, passphrase, StringPiece(iv, 8), 1, passphrase.size()));
  int outlen = 0, tmplen = 0;
  string out(in.size()+encrypt_or_decrypt*CipherGetBlockSize(cipher), 0);
  outlen = CipherUpdate(cipher, in, &out[0], out.size());
  tmplen = CipherFinal(cipher, &out[0] + outlen, out.size() - outlen);
  CipherFree(cipher);
  if (in.size() % 8) outlen += tmplen;
  if (encrypt_or_decrypt) {
    CHECK_LE(outlen, out.size());
    out.resize(outlen);
  }
  return out;
}

}; // namespace LFL
