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
#include "core/app/crypto.h"
#include "openssl/evp.h"
#include "openssl/bn.h"
#include "openssl/dh.h"
#include "openssl/ec.h"
#include "openssl/ecdh.h"
#include "openssl/rsa.h"
#include "openssl/dsa.h"
#include "openssl/ecdsa.h"
#include "openssl/pem.h"
#include "openssl/err.h"

namespace LFL {
BigNum        NewBigNum       () { return BN_new(); }
BigNumContext NewBigNumContext() { return BN_CTX_new(); }
void FreeBigNumContext(BigNumContext c) { return BN_CTX_free(FromVoid<BN_CTX*>(c)); }
void FreeBigNum(BigNum n) { return BN_free(FromVoid<BIGNUM*>(n)); }
void BigNumModExp(BigNum v, BigNum a, BigNum e, BigNum m, BigNumContext c) { BN_mod_exp(FromVoid<BIGNUM*>(v), FromVoid<BIGNUM*>(a), FromVoid<BIGNUM*>(e), FromVoid<BIGNUM*>(m), FromVoid<BN_CTX*>(c)); }
void BigNumSetValue(BigNum v, int val) { BN_set_word(FromVoid<BIGNUM*>(v), val); }
void BigNumGetData(BigNum v, char *out) { BN_bn2bin(FromVoid<BIGNUM*>(v), MakeUnsigned(out)); }
BigNum BigNumSetData(BigNum v, const StringPiece &data) { BN_bin2bn(MakeUnsigned(data.data()), data.size(), FromVoid<BIGNUM*>(v)); return v; }
BigNum BigNumRand(BigNum v, int bits, int top, int bottom) { BN_rand(FromVoid<BIGNUM*>(v), bits, top, bottom); return v; }
int BigNumDataSize(BigNum v) { return BN_num_bytes(FromVoid<BIGNUM*>(v)); }
int BigNumSignificantBits(BigNum v) { return BN_num_bits(FromVoid<BIGNUM*>(v)); }

bool Crypto::DiffieHellman::GeneratePair(int secret_bits, BigNumContext ctx) {
  x = BigNumRand(x, secret_bits, 0, -1);
  BigNumModExp(e, g, x, p, ctx);
  return true;
}

string Crypto::DiffieHellman::GenerateModulus(int generator, int bits) {
  DH *dh = DH_new();
  DH_generate_parameters_ex(dh, bits, generator, NULL);
  string ret(BN_num_bytes(dh->p), 0);
  BN_bn2bin(dh->p, MakeUnsigned(&ret[0]));
  DH_free(dh);
  return ret;
}

BigNum Crypto::DiffieHellman::Group1Modulus(BigNum g, BigNum p, int *rand_num_bits) {
  // https://tools.ietf.org/html/rfc2409 Second Oakley Group
  char buf[] =
    "\xff\xff\xff\xff\xff\xff\xff\xff\xc9\x0f\xda\xa2\x21\x68\xc2\x34\xc4\xc6\x62\x8b\x80\xdc\x1c\xd1"
    "\x29\x02\x4e\x08\x8a\x67\xcc\x74\x02\x0b\xbe\xa6\x3b\x13\x9b\x22\x51\x4a\x08\x79\x8e\x34\x04\xdd"
    "\xef\x95\x19\xb3\xcd\x3a\x43\x1b\x30\x2b\x0a\x6d\xf2\x5f\x14\x37\x4f\xe1\x35\x6d\x6d\x51\xc2\x45"
    "\xe4\x85\xb5\x76\x62\x5e\x7e\xc6\xf4\x4c\x42\xe9\xa6\x37\xed\x6b\x0b\xff\x5c\xb6\xf4\x06\xb7\xed"
    "\xee\x38\x6b\xfb\x5a\x89\x9f\xa5\xae\x9f\x24\x11\x7c\x4b\x1f\xe6\x49\x28\x66\x51\xec\xe6\x53\x81"
    "\xff\xff\xff\xff\xff\xff\xff\xff";
  BigNumSetValue(g, 2);
  *rand_num_bits = 160;
  return BigNumSetData(p, StringPiece(buf, sizeof(buf)-1));
}

BigNum Crypto::DiffieHellman::Group14Modulus(BigNum g, BigNum p, int *rand_num_bits) {
  // https://tools.ietf.org/html/rfc3526 Oakley Group 14
  char buf[] =
    "\xff\xff\xff\xff\xff\xff\xff\xff\xc9\x0f\xda\xa2\x21\x68\xc2\x34\xc4\xc6\x62\x8b\x80\xdc\x1c\xd1"
    "\x29\x02\x4e\x08\x8a\x67\xcc\x74\x02\x0b\xbe\xa6\x3b\x13\x9b\x22\x51\x4a\x08\x79\x8e\x34\x04\xdd"
    "\xef\x95\x19\xb3\xcd\x3a\x43\x1b\x30\x2b\x0a\x6d\xf2\x5f\x14\x37\x4f\xe1\x35\x6d\x6d\x51\xc2\x45"
    "\xe4\x85\xb5\x76\x62\x5e\x7e\xc6\xf4\x4c\x42\xe9\xa6\x37\xed\x6b\x0b\xff\x5c\xb6\xf4\x06\xb7\xed"
    "\xee\x38\x6b\xfb\x5a\x89\x9f\xa5\xae\x9f\x24\x11\x7c\x4b\x1f\xe6\x49\x28\x66\x51\xec\xe4\x5b\x3d"
    "\xc2\x00\x7c\xb8\xa1\x63\xbf\x05\x98\xda\x48\x36\x1c\x55\xd3\x9a\x69\x16\x3f\xa8\xfd\x24\xcf\x5f"
    "\x83\x65\x5d\x23\xdc\xa3\xad\x96\x1c\x62\xf3\x56\x20\x85\x52\xbb\x9e\xd5\x29\x07\x70\x96\x96\x6d"
    "\x67\x0c\x35\x4e\x4a\xbc\x98\x04\xf1\x74\x6c\x08\xca\x18\x21\x7c\x32\x90\x5e\x46\x2e\x36\xce\x3b"
    "\xe3\x9e\x77\x2c\x18\x0e\x86\x03\x9b\x27\x83\xa2\xec\x07\xa2\x8f\xb5\xc5\x5d\xf0\x6f\x4c\x52\xc9"
    "\xde\x2b\xcb\xf6\x95\x58\x17\x18\x39\x95\x49\x7c\xea\x95\x6a\xe5\x15\xd2\x26\x18\x98\xfa\x05\x10"
    "\x15\x72\x8e\x5a\x8a\xac\xaa\x68\xff\xff\xff\xff\xff\xff\xff\xff";
  BigNumSetValue(g, 2);
  *rand_num_bits = 224;
  return BigNumSetData(p, StringPiece(buf, sizeof(buf)-1));
}

bool Crypto::EllipticCurveDiffieHellman::GeneratePair(ECDef curve, BigNumContext ctx) {
  FreeECPair(pair);
  if (!(pair = Crypto::EllipticCurve::NewPair(curve, true))) return false;
  g = GetECPairGroup(pair);
  c = GetECPairPubKey(pair);
  c_text = ECPointGetData(FromVoid<EC_GROUP*>(g), c, ctx);
  FreeECPoint(s);
  s = NewECPoint(g);
  return true;
}

bool Crypto::EllipticCurveDiffieHellman::ComputeSecret(BigNum *K, BigNumContext ctx) {
  string k_text((EC_GROUP_get_degree(FromVoid<EC_GROUP*>(g)) + 7) / 8, 0);
  if (ECDH_compute_key(&k_text[0], k_text.size(), FromVoid<const EC_POINT*>(s),
                       FromVoid<EC_KEY*>(pair), 0) != k_text.size()) return false;
  *K = BigNumSetData(*K, k_text);
  return true;
}

}; // namespace LFL
