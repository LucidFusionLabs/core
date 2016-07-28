/*
 * $Id: openssl_pk.cpp 1330 2014-11-06 03:04:15Z justin $
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
ECPoint NewECPoint(ECGroup g) { return EC_POINT_new(FromVoid<EC_GROUP*>(g)); }
void FreeECPoint(ECPoint p) { if (p) EC_POINT_free(FromVoid<EC_POINT*>(p)); }
void FreeECPair(ECPair p) { if (p) EC_KEY_free(FromVoid<EC_KEY*>(p)); }
ECDef GetECGroupID(ECGroup g) { return Void(intptr_t(EC_GROUP_get_curve_name(FromVoid<EC_GROUP*>(g)))); }
ECGroup GetECPairGroup (ECPair p) { return const_cast<EC_GROUP*>(EC_KEY_get0_group(FromVoid<EC_KEY*>(p))); }
ECPoint GetECPairPubKey(ECPair p) { return const_cast<EC_POINT*>(EC_KEY_get0_public_key(FromVoid<EC_KEY*>(p))); }
bool SetECPairPubKey(ECPair p, ECPoint k) { return EC_KEY_set_public_key(FromVoid<EC_KEY*>(p), FromVoid<EC_POINT*>(k)); }
int ECPointDataSize(ECGroup g, ECPoint p, BigNumContext x) { return EC_POINT_point2oct(FromVoid<EC_GROUP*>(g), FromVoid<EC_POINT*>(p), POINT_CONVERSION_UNCOMPRESSED, 0, 0, FromVoid<BN_CTX*>(x)); }
void ECPointGetData(ECGroup g, ECPoint p, char *out, int len, BigNumContext x) { EC_POINT_point2oct(FromVoid<EC_GROUP*>(g), FromVoid<EC_POINT*>(p), POINT_CONVERSION_UNCOMPRESSED, MakeUnsigned(out), len, FromVoid<BN_CTX*>(x)); }
void ECPointSetData(ECGroup g, ECPoint v, const StringPiece &data) { EC_POINT_oct2point(FromVoid<EC_GROUP*>(g), FromVoid<EC_POINT*>(v), MakeUnsigned(data.buf), data.len, 0); }

RSAKey NewRSAPubKey() { RSA *v=RSA_new(); v->e=FromVoid<BIGNUM*>(NewBigNum()); v->n=FromVoid<BIGNUM*>(NewBigNum()); return RSAKey{v}; }
DSAKey NewDSAPubKey() { DSA *v=DSA_new(); v->p=FromVoid<BIGNUM*>(NewBigNum()); v->q=FromVoid<BIGNUM*>(NewBigNum()); v->g=FromVoid<BIGNUM*>(NewBigNum()); v->pub_key=FromVoid<BIGNUM*>(NewBigNum()); return DSAKey{v}; }
DSASig NewDSASig() { DSA_SIG *v=DSA_SIG_new(); v->r=FromVoid<BIGNUM*>(NewBigNum()); v->s=FromVoid<BIGNUM*>(NewBigNum()); return DSASig{v}; }
ECDSASig NewECDSASig() { return ECDSASig{ECDSA_SIG_new()}; }
BigNum GetRSAKeyE(RSAKey k) { return FromVoid<RSA*>(k)->e; }
BigNum GetRSAKeyN(RSAKey k) { return FromVoid<RSA*>(k)->n; }
BigNum GetDSAKeyP(DSAKey k) { return FromVoid<DSA*>(k)->p; }
BigNum GetDSAKeyQ(DSAKey k) { return FromVoid<DSA*>(k)->q; }
BigNum GetDSAKeyG(DSAKey k) { return FromVoid<DSA*>(k)->g; }
BigNum GetDSAKeyK(DSAKey k) { return FromVoid<DSA*>(k)->pub_key; }
BigNum GetDSASigR(DSASig k) { return FromVoid<DSA_SIG*>(k)->r; }
BigNum GetDSASigS(DSASig k) { return FromVoid<DSA_SIG*>(k)->s; }
BigNum GetECDSASigR(ECDSASig k) { return FromVoid<ECDSA_SIG*>(k)->r; }
BigNum GetECDSASigS(ECDSASig k) { return FromVoid<ECDSA_SIG*>(k)->s; }
void RSAKeyFree(RSAKey k) { RSA_free(FromVoid<RSA*>(k)); }
void DSAKeyFree(DSAKey k) { DSA_free(FromVoid<DSA*>(k)); }
void DSASigFree(DSASig s) { DSA_SIG_free(FromVoid<DSA_SIG*>(s)); }
void ECDSASigFree(ECDSASig s) { ECDSA_SIG_free(FromVoid<ECDSA_SIG*>(s)); }

int RSAVerify(const StringPiece &digest, string *out, RSAKey rsa_key) {
  return RSA_verify(NID_sha1, MakeUnsigned(digest.data()), digest.size(),
                    MakeUnsigned(&(*out)[0]), out->size(), FromVoid<RSA*>(rsa_key));
}

int DSAVerify(const StringPiece &digest, DSASig dsa_sig, DSAKey dsa_key) {
  return DSA_do_verify(MakeUnsigned(digest.data()), digest.size(),
                       FromVoid<DSA_SIG*>(dsa_sig), FromVoid<DSA*>(dsa_key));
}

int ECDSAVerify(const StringPiece &digest, ECDSASig ecdsa_sig, ECPair ecdsa_keypair) {
  return ECDSA_do_verify(MakeUnsigned(digest.data()), digest.size(),
                         FromVoid<ECDSA_SIG*>(ecdsa_sig), FromVoid<EC_KEY*>(ecdsa_keypair));
}

int RSASign(const StringPiece &digest, string *out, RSAKey rsa_key) {
  out->resize(RSA_size(FromVoid<RSA*>(rsa_key)));
  unsigned out_size = out->size();
  int ret = RSA_sign(NID_sha1, MakeUnsigned(digest.data()), digest.size(), MakeUnsigned(&(*out)[0]),
                     &out_size, FromVoid<RSA*>(rsa_key));
  out->resize(out_size);
  return ret;
}

DSASig DSASign(const StringPiece &digest, DSAKey dsa_key) {
  return DSA_do_sign(MakeUnsigned(digest.data()), digest.size(), FromVoid<DSA*>(dsa_key));
}

ECDSASig ECDSASign(const StringPiece &digest, ECPair ecdsa_keypair) {
  return ECDSA_do_sign(MakeUnsigned(digest.data()), digest.size(), FromVoid<EC_KEY*>(ecdsa_keypair));
}

ECDef Crypto::EllipticCurve::NISTP256() { return Void(NID_X9_62_prime256v1); };
ECDef Crypto::EllipticCurve::NISTP384() { return Void(NID_secp384r1); };
ECDef Crypto::EllipticCurve::NISTP521() { return Void(NID_secp521r1); };

ECPair Crypto::EllipticCurve::NewPair(ECDef id, bool generate) {
  EC_KEY *pair = EC_KEY_new_by_curve_name(int(intptr_t(id.v)));
  if (generate && pair && EC_KEY_generate_key(pair) != 1) { EC_KEY_free(pair); return NULL; }
  return ECPair(pair);
}

bool Crypto::ParsePEM(char *key, RSAKey *rsa_out, DSAKey *dsa_out, ECPair *ec_out, function<string(string)> passphrase_cb) {
  ONCE({ OpenSSL_add_all_algorithms(); });
  BIO *bio = BIO_new_mem_buf(key, strlen(key));
  EVP_PKEY *pk = PEM_read_bio_PrivateKey(bio, nullptr, [](char *buf, int size, int rwflag, void *u) {
    string pw = (*reinterpret_cast<decltype(passphrase_cb)*>(u))("");
    int len = min<int>(pw.size(), size);
    memcpy(buf, pw.data(), len);
    return len;
  }, &passphrase_cb);
  if (!pk) { ERROR("PEM_read_bio_PrivateKey: ", Crypto::GetLastErrorText()); BIO_free(bio); return false; }

  bool ret = false;
  int type = EVP_PKEY_base_id(pk);
  if      (type == EVP_PKEY_RSA) { ret = true; *rsa_out = EVP_PKEY_get1_RSA(pk); }
  else if (type == EVP_PKEY_DSA) { ret = true; *dsa_out = EVP_PKEY_get1_DSA(pk); }
  else if (type == EVP_PKEY_EC)  { ret = true; *ec_out  = EVP_PKEY_get1_EC_KEY(pk); }

  EVP_PKEY_free(pk);
  BIO_free(bio);
  return ret;
}

string Crypto::GetLastErrorText() {
  ONCE({ ERR_load_crypto_strings(); });
  string ret(256, 0);
  ERR_error_string_n(ERR_get_error(), &ret[0], ret.size());
  ret.resize(strlen(ret.c_str()));
  return ret;
}

}; // namespace LFL
