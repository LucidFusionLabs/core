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
#include "core/app/net/ssh.h"
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
static string CompleteBIO(BIO *bio) {
 size_t bio_len = BIO_pending(bio);
 string ret(bio_len, 0);
 BIO_read(bio, &ret[0], bio_len);
 BIO_free(bio);
 return ret;
}

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
bool GetECName(ECDef curve_id, string *algo_name, string *curve_name, Crypto::DigestAlgo *hash_id) {
  if      (curve_id == Crypto::EllipticCurve::NISTP256()) { if (algo_name) *algo_name="ecdsa-sha2-nistp256"; if (curve_name) *curve_name="nistp256"; if (hash_id) *hash_id=Crypto::DigestAlgos::SHA256(); return true; }
  else if (curve_id == Crypto::EllipticCurve::NISTP384()) { if (algo_name) *algo_name="ecdsa-sha2-nistp384"; if (curve_name) *curve_name="nistp384"; if (hash_id) *hash_id=Crypto::DigestAlgos::SHA384(); return true; }
  else if (curve_id == Crypto::EllipticCurve::NISTP521()) { if (algo_name) *algo_name="ecdsa-sha2-nistp521"; if (curve_name) *curve_name="nistp521"; if (hash_id) *hash_id=Crypto::DigestAlgos::SHA512(); return true; }
  else return false;
}

RSAKey NewRSAPubKey() {
  BIGNUM *e = BN_new(), *n = BN_new();
  RSA *v = RSA_new();
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  RSA_set0_key(v, n, e, nullptr);
#else
  v->e = e;
  v->n = n;
#endif
  return RSAKey{v};
}

DSAKey NewDSAPubKey() {
  BIGNUM *p = BN_new(), *q = BN_new(), *g = BN_new(), *pub_key = BN_new();
  DSA *v = DSA_new();
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  DSA_set0_pqg(v, p, q, g);
  DSA_set0_key(v, pub_key, nullptr);
#else
  v->p       = p;
  v->q       = q;
  v->g       = g;
  v->pub_key = pub_key;
#endif
  return DSAKey{v};
}

DSASig NewDSASig() {
  BIGNUM *r = BN_new(), *s = BN_new();
  DSA_SIG *v = DSA_SIG_new();
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  DSA_SIG_set0(v, r, s);
#else
  v->r = r;
  v->s = s;
#endif
  return DSASig{v};
}

ECDSASig NewECDSASig() { return ECDSASig{ECDSA_SIG_new()}; }

BigNum GetRSAKeyE(RSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *e = nullptr;
  RSA_get0_key(FromVoid<RSA*>(k), nullptr, &e, nullptr);
  return const_cast<BIGNUM*>(e);
#else
  return FromVoid<RSA*>(k)->e;
#endif
}

BigNum GetRSAKeyN(RSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *n = nullptr;
  RSA_get0_key(FromVoid<RSA*>(k), &n, nullptr, nullptr);
  return const_cast<BIGNUM*>(n);
#else
  return FromVoid<RSA*>(k)->n;
#endif
}

BigNum GetDSAKeyP(DSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *p = nullptr;
  DSA_get0_pqg(FromVoid<DSA*>(k), &p, nullptr, nullptr);
  return const_cast<BIGNUM*>(p);
#else
  return FromVoid<DSA*>(k)->p;
#endif
}

BigNum GetDSAKeyQ(DSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *q = nullptr;
  DSA_get0_pqg(FromVoid<DSA*>(k), nullptr, &q, nullptr);
  return const_cast<BIGNUM*>(q);
#else
  return FromVoid<DSA*>(k)->q;
#endif
}

BigNum GetDSAKeyG(DSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *g = nullptr;
  DSA_get0_pqg(FromVoid<DSA*>(k), nullptr, nullptr, &g);
  return const_cast<BIGNUM*>(g);
#else
  return FromVoid<DSA*>(k)->g;
#endif
}

BigNum GetDSAKeyK(DSAKey k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *pub_key = nullptr;
  DSA_get0_key(FromVoid<DSA*>(k), &pub_key, nullptr);
  return const_cast<BIGNUM*>(pub_key);
#else
  return FromVoid<DSA*>(k)->pub_key;
#endif
}

BigNum GetDSASigR(DSASig k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *r = nullptr;
  DSA_SIG_get0(FromVoid<DSA_SIG*>(k), &r, nullptr);
  return const_cast<BIGNUM*>(r);
#else
  return FromVoid<DSA_SIG*>(k)->r; 
#endif
}

BigNum GetDSASigS(DSASig k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *s = nullptr;
  DSA_SIG_get0(FromVoid<DSA_SIG*>(k), nullptr, &s);
  return const_cast<BIGNUM*>(s);
#else
  return FromVoid<DSA_SIG*>(k)->s; 
#endif
}

BigNum GetECDSASigR(ECDSASig k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *r = nullptr;
  ECDSA_SIG_get0(FromVoid<ECDSA_SIG*>(k), &r, nullptr);
  return const_cast<BIGNUM*>(r);
#else
  return FromVoid<ECDSA_SIG*>(k)->r;
#endif
}

BigNum GetECDSASigS(ECDSASig k) {
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
  const BIGNUM *s = nullptr;
  ECDSA_SIG_get0(FromVoid<ECDSA_SIG*>(k), nullptr, &s);
  return const_cast<BIGNUM*>(s);
#else
  return FromVoid<ECDSA_SIG*>(k)->s;
#endif
}

void RSAKeyFree(RSAKey k) { RSA_free(FromVoid<RSA*>(k)); }
void DSAKeyFree(DSAKey k) { DSA_free(FromVoid<DSA*>(k)); }
void DSASigFree(DSASig s) { DSA_SIG_free(FromVoid<DSA_SIG*>(s)); }
void ECDSASigFree(ECDSASig s) { ECDSA_SIG_free(FromVoid<ECDSA_SIG*>(s)); }

int RSAGeneratePair(RSAKey rsa_key, int bits) {
  BigNum e = NewBigNum();
  BigNumSetValue(e, RSA_F4);
  int ret = RSAGeneratePair(rsa_key, bits, FromVoid<BIGNUM*>(e));
  FreeBigNum(e);
  return ret;
}

int RSAGeneratePair(RSAKey key, int bits, BigNum e) {
  return RSA_generate_key_ex(FromVoid<RSA*>(key), bits, FromVoid<BIGNUM*>(e), NULL);
}

int DSAGeneratePair(DSAKey key, int bits) {
  DSA_generate_parameters_ex(FromVoid<DSA*>(key), 2048, NULL, 0, NULL, NULL, NULL);
  return DSA_generate_key(FromVoid<DSA*>(key));
}

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

string RSAOpenSSHPublicKey(RSAKey key, const string &comment) {
  string proto = SSH::RSAKey(GetRSAKeyE(key), GetRSAKeyN(key)).ToString();
  string encoded = Singleton<Base64>::Get()->Encode(proto.data(), proto.size());
  return StrCat("ssh-rsa ", encoded, comment.size() ? " " : "", comment, "\n");
}

string DSAOpenSSHPublicKey(DSAKey key, const string &comment) {
  string proto = SSH::DSSKey(GetDSAKeyP(key), GetDSAKeyQ(key), GetDSAKeyG(key), GetDSAKeyK(key)).ToString();
  string encoded = Singleton<Base64>::Get()->Encode(proto.data(), proto.size());
  return StrCat("ssh-dss ", encoded, comment.size() ? " " : "", comment, "\n");
}

string ECDSAOpenSSHPublicKey(ECPair key, const string &comment) {
  ECGroup group = GetECPairGroup(key);
  string algo_name, curve_name;
  if (!GetECName(GetECGroupID(group), &algo_name, &curve_name, nullptr))
    return ERRORv("", "unknown curve_id ", GetECGroupID(group).get());

  BigNumContext ctx = NewBigNumContext();
  string proto = SSH::ECDSAKey(algo_name, curve_name,
                               ECPointGetData(group, GetECPairPubKey(key), ctx)).ToString();
  FreeBigNumContext(ctx);

  string encoded = Singleton<Base64>::Get()->Encode(proto.data(), proto.size());
  return StrCat(algo_name, " ", encoded, comment.size() ? " " : "", comment, "\n");
}

string RSAPEMPublicKey(RSAKey key) {
  BIO *pem = BIO_new(BIO_s_mem());
  PEM_write_bio_RSAPublicKey(pem, FromVoid<RSA*>(key));
  return CompleteBIO(pem);
}

string DSAPEMPublicKey(DSAKey key) {
  BIO *pem = BIO_new(BIO_s_mem());
  PEM_write_bio_DSA_PUBKEY(pem, FromVoid<DSA*>(key));
  return CompleteBIO(pem);
}

string ECDSAPEMPublicKey(ECPair key) {
  BIO *pem = BIO_new(BIO_s_mem());
  EVP_PKEY *pkey = EVP_PKEY_new();
  if (!EVP_PKEY_assign_EC_KEY(pkey, FromVoid<EC_KEY*>(key))) return ERRORv("", "error assigning openssl pkey");
  PEM_write_bio_PUBKEY(pem, pkey);
  EVP_PKEY_free(pkey);
  return CompleteBIO(pem);
}

string RSAPEMPrivateKey(RSAKey key, string pw) {
  BIO *pem = BIO_new(BIO_s_mem());
  PEM_write_bio_RSAPrivateKey(pem, FromVoid<RSA*>(key), pw.size() ? EVP_aes_256_cbc() : nullptr,
                              pw.size() ? MakeUnsigned(&pw[0]) : nullptr, pw.size(), NULL, NULL);
  return CompleteBIO(pem);
}

string DSAPEMPrivateKey(DSAKey key, string pw) {
  BIO *pem = BIO_new(BIO_s_mem());
  PEM_write_bio_DSAPrivateKey(pem, FromVoid<DSA*>(key), pw.size() ? EVP_aes_256_cbc() : nullptr,
                              pw.size() ? MakeUnsigned(&pw[0]) : nullptr, pw.size(), NULL, NULL);
  return CompleteBIO(pem);
}

string ECDSAPEMPrivateKey(ECPair key, string pw) {
  BIO *pem = BIO_new(BIO_s_mem());
  EVP_PKEY *pkey = EVP_PKEY_new();
  if (!EVP_PKEY_set1_EC_KEY(pkey, FromVoid<EC_KEY*>(key))) return ERRORv("", "error assigning openssl pkey");
  if (!PEM_write_bio_PrivateKey(pem, pkey, pw.size() ? EVP_aes_256_cbc() : nullptr,
                                pw.size() ? MakeUnsigned(&pw[0]) : nullptr, pw.size(), NULL, NULL))
    return ERRORv("", "PEM_write_bio_PrivateKey failed: ", Crypto::GetLastErrorText());
  EVP_PKEY_free(pkey);
  return CompleteBIO(pem);
}

void Crypto::PublicKeyInit() { ONCE({ OpenSSL_add_all_algorithms(); }); }
ECDef Crypto::EllipticCurve::NISTP256() { return Void(NID_X9_62_prime256v1); };
ECDef Crypto::EllipticCurve::NISTP384() { return Void(NID_secp384r1); };
ECDef Crypto::EllipticCurve::NISTP521() { return Void(NID_secp521r1); };

ECPair Crypto::EllipticCurve::NewPair(ECDef id, bool generate) {
  EC_KEY *pair = EC_KEY_new_by_curve_name(int(intptr_t(id.get())));
  if (generate && pair) {
    EC_KEY_set_asn1_flag(pair, OPENSSL_EC_NAMED_CURVE);
    if (EC_KEY_generate_key(pair) != 1) { EC_KEY_free(pair); return NULL; }
  }
  return ECPair(pair);
}

bool Crypto::ParsePEM(const char *key, RSAKey *rsa_out, DSAKey *dsa_out, ECPair *ec_out, function<string(string)> passphrase_cb) {
  BIO *bio = BIO_new_mem_buf(const_cast<char*>(key), strlen(key));
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
