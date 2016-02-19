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

#include "lfapp/lfapp.h"
#include "lfapp/crypto.h"

#ifdef LFL_OPENSSL
#include "openssl/evp.h"
#include "openssl/err.h"
#include "openssl/bn.h"
#include "openssl/dh.h"
#include "openssl/ec.h"
#include "openssl/ecdh.h"
#include "openssl/md5.h"
#include "openssl/rsa.h"
#include "openssl/dsa.h"
#include "openssl/ecdsa.h"
#endif

namespace LFL {
string Crypto::MD5   (const string &in) { return ComputeDigest(DigestAlgos::MD5   (), in); }
string Crypto::SHA1  (const string &in) { return ComputeDigest(DigestAlgos::SHA1  (), in); }
string Crypto::SHA256(const string &in) { return ComputeDigest(DigestAlgos::SHA256(), in); }

string Crypto::ComputeDigest(DigestAlgo algo, const string &in) {
  Digest d;
  DigestOpen(&d, algo);
  DigestUpdate(&d, in);
  return DigestFinish(&d);
}

#ifdef LFL_OPENSSL
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

ECDef Crypto::EllipticCurve::NISTP256() { return NID_X9_62_prime256v1; };
ECDef Crypto::EllipticCurve::NISTP384() { return NID_secp384r1; };
ECDef Crypto::EllipticCurve::NISTP521() { return NID_secp521r1; };

ECPair Crypto::EllipticCurve::NewPair(ECDef id, bool generate) {
  ECPair pair = EC_KEY_new_by_curve_name(id);
  if (generate && pair && EC_KEY_generate_key(pair) != 1) { EC_KEY_free(pair); return NULL; }
  return pair;
}

bool Crypto::EllipticCurveDiffieHellman::GeneratePair(ECDef curve, BigNumContext ctx) {
  FreeECPair(pair);
  if (!(pair = Crypto::EllipticCurve::NewPair(curve, true))) return false;
  g = GetECPairGroup(pair);
  c = GetECPairPubKey(pair);
  c_text = ECPointGetData(g, c, ctx);
  FreeECPoint(s);
  s = NewECPoint(g);
  return true;
}

bool Crypto::EllipticCurveDiffieHellman::ComputeSecret(BigNum *K, BigNumContext ctx) {
  string k_text((EC_GROUP_get_degree(g) + 7) / 8, 0);
  if (ECDH_compute_key(&k_text[0], k_text.size(), s, pair, 0) != k_text.size()) return false;
  *K = BigNumSetData(*K, k_text);
  return true;
}

#else

bool Crypto::DiffieHellman::GeneratePair(int secret_bits, BigNumContext ctx) { FATAL("not implemented"); }
BigNum Crypto::DiffieHellman::Group1Modulus(BigNum g, BigNum p, int *rand_num_bits) { FATAL("not implemented"); }
BigNum Crypto::DiffieHellman::Group14Modulus(BigNum g, BigNum p, int *rand_num_bits) { FATAL("not implemented"); }
ECDef Crypto::EllipticCurve::NISTP256() { FATAL("not implemented"); }
ECDef Crypto::EllipticCurve::NISTP384() { FATAL("not implemented"); }
ECDef Crypto::EllipticCurve::NISTP521() { FATAL("not implemented"); }
ECPair Crypto::EllipticCurve::NewPair(ECDef id, bool generate) { FATAL("not implemented"); }
bool Crypto::EllipticCurveDiffieHellman::GeneratePair(ECDef curve, BigNumContext ctx) { FATAL("not implemented"); }
bool Crypto::EllipticCurveDiffieHellman::ComputeSecret(BigNum *K, BigNumContext ctx) { FATAL("not implemented"); }
#endif

#if defined(LFL_COMMONCRYPTO)
struct CCCipherAlgo { enum { AES128_CTR=1, AES128_CBC=2, TripDES_CBC=3, Blowfish_CBC=4, RC4=5 }; };
struct CCDigestAlgo { enum { MD5=1, SHA1=2, SHA256=3, SHA384=4, SHA512=5 }; };
string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { return CCCipherAlgo::AES128_CTR; }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { return CCCipherAlgo::AES128_CBC; }
Crypto::CipherAlgo Crypto::CipherAlgos::TripDES_CBC()  { return CCCipherAlgo::TripDES_CBC; }
Crypto::CipherAlgo Crypto::CipherAlgos::Blowfish_CBC() { return CCCipherAlgo::Blowfish_CBC; }
Crypto::CipherAlgo Crypto::CipherAlgos::RC4()          { return CCCipherAlgo::RC4; }
Crypto::DigestAlgo Crypto::DigestAlgos::MD5()          { return CCDigestAlgo::MD5; }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA1()         { return CCDigestAlgo::SHA1; }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA256()       { return CCDigestAlgo::SHA256; }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA384()       { return CCDigestAlgo::SHA384; }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA512()       { return CCDigestAlgo::SHA512; }
Crypto::MACAlgo    Crypto::   MACAlgos::MD5()          { return kCCHmacAlgMD5; }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA1()         { return kCCHmacAlgSHA1; }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA256()       { return kCCHmacAlgSHA256; }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA512()       { return kCCHmacAlgSHA512; }

const char *Crypto::CipherAlgos::Name(CipherAlgo v) {
  switch (v) {
    case CCCipherAlgo::AES128_CTR:   return "aes128-ctr";
    case CCCipherAlgo::AES128_CBC:   return "aes128-cbc";
    case CCCipherAlgo::TripDES_CBC:  return "3des-cbc";
    case CCCipherAlgo::Blowfish_CBC: return "blowfish-cbc";
    case CCCipherAlgo::RC4:          return "rc4";
    default:                         return "none";
  }
}

int Crypto::CipherAlgos::KeySize(CipherAlgo v) {
  switch (v) {
    case CCCipherAlgo::AES128_CTR:   return kCCKeySizeAES128;
    case CCCipherAlgo::AES128_CBC:   return kCCKeySizeAES128;
    case CCCipherAlgo::TripDES_CBC:  return kCCKeySize3DES;
    case CCCipherAlgo::Blowfish_CBC: return 16;
    case CCCipherAlgo::RC4:          return 16;
    default:                         return 0;
  }
}

const char *Crypto::DigestAlgos::Name(DigestAlgo v) {
  switch (v) {
    case CCDigestAlgo::MD5:    return "md5";
    case CCDigestAlgo::SHA1:   return "sha1";
    case CCDigestAlgo::SHA256: return "sha256";
    case CCDigestAlgo::SHA384: return "sha384";
    case CCDigestAlgo::SHA512: return "sha512";
    default:                   return "none";
  }
}

int Crypto::DigestAlgos::HashSize(DigestAlgo v) {
  switch (v) {
    case CCDigestAlgo::MD5:    return CC_MD5_DIGEST_LENGTH;
    case CCDigestAlgo::SHA1:   return CC_SHA1_DIGEST_LENGTH;
    case CCDigestAlgo::SHA256: return CC_SHA256_DIGEST_LENGTH;
    case CCDigestAlgo::SHA384: return CC_SHA384_DIGEST_LENGTH;
    case CCDigestAlgo::SHA512: return CC_SHA512_DIGEST_LENGTH;
    default:                   return 0;
  }
}

const char *Crypto::MACAlgos::Name(MACAlgo v) {
  switch (v) {
    case kCCHmacAlgMD5:    return "md5";
    case kCCHmacAlgSHA1:   return "sha1";
    case kCCHmacAlgSHA256: return "sha256";
    case kCCHmacAlgSHA512: return "sha512";
    default:               return "none";
  }
}

int Crypto::MACAlgos::HashSize(MACAlgo v) {
  switch (v) {
    case kCCHmacAlgMD5:    return CC_MD5_DIGEST_LENGTH;
    case kCCHmacAlgSHA1:   return CC_SHA1_DIGEST_LENGTH;
    case kCCHmacAlgSHA256: return CC_SHA256_DIGEST_LENGTH;
    case kCCHmacAlgSHA512: return CC_SHA512_DIGEST_LENGTH;
    default:               return 0;
  }
}

void Crypto::CipherInit(Cipher *c) { c->algo=0; c->ctx=0; }
void Crypto::CipherFree(Cipher *c) { CCCryptorRelease(c->ctx); }
int Crypto::CipherGetBlockSize(Cipher *c) {
  switch(c->ccalgo) {
    case kCCAlgorithmAES128:   return kCCBlockSizeAES128;
    case kCCAlgorithm3DES:     return kCCBlockSize3DES;
    case kCCAlgorithmBlowfish: return kCCBlockSizeBlowfish;
    case kCCAlgorithmRC4:      return 16;
    default:                   return -1;
  }
}

int Crypto::CipherOpen(Cipher *c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) {
  bool ctr = false;
  switch((c->algo = algo)) {
    case CCCipherAlgo::AES128_CTR:   c->ccalgo = kCCAlgorithmAES128; ctr = true; break;
    case CCCipherAlgo::AES128_CBC:   c->ccalgo = kCCAlgorithmAES128;             break;
    case CCCipherAlgo::TripDES_CBC:  c->ccalgo = kCCAlgorithm3DES;               break;
    case CCCipherAlgo::Blowfish_CBC: c->ccalgo = kCCAlgorithmBlowfish;           break;
    case CCCipherAlgo::RC4:          c->ccalgo = kCCAlgorithmRC4;                break;
    default:                         return -1;
  }
  int mode = (algo == CCCipherAlgo::RC4) ? kCCModeRC4 : (ctr ? kCCModeCTR : kCCModeCBC);
  return CCCryptorCreateWithMode(dir ? kCCEncrypt : kCCDecrypt, mode, c->ccalgo, 0, IV.data(), key.data(), key.size(),
                                 0, 0, 0, ctr ? kCCModeOptionCTR_BE : 0, &c->ctx) == kCCSuccess;
}

int Crypto::CipherUpdate(Cipher *c, const StringPiece &in, char *out, int outlen) {
  size_t wrote = 0;
  return CCCryptorUpdate(c->ctx, in.data(), in.size(), out, outlen, &wrote) == kCCSuccess;
}

int Crypto::DigestGetHashSize(Digest *d) { return DigestAlgos::HashSize(d->algo); }
void Crypto::DigestOpen(Digest *d, DigestAlgo algo) {
  d->algo = algo;
  switch(algo) {
    case CCDigestAlgo::MD5:    d->v=calloc(sizeof(CC_MD5_CTX),   1); CC_MD5_Init   (FromVoid<CC_MD5_CTX*>   (d->v)); break;
    case CCDigestAlgo::SHA1:   d->v=calloc(sizeof(CC_SHA1_CTX),  1); CC_SHA1_Init  (FromVoid<CC_SHA1_CTX*>  (d->v)); break;
    case CCDigestAlgo::SHA256: d->v=calloc(sizeof(CC_SHA256_CTX),1); CC_SHA256_Init(FromVoid<CC_SHA256_CTX*>(d->v)); break;
    case CCDigestAlgo::SHA384: d->v=calloc(sizeof(CC_SHA512_CTX),1); CC_SHA384_Init(FromVoid<CC_SHA512_CTX*>(d->v)); break;
    case CCDigestAlgo::SHA512: d->v=calloc(sizeof(CC_SHA512_CTX),1); CC_SHA512_Init(FromVoid<CC_SHA512_CTX*>(d->v)); break;
    default:                   d->v=0; break;
  }
}

void Crypto::DigestUpdate(Digest *d, const StringPiece &in) {
  switch(d->algo) {
    case CCDigestAlgo::MD5:    CC_MD5_Update   (FromVoid<CC_MD5_CTX*>   (d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA1:   CC_SHA1_Update  (FromVoid<CC_SHA1_CTX*>  (d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA256: CC_SHA256_Update(FromVoid<CC_SHA256_CTX*>(d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA384: CC_SHA384_Update(FromVoid<CC_SHA512_CTX*>(d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA512: CC_SHA512_Update(FromVoid<CC_SHA512_CTX*>(d->v), in.data(), in.size()); break;
    default: break;
  }
}

string Crypto::DigestFinish(Digest *d) {
  string ret;
  switch(d->algo) {
    case CCDigestAlgo::MD5:    ret.resize(CC_MD5_DIGEST_LENGTH);    CC_MD5_Final   (MakeUnsigned(&ret[0]), FromVoid<CC_MD5_CTX*>   (d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA1:   ret.resize(CC_SHA1_DIGEST_LENGTH);   CC_SHA1_Final  (MakeUnsigned(&ret[0]), FromVoid<CC_SHA1_CTX*>  (d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA256: ret.resize(CC_SHA256_DIGEST_LENGTH); CC_SHA256_Final(MakeUnsigned(&ret[0]), FromVoid<CC_SHA256_CTX*>(d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA384: ret.resize(CC_SHA384_DIGEST_LENGTH); CC_SHA384_Final(MakeUnsigned(&ret[0]), FromVoid<CC_SHA512_CTX*>(d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA512: ret.resize(CC_SHA512_DIGEST_LENGTH); CC_SHA512_Final(MakeUnsigned(&ret[0]), FromVoid<CC_SHA512_CTX*>(d->v)); free(d->v); d->v=0; break;
    default: break;
  }
  return ret;
}

void Crypto::MACOpen(MAC *m, MACAlgo algo, const StringPiece &k) { CCHmacInit(&m->ctx, (m->algo=algo), k.data(), k.size()); }
void Crypto::MACUpdate(MAC *m, const StringPiece &in) { CCHmacUpdate(&m->ctx, in.data(), in.size()); }
int Crypto::MACFinish(MAC *m, char *out, int outlen) {
  CCHmacFinal(&m->ctx, out); 
  switch(m->algo) {
    case kCCHmacAlgMD5:    return CC_MD5_DIGEST_LENGTH;
    case kCCHmacAlgSHA1:   return CC_SHA1_DIGEST_LENGTH;
    case kCCHmacAlgSHA256: return CC_SHA256_DIGEST_LENGTH;
    case kCCHmacAlgSHA512: return CC_SHA512_DIGEST_LENGTH;
    default:               return -1;
  }
}

#elif defined(LFL_OPENSSL)

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) {
  unsigned char iv[8] = {0,0,0,0,0,0,0,0};
  EVP_CIPHER_CTX ctx; 
  EVP_CIPHER_CTX_init(&ctx); 
  EVP_CipherInit_ex(&ctx, EVP_bf_cbc(), NULL, NULL, NULL, encrypt_or_decrypt);
  EVP_CIPHER_CTX_set_key_length(&ctx, passphrase.size());
  EVP_CipherInit_ex(&ctx, NULL, NULL, (const unsigned char *)passphrase.c_str(), iv, encrypt_or_decrypt); 

  int outlen = 0, tmplen = 0;
  string out(in.size()+encrypt_or_decrypt*EVP_MAX_BLOCK_LENGTH, 0);
  EVP_CipherUpdate(&ctx, MakeUnsigned(&out[0]), &outlen, (const unsigned char *)in.c_str(), in.size());
  EVP_CipherFinal_ex(&ctx, MakeUnsigned(&out[0]) + outlen, &tmplen); 
  if (in.size() % 8) outlen += tmplen;

  EVP_CIPHER_CTX_cleanup(&ctx); 
  if (encrypt_or_decrypt) {
    CHECK_LE(outlen, out.size());
    out.resize(outlen);
  }
  return out;
}

Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { return EVP_aes_128_ctr(); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { return EVP_aes_128_cbc(); }
Crypto::CipherAlgo Crypto::CipherAlgos::TripDES_CBC()  { return EVP_des_ede3_cbc(); }
Crypto::CipherAlgo Crypto::CipherAlgos::Blowfish_CBC() { return EVP_bf_cbc(); }
Crypto::CipherAlgo Crypto::CipherAlgos::RC4()          { return EVP_rc4(); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA1()         { return EVP_get_digestbyname("sha1"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA256()       { return EVP_get_digestbyname("sha256"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA384()       { return EVP_get_digestbyname("sha384"); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA512()       { return EVP_get_digestbyname("sha512"); }
Crypto::DigestAlgo Crypto::DigestAlgos::MD5()          { return EVP_md5(); }
Crypto::MACAlgo    Crypto::MACAlgos   ::MD5()          { return EVP_md5(); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA1()         { return EVP_sha1(); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA256()       { return EVP_sha256(); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA512()       { return EVP_sha512(); }
int         Crypto::CipherAlgos::KeySize (CipherAlgo v) { return EVP_CIPHER_key_length(v); }
int         Crypto::DigestAlgos::HashSize(DigestAlgo v) { return EVP_MD_size(v); }
int         Crypto::MACAlgos   ::HashSize(MACAlgo    v) { return EVP_MD_size(v); }
const char *Crypto::DigestAlgos::Name(DigestAlgo v) { return EVP_MD_name(v); }
const char *Crypto::CipherAlgos::Name(CipherAlgo v) { return EVP_CIPHER_name(v); }
const char *Crypto::MACAlgos   ::Name(MACAlgo    v) { return EVP_MD_name(v); }
void Crypto::CipherInit(Cipher *c) { EVP_CIPHER_CTX_init(c); }
void Crypto::CipherFree(Cipher *c) { EVP_CIPHER_CTX_cleanup(c); }
int Crypto::CipherGetBlockSize(Cipher *c) { return EVP_CIPHER_CTX_block_size(c); }
int Crypto::CipherOpen(Cipher *c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) { 
  return EVP_CipherInit(c, algo, MakeUnsigned(key.data()), MakeUnsigned(IV.data()), dir);
}
int Crypto::CipherUpdate(Cipher *c, const StringPiece &in, char *out, int outlen) {
  return EVP_Cipher(c, MakeUnsigned(out), MakeUnsigned(in.data()), in.size());
}

int Crypto::DigestGetHashSize(Digest *d) { return EVP_MD_CTX_size(d); }
void Crypto::DigestOpen(Digest *d, DigestAlgo algo) { CHECK(algo); EVP_DigestInit(d, algo); }
void Crypto::DigestUpdate(Digest *d, const StringPiece &in) { EVP_DigestUpdate(d, in.data(), in.size()); }
string Crypto::DigestFinish(Digest *d) {
  unsigned len = 0;
  string ret(EVP_MAX_MD_SIZE, 0);
  EVP_DigestFinal(d, MakeUnsigned(&ret[0]), &len);
  ret.resize(len);
  return ret;
}

void Crypto::MACOpen(MAC *m, MACAlgo algo, const StringPiece &k) { HMAC_Init(m, k.data(), k.size(), algo); }
void Crypto::MACUpdate(MAC *m, const StringPiece &in) { HMAC_Update(m, MakeUnsigned(in.data()), in.size()); }
int Crypto::MACFinish(MAC *m, char *out, int outlen) { unsigned len=outlen; HMAC_Final(m, MakeUnsigned(out), &len); return len; }

#else

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { FATAL("not implemented"); }
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
int         Crypto::   MACAlgos::HashSize(DigestAlgo v) { return 0; }
const char *Crypto::DigestAlgos::Name(DigestAlgo v) { return "none"; }
const char *Crypto::CipherAlgos::Name(CipherAlgo v) { return "none"; }
const char *Crypto::MACAlgos   ::Name(MACAlgo    v) { return "none"; }
void Crypto::CipherInit(Cipher *c) { FATAL("not implemented"); }
void Crypto::CipherFree(Cipher *c) { FATAL("not implemented"); }
int Crypto::CipherGetBlockSize(Cipher *c) { FATAL("not implemented"); }
int Crypto::CipherOpen(Cipher *c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) {  FATAL("not implemented"); }
int Crypto::CipherUpdate(Cipher *c, const StringPiece &in, char *out, int outlen) { FATAL("not implemented"); }
int Crypto::DigestGetHashSize(Digest *d) { FATAL("not implemented"); }
void Crypto::DigestOpen(Digest *d, DigestAlgo algo) { FATAL("not implemented"); }
void Crypto::DigestUpdate(Digest *d, const StringPiece &in) { FATAL("not implemented"); }
string Crypto::DigestFinish(Digest *d) { FATAL("not implemented"); }
void Crypto::MACOpen(MAC *m, MACAlgo algo, const StringPiece &k) { FATAL("not implemented"); }
void Crypto::MACUpdate(MAC *m, const StringPiece &in) { FATAL("not implemented"); }
int Crypto::MACFinish(MAC *m, char *out, int outlen) { FATAL("not implemented"); }
#endif

int SSH::BinaryPacketLength(const char *b, unsigned char *padding, unsigned char *id) {
  if (padding) *padding = *MakeUnsigned(b + 4);
  if (id)      *id      = *MakeUnsigned(b + 5);
  return ntohl(*reinterpret_cast<const int*>(b));
}

int SSH::BigNumSize(const BigNum n) { return BigNumDataSize(n) + !(BigNumSignificantBits(n) % 8); }

BigNum SSH::ReadBigNum(BigNum n, const Serializable::Stream *i) {
  int n_len = 0;
  i->Ntohl(&n_len);
  return BigNumSetData(n, StringPiece(i->Get(n_len), n_len));
}

void SSH::WriteBigNum(const BigNum n, Serializable::Stream *o) {
  int n_len = BigNumDataSize(n);
  bool prepend_zero = !(BigNumSignificantBits(n) % 8);
  o->Htonl(n_len + prepend_zero);
  if (prepend_zero) o->Write8(char(0));
  BigNumGetData(n, o->Get(n_len));
}

void SSH::UpdateDigest(Crypto::Digest *d, const StringPiece &s) {
  UpdateDigest(d, s.size());
  Crypto::DigestUpdate(d, s);
}

void SSH::UpdateDigest(Crypto::Digest *d, int n) {
  char buf[4];
  Serializable::MutableStream(buf, 4).Htonl(n);
  Crypto::DigestUpdate(d, StringPiece(buf, 4));
}

void SSH::UpdateDigest(Crypto::Digest *d, BigNum n) {
  string buf(4 + SSH::BigNumSize(n), 0);
  Serializable::MutableStream o(&buf[0], buf.size());
  SSH::WriteBigNum(n, &o);
  Crypto::DigestUpdate(d, buf);
}

#ifdef LFL_OPENSSL
static RSA *NewRSAPubKey() { RSA *v=RSA_new(); v->e=NewBigNum(); v->n=NewBigNum(); return v; }
static DSA *NewDSAPubKey() { DSA *v=DSA_new(); v->p=NewBigNum(); v->q=NewBigNum(); v->g=NewBigNum(); v->pub_key=NewBigNum(); return v; }
static DSA_SIG *NewDSASig() { DSA_SIG *v=DSA_SIG_new(); v->r=NewBigNum(); v->s=NewBigNum(); return v; }
#endif

int SSH::VerifyHostKey(const string &H_text, int hostkey_type, const StringPiece &key, const StringPiece &sig) {
#ifdef LFL_OPENSSL
  if (hostkey_type == SSH::Key::RSA) {
    string H_hash = Crypto::SHA1(H_text);
    RSA *rsa_key = NewRSAPubKey();
    SSH::RSASignature rsa_sig;
    Serializable::ConstStream rsakey_stream(key.data(), key.size());
    Serializable::ConstStream rsasig_stream(sig.data(), sig.size());
    if (rsa_sig.In(&rsasig_stream)) { RSA_free(rsa_key); return -3; }
    string rsa_sigbuf(rsa_sig.sig.data(), rsa_sig.sig.size());
    if (SSH::RSAKey(rsa_key->e, rsa_key->n).In(&rsakey_stream)) { RSA_free(rsa_key); return -2; }
    int verified = RSA_verify(NID_sha1, MakeUnsigned(H_hash.data()), H_hash.size(),
                              MakeUnsigned(&rsa_sigbuf[0]), rsa_sigbuf.size(), rsa_key);
    RSA_free(rsa_key);
    return verified;

  } else if (hostkey_type == SSH::Key::DSS) {
    string H_hash = Crypto::SHA1(H_text);
    DSA *dsa_key = NewDSAPubKey();
    DSA_SIG *dsa_sig = NewDSASig();
    Serializable::ConstStream dsakey_stream(key.data(), key.size());
    Serializable::ConstStream dsasig_stream(sig.data(), sig.size());
    if (SSH::DSSKey(dsa_key->p, dsa_key->q, dsa_key->g, dsa_key->pub_key).In(&dsakey_stream)) { DSA_free(dsa_key); return -4; }
    if (SSH::DSSSignature(dsa_sig->r, dsa_sig->s).In(&dsasig_stream)) { DSA_free(dsa_key); DSA_SIG_free(dsa_sig); return -5; }
    int verified = DSA_do_verify(MakeUnsigned(H_hash.data()), H_hash.size(), dsa_sig, dsa_key);
    DSA_free(dsa_key);
    DSA_SIG_free(dsa_sig);
    return verified;
    
  } else if (hostkey_type == SSH::Key::ECDSA_SHA2_NISTP256) {
    string H_hash = Crypto::SHA256(H_text);
    SSH::ECDSAKey key_msg;
    ECDSA_SIG *ecdsa_sig = ECDSA_SIG_new();
    Serializable::ConstStream ecdsakey_stream(key.data(), key.size());
    Serializable::ConstStream ecdsasig_stream(sig.data(), sig.size());
    if (key_msg.In(&ecdsakey_stream)) { ECDSA_SIG_free(ecdsa_sig); return -6; }
    if (SSH::ECDSASignature(ecdsa_sig->r, ecdsa_sig->s).In(&ecdsasig_stream)) { ECDSA_SIG_free(ecdsa_sig); return -7; }
    ECPair ecdsa_keypair = Crypto::EllipticCurve::NewPair(Crypto::EllipticCurve::NISTP256(), false);
    ECPoint ecdsa_key = NewECPoint(GetECPairGroup(ecdsa_keypair));
    ECPointSetData(GetECPairGroup(ecdsa_keypair), ecdsa_key, key_msg.q);
    if (!SetECPairPubKey(ecdsa_keypair, ecdsa_key)) { FreeECPair(ecdsa_keypair); ECDSA_SIG_free(ecdsa_sig); return -8; }
    int verified = ECDSA_do_verify(MakeUnsigned(H_hash.data()), H_hash.size(), ecdsa_sig, ecdsa_keypair);
    FreeECPair(ecdsa_keypair);
    ECDSA_SIG_free(ecdsa_sig);
    return verified;

  } else return -9;
#else
  return -10;
#endif
}

string SSH::ComputeExchangeHash(int kex_method, Crypto::DigestAlgo algo, const string &V_C, const string &V_S,
                                const string &KI_C, const string &KI_S, const StringPiece &k_s, BigNum K,
                                Crypto::DiffieHellman *dh, Crypto::EllipticCurveDiffieHellman *ecdh) {
  string ret;
  unsigned char kex_c_padding = 0, kex_s_padding = 0;
  int kex_c_packet_len = 4 + SSH::BinaryPacketLength(KI_C.data(), &kex_c_padding, NULL);
  int kex_s_packet_len = 4 + SSH::BinaryPacketLength(KI_S.data(), &kex_s_padding, NULL);
  Crypto::Digest H;
  Crypto::DigestOpen(&H, algo);
  UpdateDigest(&H, V_C);
  UpdateDigest(&H, V_S);
  UpdateDigest(&H, StringPiece(KI_C.data() + 5, kex_c_packet_len - 5 - kex_c_padding));
  UpdateDigest(&H, StringPiece(KI_S.data() + 5, kex_s_packet_len - 5 - kex_s_padding));
  UpdateDigest(&H, k_s); 
  if (KEX::DiffieHellmanGroupExchange(kex_method)) {
    UpdateDigest(&H, dh->gex_min);
    UpdateDigest(&H, dh->gex_pref);
    UpdateDigest(&H, dh->gex_max);
    UpdateDigest(&H, dh->p);
    UpdateDigest(&H, dh->g);
  }
  if (KEX::EllipticCurveDiffieHellman(kex_method)) {
    UpdateDigest(&H, ecdh->c_text);
    UpdateDigest(&H, ecdh->s_text);
  } else {
    UpdateDigest(&H, dh->e);
    UpdateDigest(&H, dh->f);
  }
  UpdateDigest(&H, K);
  ret = Crypto::DigestFinish(&H);
  return ret;
}

string SSH::DeriveKey(Crypto::DigestAlgo algo, const string &session_id, const string &H_text, BigNum K, char ID, int bytes) {
  string ret;
  while (ret.size() < bytes) {
    Crypto::Digest key;
    Crypto::DigestOpen(&key, algo);
    UpdateDigest(&key, K);
    Crypto::DigestUpdate(&key, H_text);
    if (!ret.size()) {
      Crypto::DigestUpdate(&key, StringPiece(&ID, 1));
      Crypto::DigestUpdate(&key, session_id);
    } else Crypto::DigestUpdate(&key, ret);
    ret.append(Crypto::DigestFinish(&key));
  }
  ret.resize(bytes);
  return ret;
}

string SSH::MAC(Crypto::MACAlgo algo, int MAC_len, const StringPiece &m, int seq, const string &k, int prefix) {
  char buf[4];
  Serializable::MutableStream(buf, 4).Htonl(seq);
  string ret(MAC_len, 0);
  Crypto::MAC mac;
  Crypto::MACOpen(&mac, algo, k);
  Crypto::MACUpdate(&mac, StringPiece(buf, 4));
  Crypto::MACUpdate(&mac, m);
  int ret_len = Crypto::MACFinish(&mac, &ret[0], ret.size());
  CHECK_EQ(ret.size(), ret_len);
  return prefix ? ret.substr(0, prefix) : ret;
}

bool SSH::Key   ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::KEX   ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::MAC   ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::Cipher::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }

string SSH::Key   ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::KEX   ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::Cipher::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::MAC   ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) StrAppendCSV(&v, Name(i)); }); return v; }

int SSH::Key   ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::KEX   ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::Cipher::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::MAC   ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }

const char *SSH::Key::Name(int id) {
  switch(id) {
    case RSA:                 return "ssh-rsa";
    case DSS:                 return "ssh-dss";
    case ECDSA_SHA2_NISTP256: return "ecdsa-sha2-nistp256";
    default:                  return "";
  }
};

const char *SSH::KEX::Name(int id) {
  switch(id) {
    case ECDH_SHA2_NISTP256: return "ecdh-sha2-nistp256";
    case ECDH_SHA2_NISTP384: return "ecdh-sha2-nistp384";
    case ECDH_SHA2_NISTP521: return "ecdh-sha2-nistp521";
    case DHGEX_SHA256:       return "diffie-hellman-group-exchange-sha256";
    case DHGEX_SHA1:         return "diffie-hellman-group-exchange-sha1";
    case DH14_SHA1:          return "diffie-hellman-group14-sha1";
    case DH1_SHA1:           return "diffie-hellman-group1-sha1";
    default:                 return "";
  }
};

const char *SSH::Cipher::Name(int id) {
  switch(id) {
    case AES128_CTR:   return "aes128-ctr";
    case AES128_CBC:   return "aes128-cbc";
    case TripDES_CBC:  return "3des-cbc";
    case Blowfish_CBC: return "blowfish-cbc";
    case RC4:          return "arcfour";
    default:           return "";
  }
};

const char *SSH::MAC::Name(int id) {
  switch(id) {
    case MD5:       return "hmac-md5";
    case MD5_96:    return "hmac-md5-96";
    case SHA1:      return "hmac-sha1";
    case SHA1_96:   return "hmac-sha1-96";
    case SHA256:    return "hmac-sha2-256";
    case SHA256_96: return "hmac-sha2-256-96";
    case SHA512:    return "hmac-sha2-512";
    case SHA512_96: return "hmac-sha2-512-96";
    default:        return "";
  }
};

Crypto::CipherAlgo SSH::Cipher::Algo(int id, int *blocksize) {
  switch(id) {
    case AES128_CTR:   if (blocksize) *blocksize = 16;    return Crypto::CipherAlgos::AES128_CTR();
    case AES128_CBC:   if (blocksize) *blocksize = 16;    return Crypto::CipherAlgos::AES128_CBC();
    case TripDES_CBC:  if (blocksize) *blocksize = 8;     return Crypto::CipherAlgos::TripDES_CBC();
    case Blowfish_CBC: if (blocksize) *blocksize = 8;     return Crypto::CipherAlgos::Blowfish_CBC();
    case RC4:          if (blocksize) *blocksize = 16;    return Crypto::CipherAlgos::RC4();
    default:           return 0;
  }
};

Crypto::MACAlgo SSH::MAC::Algo(int id, int *prefix_bytes) {
  switch(id) {
    case MD5:       if (prefix_bytes) *prefix_bytes = 0;    return Crypto::MACAlgos::MD5();
    case MD5_96:    if (prefix_bytes) *prefix_bytes = 12;   return Crypto::MACAlgos::MD5();
    case SHA1:      if (prefix_bytes) *prefix_bytes = 0;    return Crypto::MACAlgos::SHA1();
    case SHA1_96:   if (prefix_bytes) *prefix_bytes = 12;   return Crypto::MACAlgos::SHA1();
    case SHA256:    if (prefix_bytes) *prefix_bytes = 0;    return Crypto::MACAlgos::SHA256();
    case SHA256_96: if (prefix_bytes) *prefix_bytes = 12;   return Crypto::MACAlgos::SHA256();
    case SHA512:    if (prefix_bytes) *prefix_bytes = 0;    return Crypto::MACAlgos::SHA512();
    case SHA512_96: if (prefix_bytes) *prefix_bytes = 12;   return Crypto::MACAlgos::SHA512();
    default:      return 0;
  }
};

string SSH::Serializable::ToString(std::mt19937 &g, int block_size, unsigned *sequence_number) const {
  if (sequence_number) (*sequence_number)++;
  string ret;
  ToString(&ret, g, block_size);
  return ret;
}

void SSH::Serializable::ToString(string *out, std::mt19937 &g, int block_size) const {
  out->resize(NextMultipleOfN(4 + SSH::BinaryPacketHeaderSize + Size(), max(8, block_size)));
  return ToString(&(*out)[0], out->size(), g);
}

void SSH::Serializable::ToString(char *buf, int len, std::mt19937 &g) const {
  unsigned char type = Id, padding = len - SSH::BinaryPacketHeaderSize - Size();
  MutableStream os(buf, len);
  os.Htonl(len - 4);
  os.Write8(padding);
  os.Write8(type);
  Out(&os);
  memcpy(buf + len - padding, RandBytes(padding, g).data(), padding);
}

int SSH::MSG_KEXINIT::Size() const {
  return HeaderSize() + kex_algorithms.size() + server_host_key_algorithms.size() +
    encryption_algorithms_client_to_server.size() + encryption_algorithms_server_to_client.size() +
    mac_algorithms_client_to_server.size() + mac_algorithms_server_to_client.size() + 
    compression_algorithms_client_to_server.size() + compression_algorithms_server_to_client.size() +
    languages_client_to_server.size() + languages_server_to_client.size();
}

string SSH::MSG_KEXINIT::DebugString() const {
  string ret;
  StrAppend(&ret, "kex_algorithms: ",                          kex_algorithms.str(),                          "\n");
  StrAppend(&ret, "server_host_key_algorithms: ",              server_host_key_algorithms.str(),              "\n");
  StrAppend(&ret, "encryption_algorithms_client_to_server: ",  encryption_algorithms_client_to_server.str(),  "\n");
  StrAppend(&ret, "encryption_algorithms_server_to_client: ",  encryption_algorithms_server_to_client.str(),  "\n");
  StrAppend(&ret, "mac_algorithms_client_to_server: ",         mac_algorithms_client_to_server.str(),         "\n");
  StrAppend(&ret, "mac_algorithms_server_to_client: ",         mac_algorithms_server_to_client.str(),         "\n");
  StrAppend(&ret, "compression_algorithms_client_to_server: ", compression_algorithms_client_to_server.str(), "\n");
  StrAppend(&ret, "compression_algorithms_server_to_client: ", compression_algorithms_server_to_client.str(), "\n");
  StrAppend(&ret, "languages_client_to_server: ",              languages_client_to_server.str(),              "\n");
  StrAppend(&ret, "languages_server_to_client: ",              languages_server_to_client.str(),              "\n");
  StrAppend(&ret, "first_kex_packet_follows: ",                int(first_kex_packet_follows),                 "\n");
  return ret;
}

void SSH::MSG_KEXINIT::Out(Serializable::Stream *o) const {
  o->String(cookie);
  o->BString(kex_algorithms);                             o->BString(server_host_key_algorithms);
  o->BString(encryption_algorithms_client_to_server);     o->BString(encryption_algorithms_server_to_client);
  o->BString(mac_algorithms_client_to_server);            o->BString(mac_algorithms_server_to_client);
  o->BString(compression_algorithms_client_to_server);    o->BString(compression_algorithms_server_to_client);
  o->BString(languages_client_to_server);                 o->BString(languages_server_to_client);
  o->Write8(first_kex_packet_follows);
  o->Write32(0);
}

int SSH::MSG_KEXINIT::In(const Serializable::Stream *i) {
  cookie = StringPiece(i->Get(16), 16);
  i->ReadString(&kex_algorithms);                             i->ReadString(&server_host_key_algorithms);
  i->ReadString(&encryption_algorithms_client_to_server);     i->ReadString(&encryption_algorithms_server_to_client);
  i->ReadString(&mac_algorithms_client_to_server);            i->ReadString(&mac_algorithms_server_to_client);
  i->ReadString(&compression_algorithms_client_to_server);    i->ReadString(&compression_algorithms_server_to_client);
  i->ReadString(&languages_client_to_server);                 i->ReadString(&languages_server_to_client);
  i->Read8(&first_kex_packet_follows);
  return i->Result();
}

int SSH::MSG_USERAUTH_REQUEST::Size() const {
  string mn = method_name.str();
  int ret = HeaderSize() + user_name.size() + service_name.size() + method_name.size();
  if      (mn == "publickey")            ret += 4*3 + 1 + algo_name.size() + secret.size() + sig.size();
  else if (mn == "password")             ret += 4*1 + 1 + secret.size();
  else if (mn == "keyboard-interactive") ret += 4*2;
  return ret;
};

void SSH::MSG_USERAUTH_REQUEST::Out(Serializable::Stream *o) const {
  o->BString(user_name);
  o->BString(service_name);
  o->BString(method_name);
  string mn = method_name.str();
  if      (mn == "publickey")            { o->Write8(static_cast<unsigned char>(1)); o->BString(algo_name); o->BString(secret); o->BString(sig); }
  else if (mn == "password")             { o->Write8(static_cast<unsigned char>(0)); o->BString(secret); }
  else if (mn == "keyboard-interactive") { o->BString(""); o->BString(""); }
}

int SSH::MSG_USERAUTH_INFO_REQUEST::Size() const {
  int ret = HeaderSize() + name.size() + instruction.size() + language.size();
  for (auto &p : prompt) ret += 4 + 1 + p.text.size();
  return ret;
}

int SSH::MSG_USERAUTH_INFO_REQUEST::In(const Serializable::Stream *i) {
  int num_prompts = 0;
  i->ReadString(&name);
  i->ReadString(&instruction);
  i->ReadString(&language);
  i->Htonl(&num_prompts);
  prompt.resize(num_prompts);
  for (auto &p : prompt) {
    i->ReadString(&p.text);
    i->Read8(&p.echo);
  }
  return i->Result();
}

int SSH::MSG_USERAUTH_INFO_RESPONSE::Size() const {
  int ret = HeaderSize();
  for (auto &r : response) ret += 4 + r.size();
  return ret;
}

void SSH::MSG_USERAUTH_INFO_RESPONSE::Out(Serializable::Stream *o) const {
  o->Htonl(static_cast<unsigned>(response.size()));
  for (auto &r : response) o->BString(r);
}

int SSH::MSG_CHANNEL_REQUEST::Size() const {
  int ret = HeaderSize() + request_type.size();
  string rt = request_type.str();
  if      (rt == "pty-req")       ret += 4*6 + term.size() + term_mode.size();
  else if (rt == "exec")          ret += 4*1 + term.size();
  else if (rt == "window-change") ret += 4*4;
  return ret;
}

void SSH::MSG_CHANNEL_REQUEST::Out(Serializable::Stream *o) const {
  o->Htonl(recipient_channel);
  o->BString(request_type);
  o->Write8(want_reply);
  string rt = request_type.str();
  if (rt == "pty-req") {
    o->BString(term);
    o->Htonl(width);
    o->Htonl(height);
    o->Htonl(pixel_width);
    o->Htonl(pixel_height);
    o->BString(term_mode);
  } else if (rt == "exec") {
    o->BString(term);
  } else if (rt == "window-change") {
    o->Htonl(width);
    o->Htonl(height);
    o->Htonl(pixel_width);
    o->Htonl(pixel_height);
  }
}

int SSH::DSSKey::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  if (format_id.str() != "ssh-dss") { i->error = true; return -1; }
  p = ReadBigNum(p, i); 
  q = ReadBigNum(q, i); 
  g = ReadBigNum(g, i); 
  y = ReadBigNum(y, i); 
  return i->Result();
}

int SSH::DSSSignature::In(const Serializable::Stream *i) {
  StringPiece blob;
  i->ReadString(&format_id);
  i->ReadString(&blob);
  if (format_id.str() != "ssh-dss" || blob.size() != 40) { i->error = true; return -1; }
  r = BigNumSetData(r, StringPiece(blob.data(),      20));
  s = BigNumSetData(s, StringPiece(blob.data() + 20, 20));
  return i->Result();
}

int SSH::RSAKey::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  if (format_id.str() != Key::Name(Key::RSA)) { i->error = true; return -1; }
  e = ReadBigNum(e, i); 
  n = ReadBigNum(n, i); 
  return i->Result();
}

int SSH::RSASignature::In(const Serializable::Stream *i) {
  StringPiece blob;
  i->ReadString(&format_id);
  i->ReadString(&sig);
  if (format_id.str() != "ssh-rsa") { i->error = true; return -1; }
  return i->Result();
}

int SSH::ECDSAKey::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  if (!PrefixMatch(format_id.str(), "ecdsa-sha2-")) { i->error = true; return -1; }
  i->ReadString(&curve_id);
  i->ReadString(&q);
  return i->Result();
}

int SSH::ECDSASignature::In(const Serializable::Stream *i) {
  StringPiece blob;
  i->ReadString(&format_id);
  i->ReadString(&blob);
  if (!PrefixMatch(format_id.str(), "ecdsa-sha2-")) { i->error = true; return -1; }
  Serializable::ConstStream bs(blob.data(), blob.size());
  r = ReadBigNum(r, &bs); 
  s = ReadBigNum(s, &bs); 
  if (bs.error) i->error = true;
  return i->Result();
}
}; // namespace LFL
