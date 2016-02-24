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

}; // namespace LFL
