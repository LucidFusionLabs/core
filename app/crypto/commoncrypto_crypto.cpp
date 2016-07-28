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
#include <CommonCrypto/CommonCrypto.h>
#include <CommonCrypto/CommonHMAC.h>

namespace LFL {
struct CCCipherAlgo { enum { AES128_CTR=1, AES128_CBC=2, AES256_CBC=3, TripDES_CBC=4, Blowfish_CBC=5, RC4=6 }; };
struct CCDigestAlgo { enum { MD5=1, SHA1=2, SHA256=3, SHA384=4, SHA512=5 }; };

struct CCCipher { size_t algo=0; CCAlgorithm ccalgo=0; CCCryptorRef ctx=0; };
struct CCDigest { size_t algo=0; void *v=0; CCDigest(int A=0) : algo(A) {} };
struct CCMAC { CCHmacAlgorithm algo; CCHmacContext ctx; };

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { return Void(CCCipherAlgo::AES128_CTR); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { return Void(CCCipherAlgo::AES128_CBC); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES256_CBC()   { return Void(CCCipherAlgo::AES256_CBC); }
Crypto::CipherAlgo Crypto::CipherAlgos::TripDES_CBC()  { return Void(CCCipherAlgo::TripDES_CBC); }
Crypto::CipherAlgo Crypto::CipherAlgos::Blowfish_CBC() { return Void(CCCipherAlgo::Blowfish_CBC); }
Crypto::CipherAlgo Crypto::CipherAlgos::RC4()          { return Void(CCCipherAlgo::RC4); }
Crypto::DigestAlgo Crypto::DigestAlgos::MD5()          { return Void(CCDigestAlgo::MD5); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA1()         { return Void(CCDigestAlgo::SHA1); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA256()       { return Void(CCDigestAlgo::SHA256); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA384()       { return Void(CCDigestAlgo::SHA384); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA512()       { return Void(CCDigestAlgo::SHA512); }
Crypto::MACAlgo    Crypto::   MACAlgos::MD5()          { return Void(kCCHmacAlgMD5); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA1()         { return Void(kCCHmacAlgSHA1); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA256()       { return Void(kCCHmacAlgSHA256); }
Crypto::MACAlgo    Crypto::   MACAlgos::SHA512()       { return Void(kCCHmacAlgSHA512); }

const char *Crypto::CipherAlgos::Name(CipherAlgo v) {
  switch (size_t(v.v)) {
    case CCCipherAlgo::AES128_CTR:   return "aes128-ctr";
    case CCCipherAlgo::AES128_CBC:   return "aes128-cbc";
    case CCCipherAlgo::AES256_CBC:   return "aes256-cbc";
    case CCCipherAlgo::TripDES_CBC:  return "3des-cbc";
    case CCCipherAlgo::Blowfish_CBC: return "blowfish-cbc";
    case CCCipherAlgo::RC4:          return "rc4";
    default:                         return "none";
  }
}

int Crypto::CipherAlgos::KeySize(CipherAlgo v) {
  switch (size_t(v.v)) {
    case CCCipherAlgo::AES128_CTR:   return kCCKeySizeAES128;
    case CCCipherAlgo::AES128_CBC:   return kCCKeySizeAES128;
    case CCCipherAlgo::AES256_CBC:   return kCCKeySizeAES256;
    case CCCipherAlgo::TripDES_CBC:  return kCCKeySize3DES;
    case CCCipherAlgo::Blowfish_CBC: return 16;
    case CCCipherAlgo::RC4:          return 16;
    default:                         return 0;
  }
}

const char *Crypto::DigestAlgos::Name(DigestAlgo v) {
  switch (size_t(v.v)) {
    case CCDigestAlgo::MD5:    return "md5";
    case CCDigestAlgo::SHA1:   return "sha1";
    case CCDigestAlgo::SHA256: return "sha256";
    case CCDigestAlgo::SHA384: return "sha384";
    case CCDigestAlgo::SHA512: return "sha512";
    default:                   return "none";
  }
}

int Crypto::DigestAlgos::HashSize(DigestAlgo v) {
  switch (size_t(v.v)) {
    case CCDigestAlgo::MD5:    return CC_MD5_DIGEST_LENGTH;
    case CCDigestAlgo::SHA1:   return CC_SHA1_DIGEST_LENGTH;
    case CCDigestAlgo::SHA256: return CC_SHA256_DIGEST_LENGTH;
    case CCDigestAlgo::SHA384: return CC_SHA384_DIGEST_LENGTH;
    case CCDigestAlgo::SHA512: return CC_SHA512_DIGEST_LENGTH;
    default:                   return 0;
  }
}

const char *Crypto::MACAlgos::Name(MACAlgo v) {
  switch (size_t(v.v)) {
    case kCCHmacAlgMD5:    return "md5";
    case kCCHmacAlgSHA1:   return "sha1";
    case kCCHmacAlgSHA256: return "sha256";
    case kCCHmacAlgSHA512: return "sha512";
    default:               return "none";
  }
}

int Crypto::MACAlgos::HashSize(MACAlgo v) {
  switch (size_t(v.v)) {
    case kCCHmacAlgMD5:    return CC_MD5_DIGEST_LENGTH;
    case kCCHmacAlgSHA1:   return CC_SHA1_DIGEST_LENGTH;
    case kCCHmacAlgSHA256: return CC_SHA256_DIGEST_LENGTH;
    case kCCHmacAlgSHA512: return CC_SHA512_DIGEST_LENGTH;
    default:               return 0;
  }
}

Crypto::Cipher Crypto::CipherInit() { return new CCCipher(); }
void Crypto::CipherFree(Cipher x) { auto c=FromVoid<CCCipher*>(x); CCCryptorRelease(c->ctx); delete c; }
int Crypto::CipherGetBlockSize(Cipher c) {
  switch(FromVoid<CCCipher*>(c)->ccalgo) {
    case kCCAlgorithmAES128:   return kCCBlockSizeAES128;
    case kCCAlgorithm3DES:     return kCCBlockSize3DES;
    case kCCAlgorithmBlowfish: return kCCBlockSizeBlowfish;
    case kCCAlgorithmRC4:      return 16;
    default:                   return -1;
  }
}

int Crypto::CipherOpen(Cipher x, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) {
  bool ctr = false;
  auto c = FromVoid<CCCipher*>(x);
  switch((c->algo = size_t(algo.v))) {
    case CCCipherAlgo::AES128_CTR:   c->ccalgo = kCCAlgorithmAES; ctr = true; break;
    case CCCipherAlgo::AES128_CBC:   c->ccalgo = kCCAlgorithmAES;             break;
    case CCCipherAlgo::AES256_CBC:   c->ccalgo = kCCAlgorithmAES;             break;
    case CCCipherAlgo::TripDES_CBC:  c->ccalgo = kCCAlgorithm3DES;            break;
    case CCCipherAlgo::Blowfish_CBC: c->ccalgo = kCCAlgorithmBlowfish;        break;
    case CCCipherAlgo::RC4:          c->ccalgo = kCCAlgorithmRC4;             break;
    default:                         return -1;
  }
  int mode = (size_t(algo.v) == CCCipherAlgo::RC4) ? kCCModeRC4 : (ctr ? kCCModeCTR : kCCModeCBC);
  return CCCryptorCreateWithMode(dir ? kCCEncrypt : kCCDecrypt, mode, c->ccalgo, 0, IV.data(), key.data(), key.size(),
                                 0, 0, 0, ctr ? kCCModeOptionCTR_BE : 0, &c->ctx) == kCCSuccess;
}

int Crypto::CipherUpdate(Cipher c, const StringPiece &in, char *out, int outlen) {
  size_t wrote = 0;
  return CCCryptorUpdate(FromVoid<CCCipher*>(c)->ctx, in.data(), in.size(), out, outlen, &wrote) == kCCSuccess;
}

Crypto::Digest Crypto::DigestOpen(DigestAlgo algo) {
  auto d = new CCDigest(size_t(algo.v));
  switch(d->algo) {
    case CCDigestAlgo::MD5:    d->v=calloc(sizeof(CC_MD5_CTX),   1); CC_MD5_Init   (static_cast<CC_MD5_CTX*>   (d->v)); break;
    case CCDigestAlgo::SHA1:   d->v=calloc(sizeof(CC_SHA1_CTX),  1); CC_SHA1_Init  (static_cast<CC_SHA1_CTX*>  (d->v)); break;
    case CCDigestAlgo::SHA256: d->v=calloc(sizeof(CC_SHA256_CTX),1); CC_SHA256_Init(static_cast<CC_SHA256_CTX*>(d->v)); break;
    case CCDigestAlgo::SHA384: d->v=calloc(sizeof(CC_SHA512_CTX),1); CC_SHA384_Init(static_cast<CC_SHA512_CTX*>(d->v)); break;
    case CCDigestAlgo::SHA512: d->v=calloc(sizeof(CC_SHA512_CTX),1); CC_SHA512_Init(static_cast<CC_SHA512_CTX*>(d->v)); break;
    default:                   d->v=0; break;
  }
  return d;
}

int Crypto::DigestGetHashSize(Digest d) { return DigestAlgos::HashSize(Void(FromVoid<CCDigest*>(d)->algo)); }
void Crypto::DigestUpdate(Digest x, const StringPiece &in) {
  auto d = FromVoid<CCDigest*>(x);
  switch(d->algo) {
    case CCDigestAlgo::MD5:    CC_MD5_Update   (static_cast<CC_MD5_CTX*>   (d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA1:   CC_SHA1_Update  (static_cast<CC_SHA1_CTX*>  (d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA256: CC_SHA256_Update(static_cast<CC_SHA256_CTX*>(d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA384: CC_SHA384_Update(static_cast<CC_SHA512_CTX*>(d->v), in.data(), in.size()); break;
    case CCDigestAlgo::SHA512: CC_SHA512_Update(static_cast<CC_SHA512_CTX*>(d->v), in.data(), in.size()); break;
    default: break;
  }
}

string Crypto::DigestFinish(Digest x) {
  auto d = FromVoid<CCDigest*>(x);
  string ret;
  switch(d->algo) {
    case CCDigestAlgo::MD5:    ret.resize(CC_MD5_DIGEST_LENGTH);    CC_MD5_Final   (MakeUnsigned(&ret[0]), static_cast<CC_MD5_CTX*>   (d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA1:   ret.resize(CC_SHA1_DIGEST_LENGTH);   CC_SHA1_Final  (MakeUnsigned(&ret[0]), static_cast<CC_SHA1_CTX*>  (d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA256: ret.resize(CC_SHA256_DIGEST_LENGTH); CC_SHA256_Final(MakeUnsigned(&ret[0]), static_cast<CC_SHA256_CTX*>(d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA384: ret.resize(CC_SHA384_DIGEST_LENGTH); CC_SHA384_Final(MakeUnsigned(&ret[0]), static_cast<CC_SHA512_CTX*>(d->v)); free(d->v); d->v=0; break;
    case CCDigestAlgo::SHA512: ret.resize(CC_SHA512_DIGEST_LENGTH); CC_SHA512_Final(MakeUnsigned(&ret[0]), static_cast<CC_SHA512_CTX*>(d->v)); free(d->v); d->v=0; break;
    default: break;
  }
  delete d;
  return ret;
}

Crypto::MAC Crypto::MACOpen(MACAlgo algo, const StringPiece &k) {
  auto m = new CCMAC();
  CCHmacInit(&m->ctx, (m->algo = size_t(algo.v)), k.data(), k.size());
  return m;
}

void Crypto::MACUpdate(MAC m, const StringPiece &in) {
  CCHmacUpdate(&FromVoid<CCMAC*>(m)->ctx, in.data(), in.size());
}

int Crypto::MACFinish(MAC x, char *out, int outlen) {
  auto m = FromVoid<CCMAC*>(x);
  CCHmacFinal(&m->ctx, out); 
  switch(m->algo) {
    case kCCHmacAlgMD5:    return CC_MD5_DIGEST_LENGTH;
    case kCCHmacAlgSHA1:   return CC_SHA1_DIGEST_LENGTH;
    case kCCHmacAlgSHA256: return CC_SHA256_DIGEST_LENGTH;
    case kCCHmacAlgSHA512: return CC_SHA512_DIGEST_LENGTH;
    default:               return -1;
  }
}

}; // namespace LFL
