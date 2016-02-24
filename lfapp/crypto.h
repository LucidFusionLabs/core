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

#ifndef LFL_LFAPP_CRYPTO_H__
#define LFL_LFAPP_CRYPTO_H__

#if defined(LFL_COMMONCRYPTO)
#include <CommonCrypto/CommonCrypto.h>
#include <CommonCrypto/CommonHMAC.h>
#elif defined(LFL_OPENSSL)
#include "openssl/evp.h"
#include "openssl/hmac.h"
#endif

namespace LFL {
struct Crypto {
#if defined(LFL_COMMONCRYPTO)
  struct Cipher { int algo=0; CCAlgorithm ccalgo; CCCryptorRef ctx; };
  struct Digest { int algo=0; void *v=0; };
  struct MAC { CCHmacAlgorithm algo; CCHmacContext ctx; };
  typedef int CipherAlgo;
  typedef int DigestAlgo;
  typedef CCHmacAlgorithm MACAlgo;
#elif defined(LFL_OPENSSL)
  typedef EVP_CIPHER_CTX Cipher;
  typedef EVP_MD_CTX Digest;
  typedef HMAC_CTX MAC;
  typedef const EVP_CIPHER* CipherAlgo;
  typedef const EVP_MD* DigestAlgo;
  typedef const EVP_MD* MACAlgo;
#else
  typedef void* Cipher;
  typedef void* Digest;
  typedef void* MAC;
  typedef void* CipherAlgo;
  typedef void* DigestAlgo;
  typedef void* MACAlgo;
#endif

  struct CipherAlgos {
    static CipherAlgo AES128_CTR();
    static CipherAlgo AES128_CBC();
    static CipherAlgo TripDES_CBC();
    static CipherAlgo Blowfish_CBC();
    static CipherAlgo RC4();
    static const char *Name(CipherAlgo);
    static int KeySize(CipherAlgo);
  };

  struct DigestAlgos {
    static DigestAlgo MD5();
    static DigestAlgo SHA1();
    static DigestAlgo SHA256();
    static DigestAlgo SHA384();
    static DigestAlgo SHA512();
    static const char *Name(DigestAlgo);
    static int HashSize(DigestAlgo);
  };

  struct MACAlgos {
    static MACAlgo MD5();
    static MACAlgo SHA1();
    static MACAlgo SHA256();
    static MACAlgo SHA512();
    static const char *Name(MACAlgo);
    static int HashSize(MACAlgo);
  };

  struct DiffieHellman {
    int gex_min=1024, gex_max=8192, gex_pref=2048;
    BigNum g, p, x, e, f;
    DiffieHellman() : g(NewBigNum()), p(NewBigNum()), x(NewBigNum()), e(NewBigNum()), f(NewBigNum()) {}
    virtual ~DiffieHellman() { FreeBigNum(g); FreeBigNum(p); FreeBigNum(x); FreeBigNum(e); FreeBigNum(f); }
    bool GeneratePair(int secret_bits, BigNumContext ctx);
    bool ComputeSecret(BigNum *K, BigNumContext ctx) { BigNumModExp(*K, f, x, p, ctx); return true; }
    static string GenerateModulus(int generator, int bits);
    static BigNum Group1Modulus (BigNum g, BigNum p, int *rand_num_bits);
    static BigNum Group14Modulus(BigNum g, BigNum p, int *rand_num_bits);
  };

  struct EllipticCurve {
    static ECDef NISTP256();
    static ECDef NISTP384();
    static ECDef NISTP521();
    static ECPair NewPair(ECDef, bool generate);
  };

  struct EllipticCurveDiffieHellman {
    ECPair pair=0;
    ECGroup g=0;
    ECPoint c=0, s=0;
    string c_text, s_text;
    virtual ~EllipticCurveDiffieHellman() { FreeECPair(pair); FreeECPoint(s); }
    bool GeneratePair(ECDef curve, BigNumContext ctx);
    bool ComputeSecret(BigNum *K, BigNumContext ctx);
  };

  static string MD5(const string &in);
  static string SHA1(const string &in);
  static string SHA256(const string &in);
  static string Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt);
  static string ComputeDigest(DigestAlgo algo, const string &in);

  static void CipherInit(Cipher*);
  static void CipherFree(Cipher*);
  static int  CipherGetBlockSize(Cipher*);
  static int  CipherOpen(Cipher*, CipherAlgo, bool dir, const StringPiece &key, const StringPiece &iv);
  static int  CipherUpdate(Cipher*, const StringPiece &in, char *out, int outlen);

  static int  DigestGetHashSize(Digest*);
  static void DigestOpen(Digest*, DigestAlgo);
  static void DigestUpdate(Digest*, const StringPiece &in);
  static string DigestFinish(Digest*);

  static void MACOpen(MAC*, MACAlgo, const StringPiece &key);
  static void MACUpdate(MAC*, const StringPiece &in);
  static int  MACFinish(MAC*, char *out, int outlen);
};

}; // namespace LFL
#endif // LFL_LFAPP_CRYPTO_H__
