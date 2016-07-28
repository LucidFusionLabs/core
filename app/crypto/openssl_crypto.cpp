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
#include "openssl/evp.h"
#include "openssl/err.h"
#include "openssl/md5.h"
#include "openssl/hmac.h"

namespace LFL {
#ifdef LFL_APPLE
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { return 0; }
#else
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CTR()   { return CipherAlgo(EVP_aes_128_ctr()); }
#endif
Crypto::CipherAlgo Crypto::CipherAlgos::AES128_CBC()   { return CipherAlgo(EVP_aes_128_cbc()); }
Crypto::CipherAlgo Crypto::CipherAlgos::AES256_CBC()   { return CipherAlgo(EVP_aes_256_cbc()); }
Crypto::CipherAlgo Crypto::CipherAlgos::TripDES_CBC()  { return CipherAlgo(EVP_des_ede3_cbc()); }
Crypto::CipherAlgo Crypto::CipherAlgos::Blowfish_CBC() { return CipherAlgo(EVP_bf_cbc()); }
Crypto::CipherAlgo Crypto::CipherAlgos::RC4()          { return CipherAlgo(EVP_rc4()); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA1()         { return DigestAlgo(EVP_get_digestbyname("sha1")); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA256()       { return DigestAlgo(EVP_get_digestbyname("sha256")); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA384()       { return DigestAlgo(EVP_get_digestbyname("sha384")); }
Crypto::DigestAlgo Crypto::DigestAlgos::SHA512()       { return DigestAlgo(EVP_get_digestbyname("sha512")); }
Crypto::DigestAlgo Crypto::DigestAlgos::MD5()          { return DigestAlgo(EVP_md5()); }
Crypto::MACAlgo    Crypto::MACAlgos   ::MD5()          { return MACAlgo(EVP_md5()); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA1()         { return MACAlgo(EVP_sha1()); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA256()       { return MACAlgo(EVP_sha256()); }
Crypto::MACAlgo    Crypto::MACAlgos   ::SHA512()       { return MACAlgo(EVP_sha512()); }
int         Crypto::CipherAlgos::KeySize (CipherAlgo v) { return EVP_CIPHER_key_length(FromVoid<const EVP_CIPHER*>(v)); }
int         Crypto::DigestAlgos::HashSize(DigestAlgo v) { return EVP_MD_size(FromVoid<const EVP_MD*>(v)); }
int         Crypto::MACAlgos   ::HashSize(MACAlgo    v) { return EVP_MD_size(FromVoid<const EVP_MD*>(v)); }
const char *Crypto::DigestAlgos::Name(DigestAlgo v) { return EVP_MD_name(FromVoid<const EVP_MD*>(v)); }
const char *Crypto::CipherAlgos::Name(CipherAlgo v) { return EVP_CIPHER_name(FromVoid<const EVP_CIPHER*>(v)); }
const char *Crypto::MACAlgos   ::Name(MACAlgo    v) { return EVP_MD_name(FromVoid<const EVP_MD*>(v)); }

Crypto::Cipher Crypto::CipherInit() { return EVP_CIPHER_CTX_new(); }
void Crypto::CipherFree(Cipher c) { EVP_CIPHER_CTX_free(FromVoid<EVP_CIPHER_CTX*>(c)); }
int Crypto::CipherGetBlockSize(Cipher c) { return EVP_CIPHER_CTX_block_size(FromVoid<EVP_CIPHER_CTX*>(c)); }
int Crypto::CipherOpen(Cipher c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) { 
  return EVP_CipherInit(FromVoid<EVP_CIPHER_CTX*>(c), FromVoid<const EVP_CIPHER*>(algo),
                        MakeUnsigned(key.data()), MakeUnsigned(IV.data()), dir);
}
int Crypto::CipherUpdate(Cipher c, const StringPiece &in, char *out, int outlen) {
  return EVP_Cipher(FromVoid<EVP_CIPHER_CTX*>(c), MakeUnsigned(out), MakeUnsigned(in.data()), in.size());
}

Crypto::Digest Crypto::DigestOpen(DigestAlgo algo) { CHECK(algo); EVP_MD_CTX *d=new EVP_MD_CTX(); EVP_DigestInit(d, FromVoid<const EVP_MD*>(algo)); return d; }
void Crypto::DigestUpdate(Digest d, const StringPiece &in) { EVP_DigestUpdate(FromVoid<EVP_MD_CTX*>(d), in.data(), in.size()); }
int Crypto::DigestGetHashSize(Digest d) { return EVP_MD_CTX_size(FromVoid<EVP_MD_CTX*>(d)); }
string Crypto::DigestFinish(Digest d) {
  unsigned len = 0;
  string ret(EVP_MAX_MD_SIZE, 0);
  EVP_DigestFinal(FromVoid<EVP_MD_CTX*>(d), MakeUnsigned(&ret[0]), &len);
  delete FromVoid<EVP_MD_CTX*>(d);
  ret.resize(len);
  return ret;
}

Crypto::MAC Crypto::MACOpen(MACAlgo algo, const StringPiece &k) { HMAC_CTX *m=new HMAC_CTX(); HMAC_Init(m, k.data(), k.size(), FromVoid<const EVP_MD*>(algo)); return m; }
void Crypto::MACUpdate(MAC m, const StringPiece &in) { HMAC_Update(FromVoid<HMAC_CTX*>(m), MakeUnsigned(in.data()), in.size()); }
int Crypto::MACFinish(MAC m, char *out, int outlen) { unsigned len=outlen; HMAC_Final(FromVoid<HMAC_CTX*>(m), MakeUnsigned(out), &len); delete FromVoid<HMAC_CTX*>(m); return len; }

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) {
  unsigned char iv[8] = {0,0,0,0,0,0,0,0};
  EVP_CIPHER_CTX ctx; 
  EVP_CIPHER_CTX_init(&ctx); 
  EVP_CipherInit_ex(&ctx, EVP_bf_cbc(), NULL, NULL, NULL, encrypt_or_decrypt);
  EVP_CIPHER_CTX_set_key_length(&ctx, passphrase.size());
  EVP_CipherInit_ex(&ctx, NULL, NULL, MakeUnsigned(passphrase.c_str()), iv, encrypt_or_decrypt); 

  int outlen = 0, tmplen = 0;
  string out(in.size()+encrypt_or_decrypt*EVP_MAX_BLOCK_LENGTH, 0);
  EVP_CipherUpdate(&ctx, MakeUnsigned(&out[0]), &outlen, MakeUnsigned(in.c_str()), in.size());
  EVP_CipherFinal_ex(&ctx, MakeUnsigned(&out[0]) + outlen, &tmplen); 
  if (in.size() % 8) outlen += tmplen;

  EVP_CIPHER_CTX_cleanup(&ctx); 
  if (encrypt_or_decrypt) {
    CHECK_LE(outlen, out.size());
    out.resize(outlen);
  }
  return out;
}

}; // namespace LFL
