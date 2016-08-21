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

#ifndef LFL_CORE_APP_CRYPTO_H__
#define LFL_CORE_APP_CRYPTO_H__

namespace LFL {
struct BigNum        : public VoidPtr { using VoidPtr::VoidPtr; };
struct BigNumContext : public VoidPtr { using VoidPtr::VoidPtr; };
BigNum        NewBigNum();
BigNumContext NewBigNumContext();
void FreeBigNumContext(BigNumContext c);
void FreeBigNum(BigNum n);
void BigNumModExp(BigNum v, BigNum a, BigNum e, BigNum m, BigNumContext);
void BigNumSetValue(BigNum v, int val);
void BigNumGetData(BigNum v, char *out);
BigNum BigNumSetData(BigNum v, const StringPiece &data);
BigNum BigNumRand(BigNum v, int bits, int top, int bottom);
int BigNumDataSize(BigNum v);
int BigNumSignificantBits(BigNum v);

struct ECDef   : public VoidPtr { using VoidPtr::VoidPtr; };
struct ECGroup : public VoidPtr { using VoidPtr::VoidPtr; };
struct ECPoint : public VoidPtr { using VoidPtr::VoidPtr; };
struct ECPair  : public VoidPtr { using VoidPtr::VoidPtr; };
ECPoint NewECPoint(ECGroup);
void FreeECPoint(ECPoint);
void FreeECPair(ECPair);
ECDef GetECGroupID(ECGroup);
ECGroup GetECPairGroup(ECPair);
ECPoint GetECPairPubKey(ECPair);
bool SetECPairPubKey(ECPair, ECPoint);
string ECPointGetData(ECGroup, ECPoint, BigNumContext);
int ECPointDataSize(ECGroup, ECPoint, BigNumContext);
void ECPointGetData(ECGroup, ECPoint, char *out, int len, BigNumContext);
void ECPointSetData(ECGroup, ECPoint out, const StringPiece &data);

struct Ed25519Pair { string pubkey, privkey; };

struct RSAKey   : public VoidPtr { using VoidPtr::VoidPtr; };
struct DSAKey   : public VoidPtr { using VoidPtr::VoidPtr; };
struct DSASig   : public VoidPtr { using VoidPtr::VoidPtr; };
struct ECDSASig : public VoidPtr { using VoidPtr::VoidPtr; };
RSAKey NewRSAPubKey();
DSAKey NewDSAPubKey();
DSASig NewDSASig();
ECDSASig NewECDSASig();
BigNum GetRSAKeyE(RSAKey);
BigNum GetRSAKeyN(RSAKey);
BigNum GetDSAKeyP(DSAKey);
BigNum GetDSAKeyQ(DSAKey);
BigNum GetDSAKeyG(DSAKey);
BigNum GetDSAKeyK(DSAKey);
BigNum GetDSASigR(DSASig);
BigNum GetDSASigS(DSASig);
BigNum GetECDSASigR(ECDSASig);
BigNum GetECDSASigS(ECDSASig);
void RSAKeyFree(RSAKey);
void DSAKeyFree(DSAKey);
void DSASigFree(DSASig);
void ECDSASigFree(ECDSASig);
int RSAGeneratePair(RSAKey key, int bits, BigNum);
int RSAGeneratePair(RSAKey key, int bits);
int DSAGeneratePair(DSAKey key, int bits);
int Ed25519GeneratePair(Ed25519Pair *key, std::mt19937&);
int RSAVerify(const StringPiece &digest, string *out, RSAKey rsa_key);
int DSAVerify(const StringPiece &digest, DSASig dsa_sig, DSAKey dsa_key);
int ECDSAVerify(const StringPiece &digest, ECDSASig dsa_sig, ECPair ecdsa_keypair);
int Ed25519Verify(const StringPiece &msg, const StringPiece &sig, const StringPiece &ed25519_key);
int RSASign(const StringPiece &digest, string *out, RSAKey rsa_key);
DSASig DSASign(const StringPiece &digest, DSAKey dsa_key);
ECDSASig ECDSASign(const StringPiece &digest, ECPair ecdsa_keypair);
string Ed25519Sign(const StringPiece &msg, const StringPiece &ed25519_key);

struct Crypto {
  struct Cipher     : public VoidPtr      { using VoidPtr::VoidPtr; };
  struct Digest     : public VoidPtr      { using VoidPtr::VoidPtr; };
  struct MAC        : public VoidPtr      { using VoidPtr::VoidPtr; };
  struct CipherAlgo : public ConstVoidPtr { using ConstVoidPtr::ConstVoidPtr; };
  struct DigestAlgo : public ConstVoidPtr { using ConstVoidPtr::ConstVoidPtr; };
  struct MACAlgo    : public ConstVoidPtr { using ConstVoidPtr::ConstVoidPtr; };

  struct CipherAlgos {
    static CipherAlgo AES128_CTR();
    static CipherAlgo AES128_CBC();
    static CipherAlgo AES256_CBC();
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

  struct X25519DiffieHellman {
    string myprivkey, mypubkey, remotepubkey;
    void GeneratePair(std::mt19937&);
    bool ComputeSecret(BigNum *K);
  };

  static string MD5(const string &in);
  static string SHA1(const string &in);
  static string SHA256(const string &in);
  static string SHA512(const string &in);
  static string Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt);
  static string ComputeDigest(DigestAlgo algo, const string &in);

  static Cipher CipherInit();
  static void CipherFree(Cipher);
  static int  CipherGetBlockSize(Cipher);
  static int  CipherOpen(Cipher, CipherAlgo, bool dir, const StringPiece &key, const StringPiece &iv);
  static int  CipherUpdate(Cipher, const StringPiece &in, char *out, int outlen);

  static Digest DigestOpen(DigestAlgo);
  static void DigestUpdate(Digest, const StringPiece &in);
  static int  DigestGetHashSize(Digest);
  static string DigestFinish(Digest);

  static MAC MACOpen(MACAlgo, const StringPiece &key);
  static void MACUpdate(MAC, const StringPiece &in);
  static int  MACFinish(MAC, char *out, int outlen);

  static bool GenerateKey(const string &algo, int bits, const string &pw, const string &comment,
                          string *pubkeyout, string *privkeyout);
  static string ParsePEMHeader(const char *key,
                               const char **start, const char **end, const char **headers_end);
  static bool ParsePEM(const char *key, RSAKey *rsa_out, DSAKey *dsa_out, ECPair *ec_out,
                       function<string(string)> passphrase_cb = function<string(string)>());
  static bool ParsePEM(const char *key, RSAKey *rsa_out, DSAKey *dsa_out, ECPair *ec_out, Ed25519Pair *pair_out,
                       function<string(string)> passphrase_cb = function<string(string)>());
  static string BCryptPBKDF(const StringPiece &pw, const StringPiece &salt, int size, int rounds);
  static string GetLastErrorText();
};

bool GetECName(ECDef, string *algo_name, string *curve_name, Crypto::DigestAlgo *hash_id);
string RSAPEMPublicKey(RSAKey key);
string DSAPEMPublicKey(DSAKey key);
string ECDSAPEMPublicKey(ECPair key);
string RSAOpenSSHPublicKey(RSAKey key, const string &comment);
string DSAOpenSSHPublicKey(DSAKey key, const string &comment);
string ECDSAOpenSSHPublicKey(ECPair key, const string &comment);
string Ed25519OpenSSHPublicKey(const Ed25519Pair &key, const string &comment);
string RSAPEMPrivateKey(RSAKey key, string pw);
string DSAPEMPrivateKey(DSAKey key, string pw);
string ECDSAPEMPrivateKey(ECPair key, string pw);
string Ed25519PEMPrivateKey(const Ed25519Pair &key, const string &pw, const string &comment, int checkint);

}; // namespace LFL
#endif // LFL_CORE_APP_CRYPTO_H__
