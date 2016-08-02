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

#include "core/app/crypto.h"
#include "core/app/crypto/openssh.h"
#include "core/app/net/ssh.h"
#include "curve25519/nacl_includes/crypto_scalarmult_curve25519.h"
#include "ed25519/nacl_includes/crypto_sign.h"
extern "C" {
#include "ed25519/ge.h"
};

namespace LFL {
int Ed25519GeneratePair(Ed25519Pair *key, std::mt19937 &rand_eng) {
  string seed = Crypto::SHA256(RandBytes(32, rand_eng));
  key->privkey = Crypto::SHA512(seed);
  key->privkey[0] &= 248;
  key->privkey[31] &= 63;
  key->privkey[31] |= 64;
  key->pubkey.resize(32);

  ge_p3 A;
  ge_scalarmult_base(&A, MakeUnsigned(key->privkey.data()));
  ge_p3_tobytes(MakeUnsigned(&key->pubkey[0]), &A);

  memcpy(&key->privkey[0],  seed.data(),        32);
  memcpy(&key->privkey[32], key->pubkey.data(), 32);
  return 1;
}

string Ed25519Sign(const StringPiece &msg, const StringPiece &ed25519_key) {
  string ret(64 + msg.size(), 0);
  unsigned long long ret_size = ret.size();
  if (crypto_sign_edwards25519sha512batch(MakeUnsigned(&ret[0]), &ret_size,
                                          MakeUnsigned(msg.data()), msg.size(),
                                          MakeUnsigned(ed25519_key.data())) || ret_size < 64) return "";
  return ret.substr(0, 64);
}

int Ed25519Verify(const StringPiece &msg, const StringPiece &sig, const StringPiece &ed25519_key) {
  if (sig.size() != 64) return false;
  if (ed25519_key.size() != 32) return false;
  string verify = StrCat(sig.str(), msg.str()), result(verify.size(), 0);
  unsigned long long result_size = result.size();
  return 0 == crypto_sign_edwards25519sha512batch_open(MakeUnsigned(&result[0]), &result_size,
                                                       MakeUnsigned(verify.data()), verify.size(),
                                                       MakeUnsigned(ed25519_key.data()));
}

string Ed25519PublicKeyPEM(const Ed25519Pair &key, const string &comment) {
  string proto = OpenSSHEd25519PublicKey(key.pubkey).ToString();
  string encoded = Singleton<Base64>::Get()->Encode(proto.data(), proto.size());
  return StrCat("ssh-ed25519 ", encoded, comment.size() ? " " : "", comment, "\n");
}

string OpenSSHKeyCrypt(Crypto::CipherAlgo cipher_algo, bool direction, const string &pw,
                    int rounds, const StringPiece &salt, const StringPiece &in) {
  Crypto::Cipher cipher = Crypto::CipherInit();
  int cipher_keysize = Crypto::CipherAlgos::KeySize(cipher_algo);
  int cipher_blocksize = Crypto::CipherGetBlockSize(cipher);
  string key = Crypto::BCryptPBKDF(pw, salt, cipher_keysize + cipher_blocksize, rounds);
  Crypto::CipherOpen(cipher, cipher_algo, direction, StringPiece(key.data(), cipher_keysize),
                     StringPiece(key.data() + cipher_keysize, cipher_blocksize));
  string ret(in.size(), 0);
  Crypto::CipherUpdate(cipher, in, &ret[0], ret.size());
  Crypto::CipherFree(cipher);
  return ret;
}

string Ed25519PrivateKeyPEM(const Ed25519Pair &key, const string &pw, Crypto::CipherAlgo enc,
                            const string &comment, int checkint) {
  OpenSSHEd25519PrivateKey ed25519priv(key.pubkey, key.privkey, comment);
  OpenSSHPrivateKeyHeader opensshprivkey;
  opensshprivkey.checkint1 = opensshprivkey.checkint2 = checkint;
  string privkey = StrCat(opensshprivkey.ToString(), ed25519priv.ToString());
  int cipher_block_size = 32, padded_privkey_len = NextMultipleOfN(privkey.size(), cipher_block_size);
  for (int i=0, l=padded_privkey_len-privkey.size(); i != l; ++i) privkey.append(1, i+1);

  std::mt19937 rand_eng;
  OpenSSHKey opensshkey("none", "none", "", privkey);
  opensshkey.publickey.emplace_back(OpenSSHEd25519PublicKey(key.pubkey).ToString());
  string proto = !enc ? opensshkey.ToString() :
    OpenSSHKeyCrypt(Crypto::CipherAlgos::AES256_CBC(), true, pw, 16, 
                    RandBytes(16, rand_eng), opensshkey.ToString());
  string encoded = Singleton<Base64>::Get()->Encode(proto.data(), proto.size());

  string type = "OPENSSH PRIVATE KEY", ret = StrCat("-----BEGIN ", type, "-----\n");
  StringChunkIterT<70, char> iter(encoded);
  for (iter.Next(); !iter.Done(); iter.Next()) {
    ret.append(iter.Current(), iter.CurrentLength());
    ret.append("\n");
  }
  StrAppend(&ret, "-----END ", type, "-----\n");
  return ret;
}

void Crypto::X25519DiffieHellman::GeneratePair(std::mt19937 &rand_eng) {
  myprivkey = RandBytes(crypto_scalarmult_curve25519_BYTES, rand_eng);
  mypubkey.resize(crypto_scalarmult_curve25519_BYTES);
  unsigned char g[crypto_scalarmult_curve25519_BYTES] = {9};
  crypto_scalarmult_curve25519(MakeUnsigned(&mypubkey[0]), MakeUnsigned(myprivkey.data()), g);
}

bool Crypto::X25519DiffieHellman::ComputeSecret(BigNum *K) {
  string sharedkey(crypto_scalarmult_curve25519_BYTES, 0);
  if (remotepubkey.size() != crypto_scalarmult_curve25519_BYTES) return false;
  if (sharedkey == remotepubkey) return false;
  crypto_scalarmult_curve25519(MakeUnsigned(&sharedkey[0]), MakeUnsigned(myprivkey.data()),
                               MakeUnsigned(remotepubkey.data()));
  *K = BigNumSetData(*K, sharedkey);
  sharedkey.assign(sharedkey.size(), 0);
  return true;
}

string Crypto::ParsePEMHeader(const char *key, const char **start, const char **end, const char **headers_end) {
  static string begin_text="-----BEGIN ", end_text="-----END ", term_text="-----";
  const char *begin_beg, *begin_end, *end_beg, *end_end;
  if (!(begin_beg = strstr(key,                           begin_text.c_str()))) return ERRORv("", begin_text, " not found");
  if (!(begin_end = strstr(begin_beg + begin_text.size(), term_text .c_str()))) return ERRORv("", term_text,  " not found");
  if (!(end_beg   = strstr(begin_end + term_text .size(), end_text  .c_str()))) return ERRORv("", end_text,   " not found");
  if (!(end_end   = strstr(end_beg   + end_text  .size(), term_text .c_str()))) return ERRORv("", term_text,  " not found");

  string type(begin_beg + begin_text.size(), begin_end - begin_beg - begin_text.size());
  if (type.size() != end_end - end_beg - end_text.size())            return ERRORv("", "begin/end length disagreement");
  if (strncmp(type.c_str(), end_beg + end_text.size(), type.size())) return ERRORv("", "begin/end text disagreement");

  *end = end_beg;
  *start = IncrementNewline(begin_end + term_text.size());
  StringPiece encapsulated(*start, *end - *start), headers_sep("\n\n", 2);
  *headers_end = FindString(encapsulated, headers_sep);
  if (!*headers_end) *headers_end = FindString(encapsulated, (headers_sep = StringPiece("\r\n\r\n", 4)));
  return type;
}

bool Crypto::ParsePEM(char *key, RSAKey *rsa_out, DSAKey *dsa_out, ECPair *ec_out, Ed25519Pair *ed25519_out,
                      function<string(string)> passphrase_cb) {
  const char *start=0, *end=0, *headers_end=0;
  string type = ParsePEMHeader(key, &start, &end, &headers_end), cipher, cbcinit, b64text;
  StringPiece encapsulated(start, end - start), privatekey;
  if (headers_end || type != "OPENSSH PRIVATE KEY")
    return ParsePEM(key, rsa_out, dsa_out, ec_out, move(passphrase_cb));

  StringLineIter lines(encapsulated);
  for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) b64text.append(line);
  string decoded = Singleton<Base64>::Get()->Decode(b64text.data(), b64text.size()), decrypted;
  Serializable::ConstStream decodeds(decoded.data(), decoded.size());
  OpenSSHKey opensshkey;
  if (opensshkey.Read(&decodeds)) return ERRORv(0, "openssh parse key");
  string ciphername = opensshkey.ciphername.str();
  Crypto::CipherAlgo cipher_algo;

  DestructorCallbacks clear_decrypted([&](){ decrypted.assign(decrypted.size(), 0); });
  if (opensshkey.kdfname.str() == "bcrypt") {
    Serializable::ConstStream kdfoptionss(opensshkey.kdfoptions.data(), opensshkey.kdfoptions.size());
    OpenSSHBCryptKDFOptions kdfoptions;
    if (kdfoptions.Read(&kdfoptionss)) return ERRORv(0, "openssh parse bcrypt kdfoptions");
    if (ciphername == "aes256-cbc") cipher_algo = Crypto::CipherAlgos::AES256_CBC();
    else return ERRORv(0, "openssh parse cipher");
    decrypted = OpenSSHKeyCrypt(cipher_algo, 0, passphrase_cb(""), kdfoptions.rounds, kdfoptions.salt,
                                opensshkey.privatekey);
    privatekey = decrypted;

  } else privatekey = opensshkey.privatekey;

  Serializable::ConstStream privatekeys(privatekey.data(), privatekey.size());
  OpenSSHPrivateKeyHeader opensshprivkey;
  if (opensshprivkey.Read(&privatekeys)) return ERRORv(0, "openssh parse private key");

  OpenSSHEd25519PrivateKey ed25519;
  if (ed25519.Read(&privatekeys)) return ERRORv(0, "openssh parse ed25519 private key");
  ed25519_out->pubkey .assign(ed25519.pubkey .data(), ed25519.pubkey .size());
  ed25519_out->privkey.assign(ed25519.privkey.data(), ed25519.privkey.size());
  return true;
}

}; // namespace LFL
