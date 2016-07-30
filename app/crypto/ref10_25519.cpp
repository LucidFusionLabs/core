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

namespace LFL {
void Crypto::X25519DiffieHellman::GeneratePair(std::mt19937 &rand_eng) {
  mykey.privkey = RandBytes(crypto_scalarmult_curve25519_BYTES, rand_eng);
  mykey.pubkey.resize(crypto_scalarmult_curve25519_BYTES);
  unsigned char g[crypto_scalarmult_curve25519_BYTES] = {9};
  crypto_scalarmult_curve25519(MakeUnsigned(&mykey.pubkey[0]), MakeUnsigned(mykey.privkey.data()), g);
}

bool Crypto::X25519DiffieHellman::ComputeSecret(BigNum *K) {
  string sharedkey(crypto_scalarmult_curve25519_BYTES, 0);
  if (remotepubkey.size() != crypto_scalarmult_curve25519_BYTES) return false;
  if (sharedkey == remotepubkey) return false;
  crypto_scalarmult_curve25519(MakeUnsigned(&sharedkey[0]), MakeUnsigned(mykey.privkey.data()),
                               MakeUnsigned(remotepubkey.data()));
  *K = BigNumSetData(*K, sharedkey);
  sharedkey.assign(sharedkey.size(), 0);
  return true;
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

    Crypto::Cipher cipher = Crypto::CipherInit();
    int cipher_keysize = Crypto::CipherAlgos::KeySize(cipher_algo);
    int cipher_blocksize = CipherGetBlockSize(cipher);
    string pw = passphrase_cb("");
    string key = Crypto::BCryptPBKDF(pw, kdfoptions.salt, cipher_keysize + cipher_blocksize, kdfoptions.rounds);
    Crypto::CipherOpen(cipher, cipher_algo, 0, StringPiece(key.data(), cipher_keysize),
                       StringPiece(key.data() + cipher_keysize, cipher_blocksize));
    decrypted.resize(opensshkey.privatekey.size());
    Crypto::CipherUpdate(cipher, opensshkey.privatekey, &decrypted[0], decrypted.size());
    Crypto::CipherFree(cipher);
    privatekey = decrypted;

  } else privatekey = opensshkey.privatekey;

  Serializable::ConstStream privatekeys(privatekey.data(), privatekey.size());
  OpenSSHPrivateKeyHeader opensshprivkey(opensshkey.publickey.size());
  if (opensshprivkey.Read(&privatekeys)) return ERRORv(0, "openssh parse private key");

  StringPiece ed25519_keytype, ed25519_pubkey, ed25519_privkey;
  privatekeys.ReadString(&ed25519_keytype);
  if (ed25519_keytype.str() != "ssh-ed25519") return ERRORv(0, "openssh parse ed25519 private key");

  privatekeys.ReadString(&ed25519_pubkey);
  if (ed25519_pubkey.size() != 32) return ERRORv(0, "openssh ed25519 pubkey length");

  privatekeys.ReadString(&ed25519_privkey);
  if (ed25519_privkey.size() != 64) return ERRORv(0, "openssh ed25519 privkey length");

  ed25519_out->pubkey .assign(ed25519_pubkey .data(), ed25519_pubkey .size());
  ed25519_out->privkey.assign(ed25519_privkey.data(), ed25519_privkey.size());
  return true;
}

}; // namespace LFL
