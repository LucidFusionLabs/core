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

namespace LFL {
BigNum        NewBigNum        ()                { FATAL("not implemented"); }
BigNumContext NewBigNumContext ()                { FATAL("not implemented"); }
void          FreeBigNumContext(BigNumContext c) { FATAL("not implemented"); }
void          FreeBigNum       (BigNum        n) { FATAL("not implemented"); }
void BigNumSetValue(BigNum v, int val)        { FATAL("not implemented"); }
void BigNumGetData(const BigNum v, char *out) { FATAL("not implemented"); }
int  BigNumDataSize       (const BigNum v)    { FATAL("not implemented"); }
int  BigNumSignificantBits(const BigNum v)    { FATAL("not implemented"); }
void BigNumModExp(BigNum v, const BigNum a, const BigNum e, const BigNum m, BigNumContext) { FATAL("not implemented"); }
BigNum BigNumSetData(BigNum v, const StringPiece &data)       { FATAL("not implemented"); }
BigNum BigNumRand   (BigNum v, int bits, int top, int bottom) { FATAL("not implemented"); }

#if 0
string Crypto::ParsePEM(const char *key, string *out, const function<string(string)> &passphrase_cb) {
  static string begin_text="-----BEGIN ", end_text="-----END ", term_text="-----";
  const char *begin_beg, *begin_end, *end_beg, *end_end;
  string cipher, cbcinit, b64text;
  out->clear();
  if (!(begin_beg = strstr(key,                           begin_text.c_str()))) return ERRORv("", begin_text, " not found");
  if (!(begin_end = strstr(begin_beg + begin_text.size(), term_text .c_str()))) return ERRORv("", term_text,  " not found");
  if (!(end_beg   = strstr(begin_end + term_text .size(), end_text  .c_str()))) return ERRORv("", end_text,   " not found");
  if (!(end_end   = strstr(end_beg   + end_text  .size(), term_text .c_str()))) return ERRORv("", term_text,  " not found");
  string type(begin_beg + begin_text.size(), begin_end - begin_beg - begin_text.size());
  if (type.size() != end_end - end_beg - end_text.size())            return ERRORv("", "begin/end length disagreement");
  if (strncmp(type.c_str(), end_beg + end_text.size(), type.size())) return ERRORv("", "begin/end text disagreement");

  const char *start = IncrementNewline(begin_end + term_text.size()), *end = end_beg;
  StringPiece encapsulated(start, end - start), headers_sep("\n\n", 2);
  const char *headers_end = FindString(encapsulated, headers_sep);
  if (!headers_end) headers_end = FindString(encapsulated, (headers_sep = StringPiece("\r\n\r\n", 4)));
  if (headers_end) do {
    encapsulated.pop_front(headers_end + headers_sep.len - encapsulated.begin());
    string cbcinittext;
    StringPiece proc_type, dek_info;
    HTTP::GrepHeaders(start, headers_end, 2, "Proc-Type", &proc_type, "DEK-Info", &dek_info);
    if (proc_type.str() != "4,ENCRYPTED") break;
    Split(dek_info.str(), iscomma, &cipher, &cbcinittext);
    if (cbcinittext.size() % 8 != 0) return ERRORv("", "unexpected cbcinit size ", cbcinittext.size());
    cbcinit = string(cbcinittext.size() / 2, 0);
    uint32_t *cbc = reinterpret_cast<uint32_t*>(&cbcinit[0]);
    for (int i = 0, l = cbcinittext.size() / 8; i != l; ++i)
      cbc[i] = htonl(strtoul(string(cbcinittext.data() + i * 8, 8).c_str(), 0, 16));
  } while(0);

  StringLineIter lines(encapsulated);
  for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) b64text.append(line);
  *out = Singleton<Base64>::Get()->Decode(b64text.data(), b64text.size()); 

  if (!cipher.empty()) {
    int ret;
    CipherAlgo algo;
    if (cipher == "DES-EDE3-CBC") algo = Crypto::CipherAlgos::TripDES_CBC();
    if (!algo) return ERRORv("", "unknown algo ", cipher);
    string ciphertext = *out, pw;
    Crypto::Cipher cipher = Crypto::CipherInit();
    if ((ret = Crypto::CipherOpen(cipher, algo, 0, (pw = passphrase_cb("")), cbcinit))  != 1) return ERRORv("", "decrypt failed: ", ret);
    if ((ret = Crypto::CipherUpdate(cipher, ciphertext, &(*out)[0], ciphertext.size())) != 1) return ERRORv("", "decrypt failed: ", ret);
    Crypto::CipherFree(cipher);
  }
  return type;
}
#endif

}; // namespace LFL
