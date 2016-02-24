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
#include "lfapp/network.h"
#include "lfapp/crypto.h"
#include "lfapp/net/ssh.h"

#ifdef LFL_OPENSSL
#include "openssl/evp.h"
#include "openssl/rsa.h"
#include "openssl/dsa.h"
#include "openssl/ecdsa.h"
#endif

#ifdef LFL_SSH_DEBUG
#define SSHTrace(...) INFO(__VA_ARGS__)
#else
#define SSHTrace(...)
#endif

namespace LFL {
struct SSHClientConnection : public Connection::Handler {
  enum { INIT=0, FIRST_KEXINIT=1, FIRST_KEXREPLY=2, FIRST_NEWKEYS=3, KEXINIT=4, KEXREPLY=5, NEWKEYS=6 };

  SSHClient::ResponseCB cb;
  SSHClient::LoadPasswordCB load_password_cb;
  SSHClient::SavePasswordCB save_password_cb;
  string V_C, V_S, KEXINIT_C, KEXINIT_S, H_text, session_id, integrity_c2s, integrity_s2c, decrypt_buf, host, user, pw;
  int state=0, packet_len=0, packet_MAC_len=0, MAC_len_c=0, MAC_len_s=0, encrypt_block_size=0, decrypt_block_size=0;
  unsigned sequence_number_c2s=0, sequence_number_s2c=0, password_prompts=0, userauth_fail=0;
  bool guessed_c=0, guessed_s=0, guessed_right_c=0, guessed_right_s=0, loaded_pw=0;
  unsigned char padding=0, packet_id=0;
  pair<int, int> pty_channel;
  std::mt19937 rand_eng;
  BigNumContext ctx;
  BigNum K;
  Crypto::DiffieHellman dh;
  Crypto::EllipticCurveDiffieHellman ecdh;
  Crypto::DigestAlgo kex_hash;
  Crypto::Cipher encrypt, decrypt;
  Crypto::CipherAlgo cipher_algo_c2s=0, cipher_algo_s2c=0;
  Crypto::MACAlgo mac_algo_c2s=0, mac_algo_s2c=0;
  ECDef curve_id;
  int kex_method=0, hostkey_type=0, mac_prefix_c2s=0, mac_prefix_s2c=0, window_c=0, window_s=0;
  int initial_window_size=1048576, max_packet_size=32768, term_width=80, term_height=25;

  SSHClientConnection(const SSHClient::ResponseCB &CB, const string &H) : cb(CB), V_C("SSH-2.0-LFL_1.0"), host(H), rand_eng(std::random_device{}()),
    pty_channel(1,-1), ctx(NewBigNumContext()), K(NewBigNum()) { Crypto::CipherInit(&encrypt); Crypto::CipherInit(&decrypt); }
  virtual ~SSHClientConnection() { ClearPassword(); FreeBigNumContext(ctx); FreeBigNum(K); Crypto::CipherFree(&encrypt); Crypto::CipherFree(&decrypt); }

  void Close(Connection *c) { cb(c, StringPiece()); }
  int Connected(Connection *c) {
    if (state != INIT) return -1;
    string version_text = StrCat(V_C, "\r\n");
    if (c->WriteFlush(version_text) != version_text.size()) return ERRORv(-1, c->Name(), ": write");
    if (!WriteKeyExchangeInit(c, false)) return ERRORv(-1, c->Name(), ": write");
    return 0;
  }

  int Read(Connection *c) {
    if (state == INIT) {
      int processed = 0;
      StringLineIter lines(c->rb.buf, StringLineIter::Flag::BlankLines);
      for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) {
        SSHTrace(c->Name(), ": SSH_INIT: ", line);
        processed = lines.next_offset;
        if (PrefixMatch(line, "SSH-")) { V_S=line; state++; break; }
      }
      c->ReadFlush(processed);
      if (state == INIT) return 0;
    }
    for (;;) {
      bool encrypted = state > FIRST_NEWKEYS;
      if (!packet_len) {
        packet_MAC_len = MAC_len_s ? X_or_Y(mac_prefix_s2c, MAC_len_s) : 0;
        if (c->rb.size() < SSH::BinaryPacketHeaderSize || (encrypted && c->rb.size() < decrypt_block_size)) return 0;
        if (encrypted) decrypt_buf = ReadCipher(c, StringPiece(c->rb.begin(), decrypt_block_size));
        const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
        packet_len = 4 + SSH::BinaryPacketLength(packet_text, &padding, &packet_id) + packet_MAC_len;
      }
      if (c->rb.size() < packet_len) return 0;
      if (encrypted) decrypt_buf +=
        ReadCipher(c, StringPiece(c->rb.begin() + decrypt_block_size, packet_len - decrypt_block_size - packet_MAC_len));

      sequence_number_s2c++;
      const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
      Serializable::ConstStream s(packet_text + SSH::BinaryPacketHeaderSize,
                                  packet_len  - SSH::BinaryPacketHeaderSize - packet_MAC_len);
      if (encrypted && packet_MAC_len) {
        string mac = SSH::MAC(mac_algo_s2c, MAC_len_s, StringPiece(decrypt_buf.data(), packet_len - packet_MAC_len),
                              sequence_number_s2c-1, integrity_s2c, mac_prefix_s2c);
        if (mac != string(c->rb.begin() + packet_len - packet_MAC_len, packet_MAC_len))
          return ERRORv(-1, c->Name(), ": verify MAC failed");
      }

      int v;
      switch (packet_id) {
        case SSH::MSG_DISCONNECT::ID: {
          SSH::MSG_DISCONNECT msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_DISCONNECT");
          SSHTrace(c->Name(), ": MSG_DISCONNECT ", msg.reason_code, " ", msg.description.str());
        } break;

        case SSH::MSG_DEBUG::ID: {
          SSH::MSG_DEBUG msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_DEBUG");
          SSHTrace(c->Name(), ": MSG_DEBUG ", msg.message.str());
        } break;

        case SSH::MSG_KEXINIT::ID: {
          state = state == FIRST_KEXINIT ? FIRST_KEXREPLY : KEXREPLY;
          SSH::MSG_KEXINIT msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEXINIT");
          SSHTrace(c->Name(), ": MSG_KEXINIT ", msg.DebugString());

          int cipher_id_c2s=0, cipher_id_s2c=0, mac_id_c2s=0, mac_id_s2c=0;
          guessed_s = msg.first_kex_packet_follows;
          KEXINIT_S.assign(packet_text, packet_len - packet_MAC_len);
          if (!SSH::KEX   ::PreferenceIntersect(msg.kex_algorithms,                         &kex_method))    return ERRORv(-1, c->Name(), ": negotiate kex");
          if (!SSH::Key   ::PreferenceIntersect(msg.server_host_key_algorithms,             &hostkey_type))  return ERRORv(-1, c->Name(), ": negotiate hostkey");
          if (!SSH::Cipher::PreferenceIntersect(msg.encryption_algorithms_client_to_server, &cipher_id_c2s)) return ERRORv(-1, c->Name(), ": negotiate c2s cipher");
          if (!SSH::Cipher::PreferenceIntersect(msg.encryption_algorithms_server_to_client, &cipher_id_s2c)) return ERRORv(-1, c->Name(), ": negotiate s2c cipher");
          if (!SSH::MAC   ::PreferenceIntersect(msg.mac_algorithms_client_to_server,        &mac_id_c2s))    return ERRORv(-1, c->Name(), ": negotiate c2s mac");
          if (!SSH::MAC   ::PreferenceIntersect(msg.mac_algorithms_server_to_client,        &mac_id_s2c))    return ERRORv(-1, c->Name(), ": negotiate s2c mac");
          guessed_right_s = kex_method == SSH::KEX::Id(Split(msg.kex_algorithms, iscomma)) && hostkey_type == SSH::Key::Id(Split(msg.server_host_key_algorithms, iscomma));
          guessed_right_c = kex_method == 1                                                && hostkey_type == 1;
          cipher_algo_c2s = SSH::Cipher::Algo(cipher_id_c2s, &encrypt_block_size);
          cipher_algo_s2c = SSH::Cipher::Algo(cipher_id_s2c, &decrypt_block_size);
          mac_algo_c2s = SSH::MAC::Algo(mac_id_c2s, &mac_prefix_c2s);
          mac_algo_s2c = SSH::MAC::Algo(mac_id_s2c, &mac_prefix_s2c);
          INFO(c->Name(), ": ssh negotiated { kex=", SSH::KEX::Name(kex_method), ", hostkey=", SSH::Key::Name(hostkey_type),
               cipher_algo_c2s == cipher_algo_s2c ? StrCat(", cipher=", Crypto::CipherAlgos::Name(cipher_algo_c2s)) : StrCat(", cipher_c2s=", Crypto::CipherAlgos::Name(cipher_algo_c2s), ", cipher_s2c=", Crypto::CipherAlgos::Name(cipher_algo_s2c)),
               mac_id_c2s      == mac_id_s2c      ? StrCat(", mac=",               SSH::MAC::Name(mac_id_c2s))      : StrCat(", mac_c2s=",               SSH::MAC::Name(mac_id_c2s),      ", mac_s2c=",               SSH::MAC::Name(mac_id_s2c)),
               " }");
          SSHTrace(c->Name(), ": block_size=", encrypt_block_size, ",", decrypt_block_size, " mac_len=", MAC_len_c, ",", MAC_len_s);

          if (SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            switch (kex_method) {
              case SSH::KEX::ECDH_SHA2_NISTP256: curve_id=Crypto::EllipticCurve::NISTP256(); kex_hash=Crypto::DigestAlgos::SHA256(); break;
              case SSH::KEX::ECDH_SHA2_NISTP384: curve_id=Crypto::EllipticCurve::NISTP384(); kex_hash=Crypto::DigestAlgos::SHA384(); break;
              case SSH::KEX::ECDH_SHA2_NISTP521: curve_id=Crypto::EllipticCurve::NISTP521(); kex_hash=Crypto::DigestAlgos::SHA512(); break;
              default:                           return ERRORv(-1, c->Name(), ": ecdh curve");
            }
            if (!ecdh.GeneratePair(curve_id, ctx)) return ERRORv(-1, c->Name(), ": generate ecdh key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_ECDH_INIT(ecdh.c_text))) return ERRORv(-1, c->Name(), ": write");

          } else if (SSH::KEX::DiffieHellmanGroupExchange(kex_method)) {
            if      (kex_method == SSH::KEX::DHGEX_SHA1)   kex_hash = Crypto::DigestAlgos::SHA1();
            else if (kex_method == SSH::KEX::DHGEX_SHA256) kex_hash = Crypto::DigestAlgos::SHA256();
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_DH_GEX_REQUEST(dh.gex_min, dh.gex_max, dh.gex_pref)))
              return ERRORv(-1, c->Name(), ": write");

          } else if (SSH::KEX::DiffieHellman(kex_method)) {
            int secret_bits=0;
            if      (kex_method == SSH::KEX::DH14_SHA1) dh.p = Crypto::DiffieHellman::Group14Modulus(dh.g, dh.p, &secret_bits);
            else if (kex_method == SSH::KEX::DH1_SHA1)  dh.p = Crypto::DiffieHellman::Group1Modulus (dh.g, dh.p, &secret_bits);
            kex_hash = Crypto::DigestAlgos::SHA1();
            if (!dh.GeneratePair(secret_bits, ctx)) return ERRORv(-1, c->Name(), ": generate dh key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEXDH_INIT(dh.e))) return ERRORv(-1, c->Name(), ": write");

          } else return ERRORv(-1, c->Name(), "unkown kex method: ", kex_method);
        } break;

        case SSH::MSG_KEXDH_REPLY::ID:
        case SSH::MSG_KEX_DH_GEX_REPLY::ID: {
          if (state != FIRST_KEXREPLY && state != KEXREPLY) return ERRORv(-1, c->Name(), ": unexpected state ", state);
          if (guessed_s && !guessed_right_s && !(guessed_s=0)) { INFO(c->Name(), ": server guessed wrong, ignoring packet"); break; }
          if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            SSH::MSG_KEX_ECDH_REPLY msg; // MSG_KEX_ECDH_REPLY and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_ECDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEX_ECDH_REPLY");

            ecdh.s_text = msg.q_s.str();
            ECPointSetData(ecdh.g, ecdh.s, ecdh.s_text);
            if (!ecdh.ComputeSecret(&K, ctx)) return ERRORv(-1, c->Name(), ": ecdh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward

          } else if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::DiffieHellmanGroupExchange(kex_method)) {
            SSH::MSG_KEX_DH_GEX_GROUP msg(dh.p, dh.g); // MSG_KEX_DH_GEX_GROUP and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_DH_GEX_GROUP");
            SSHTrace(c->Name(), ": MSG_KEX_DH_GEX_GROUP");

            if (!dh.GeneratePair(256, ctx)) return ERRORv(-1, c->Name(), ": generate dh_gex key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_DH_GEX_INIT(dh.e))) return ERRORv(-1, c->Name(), ": write");
            break;

          } else {
            SSH::MSG_KEXDH_REPLY msg(dh.f);
            if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_KEXDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEXDH_REPLY");
            if (!dh.ComputeSecret(&K, ctx)) return ERRORv(-1, c->Name(), ": dh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward
          }

          state = state == FIRST_KEXREPLY ? FIRST_NEWKEYS : NEWKEYS;
          if (!WriteClearOrEncrypted(c, SSH::MSG_NEWKEYS())) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_NEWKEYS::ID: {
          if (state != FIRST_NEWKEYS && state != NEWKEYS) return ERRORv(-1, c->Name(), ": unexpected state ", state);
          SSHTrace(c->Name(), ": MSG_NEWKEYS");
          int key_len_c = Crypto::CipherAlgos::KeySize(cipher_algo_c2s), key_len_s = Crypto::CipherAlgos::KeySize(cipher_algo_s2c);
          if ((v = InitCipher(c, &encrypt, cipher_algo_c2s, DeriveKey(kex_hash, 'A', 24), DeriveKey(kex_hash, 'C', key_len_c), true))  != 1) return ERRORv(-1, c->Name(), ": init c->s cipher ", v, " keylen=", key_len_c);
          if ((v = InitCipher(c, &decrypt, cipher_algo_s2c, DeriveKey(kex_hash, 'B', 24), DeriveKey(kex_hash, 'D', key_len_s), false)) != 1) return ERRORv(-1, c->Name(), ": init s->c cipher ", v, " keylen=", key_len_s);
          if ((MAC_len_c = Crypto::MACAlgos::HashSize(mac_algo_c2s)) <= 0) return ERRORv(-1, c->Name(), ": invalid maclen ", encrypt_block_size);
          if ((MAC_len_s = Crypto::MACAlgos::HashSize(mac_algo_s2c)) <= 0) return ERRORv(-1, c->Name(), ": invalid maclen ", encrypt_block_size);
          integrity_c2s = DeriveKey(kex_hash, 'E', MAC_len_c);
          integrity_s2c = DeriveKey(kex_hash, 'F', MAC_len_s);
          state = NEWKEYS;
          if (!WriteCipher(c, SSH::MSG_SERVICE_REQUEST("ssh-userauth"))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_SERVICE_ACCEPT::ID: {
          SSHTrace(c->Name(), ": MSG_SERVICE_ACCEPT");
          if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "keyboard-interactive", "", "", "")))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_FAILURE::ID: {
          SSH::MSG_USERAUTH_FAILURE msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_FAILURE");
          SSHTrace(c->Name(), ": MSG_USERAUTH_FAILURE: auth_left='", msg.auth_left.str(), "'");

          if (!loaded_pw) ClearPassword();
          if (!userauth_fail++) { cb(c, "Password:"); if ((password_prompts=1)) LoadPassword(c); }
          else return ERRORv(-1, c->Name(), ": authorization failed");
        } break;

        case SSH::MSG_USERAUTH_SUCCESS::ID: {
          SSHTrace(c->Name(), ": MSG_USERAUTH_SUCCESS");
          window_s = initial_window_size;
          if (!loaded_pw) { if (save_password_cb) save_password_cb(host, user, pw); ClearPassword(); }
          if (!WriteCipher(c, SSH::MSG_CHANNEL_OPEN("session", pty_channel.first, initial_window_size, max_packet_size)))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_INFO_REQUEST::ID: {
          SSH::MSG_USERAUTH_INFO_REQUEST msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_INFO_REQUEST");
          SSHTrace(c->Name(), ": MSG_USERAUTH_INFO_REQUEST prompts=", msg.prompt.size());
          if (!msg.instruction.empty()) { SSHTrace(c->Name(), ": instruction: ", msg.instruction.str()); cb(c, msg.instruction); }
          for (auto &i : msg.prompt)    { SSHTrace(c->Name(), ": prompt: ",      i.text.str());          cb(c, i.text); }

          if ((password_prompts = msg.prompt.size())) LoadPassword(c);
          else if (!WriteCipher(c, SSH::MSG_USERAUTH_INFO_RESPONSE())) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_OPEN_CONFIRMATION::ID: {
          SSH::MSG_CHANNEL_OPEN_CONFIRMATION msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_OPEN_CONFIRMATION");
          SSHTrace(c->Name(), ": MSG_CHANNEL_OPEN_CONFIRMATION");
          pty_channel.second = msg.sender_channel;
          window_c = msg.initial_win_size;

          if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST
                           (pty_channel.second, "pty-req", point(term_width, term_height),
                            point(term_width*8, term_height*12), "screen", "", true))) return ERRORv(-1, c->Name(), ": write");

          if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(pty_channel.second, "shell", "", true))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_WINDOW_ADJUST::ID: {
          SSH::MSG_CHANNEL_WINDOW_ADJUST msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_WINDOW_ADJUST");
          SSHTrace(c->Name(), ": MSG_CHANNEL_WINDOW_ADJUST add ", msg.bytes_to_add, " to channel ", msg.recipient_channel);
          window_c += msg.bytes_to_add;
        } break;

        case SSH::MSG_CHANNEL_DATA::ID: {
          SSH::MSG_CHANNEL_DATA msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_DATA");
          SSHTrace(c->Name(), ": MSG_CHANNEL_DATA: channel ", msg.recipient_channel, ": ", msg.data.size(), " bytes");

          window_s -= (packet_len - packet_MAC_len - 4);
          if (window_s < initial_window_size / 2) {
            if (!WriteClearOrEncrypted(c, SSH::MSG_CHANNEL_WINDOW_ADJUST(pty_channel.second, initial_window_size)))
              return ERRORv(-1, c->Name(), ": write");
            window_s += initial_window_size;
          }

          cb(c, msg.data);
        } break;

        case SSH::MSG_CHANNEL_EOF::ID: {
          SSH::MSG_CHANNEL_EOF msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_EOF");
        } break;

        case SSH::MSG_CHANNEL_CLOSE::ID: {
          SSH::MSG_CHANNEL_CLOSE msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_CLOSE");
          if (!WriteCipher(c, SSH::MSG_CHANNEL_CLOSE(msg.recipient_channel))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_REQUEST::ID: {
          SSH::MSG_CHANNEL_REQUEST msg;
          if (msg.In(&s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_REQUEST");
        } break;

        case SSH::MSG_CHANNEL_SUCCESS::ID: {
          SSHTrace(c->Name(), ": MSG_CHANNEL_SUCCESS");
        } break;

        case SSH::MSG_CHANNEL_FAILURE::ID: {
          SSHTrace(c->Name(), ": MSG_CHANNEL_FAILURE");
        } break;

        default: {
          ERROR(c->Name(), " unknown packet number ", int(packet_id), " len ", packet_len);
        } break;
      }
      c->ReadFlush(packet_len);
      packet_len = 0;
    }
    return 0;
  }

  bool WriteKeyExchangeInit(Connection *c, bool guess) {
    string cipher_pref = SSH::Cipher::PreferenceCSV(), mac_pref = SSH::MAC::PreferenceCSV();
    KEXINIT_C = SSH::MSG_KEXINIT(RandBytes(16, rand_eng), SSH::KEX::PreferenceCSV(), SSH::Key::PreferenceCSV(),
                                 cipher_pref, cipher_pref, mac_pref, mac_pref, "none", "none",
                                 "", "", guess).ToString(rand_eng, 8, &sequence_number_c2s);
    SSHTrace(c->Name(), " wrote KEXINIT_C { kex=", SSH::KEX::PreferenceCSV(), " key=", SSH::Key::PreferenceCSV(),
             " cipher=", cipher_pref, " mac=", mac_pref, " }");
    return c->WriteFlush(KEXINIT_C) == KEXINIT_C.size();
  }

  int ComputeExchangeHashAndVerifyHostKey(Connection *c, const StringPiece &k_s, const StringPiece &h_sig) {
    H_text = SSH::ComputeExchangeHash(kex_method, kex_hash, V_C, V_S, KEXINIT_C, KEXINIT_S, k_s, K, &dh, &ecdh);
    if (state == FIRST_KEXREPLY) session_id = H_text;
    SSHTrace(c->Name(), ": H = \"", CHexEscape(H_text), "\"");
    return SSH::VerifyHostKey(H_text, hostkey_type, k_s, h_sig);
  }

  string DeriveKey(Crypto::DigestAlgo algo, char ID, int bytes) {
    return SSH::DeriveKey(algo, session_id, H_text, K, ID, bytes);
  }

  int InitCipher(Connection *c, Crypto::Cipher *cipher, Crypto::CipherAlgo algo, const string &IV, const string &key, bool dir) {
    SSHTrace(c->Name(), ": ", dir ? "C->S" : "S->C", " IV  = \"", CHexEscape(IV),  "\"");
    SSHTrace(c->Name(), ": ", dir ? "C->S" : "S->C", " key = \"", CHexEscape(key), "\"");
    Crypto::CipherFree(cipher);
    Crypto::CipherInit(cipher);
    return Crypto::CipherOpen(cipher, algo, dir, key, IV);
  }

  string ReadCipher(Connection *c, const StringPiece &m) {
    string dec_text(m.size(), 0);
    if (Crypto::CipherUpdate(&decrypt, m, &dec_text[0], dec_text.size()) != 1) return ERRORv("", c->Name(), ": decrypt failed");
    return dec_text;
  }

  int WriteCipher(Connection *c, const string &m) {
    string enc_text(m.size(), 0);
    if (Crypto::CipherUpdate(&encrypt, m, &enc_text[0], enc_text.size()) != 1) return ERRORv(-1, c->Name(), ": encrypt failed");
    enc_text += SSH::MAC(mac_algo_c2s, MAC_len_c, m, sequence_number_c2s-1, integrity_c2s, mac_prefix_c2s);
    return c->WriteFlush(enc_text) == m.size() + X_or_Y(mac_prefix_c2s, MAC_len_c);
  }

  bool WriteCipher(Connection *c, const SSH::Serializable &m) {
    string text = m.ToString(rand_eng, encrypt_block_size, &sequence_number_c2s);
    return WriteCipher(c, text);
  }

  bool WriteClearOrEncrypted(Connection *c, const SSH::Serializable &m) {
    if (state > FIRST_NEWKEYS) return WriteCipher(c, m);
    string text = m.ToString(rand_eng, encrypt_block_size, &sequence_number_c2s);
    return c->WriteFlush(text) == text.size();
  }

  int WriteChannelData(Connection *c, const StringPiece &b) {
    if (!password_prompts) {
      if (!WriteCipher(c, SSH::MSG_CHANNEL_DATA(pty_channel.second, b))) return ERRORv(-1, c->Name(), ": write");
      window_c -= (b.size() - 4);
    } else {
      bool cr = b.len && b.back() == '\r';
      pw.append(b.data(), b.size() - cr);
      if (cr && !WritePassword(c)) return ERRORv(-1, c->Name(), ": write");
    }
    return b.size();
  }

  bool WritePassword(Connection *c) {
    cb(c, "\r\n");
    bool success = false;
    if (userauth_fail) {
      success = WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "password", "", pw, ""));
    } else {
      vector<StringPiece> prompt(password_prompts);
      prompt.back() = StringPiece(pw.data(), pw.size());
      success = WriteCipher(c, SSH::MSG_USERAUTH_INFO_RESPONSE(prompt));
    }
    password_prompts = 0;
    return success;
  }

  void LoadPassword(Connection *c) {
    if ((loaded_pw = load_password_cb && load_password_cb(host, user, &pw))) WritePassword(c);
    if (loaded_pw) ClearPassword();
  }

  void ClearPassword() { pw.assign(pw.size(), ' '); pw.clear(); }
  void SetPasswordCB(const SSHClient::LoadPasswordCB &L, const SSHClient::SavePasswordCB &S) { load_password_cb=L; save_password_cb=S; }
  int SetTerminalWindowSize(Connection *c, int w, int h) {
    term_width = w;
    term_height = h;
    if (!c || c->state != Connection::Connected || state <= FIRST_NEWKEYS || pty_channel.second < 0) return 0;
    if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(pty_channel.second, "window-change", point(term_width, term_height),
                                                 point(term_width*8, term_height*12),
                                                 "", "", false))) return ERRORv(-1, c->Name(), ": write");
    return 0;
  }
};

Connection *SSHClient::Open(const string &hostport, const SSHClient::ResponseCB &cb, Callback *detach) { 
  Connection *c = app->net->tcp_client->Connect(hostport, 22, detach);
  if (!c) return 0;
  c->handler = make_unique<SSHClientConnection>(cb, hostport);
  return c;
}
int  SSHClient::WriteChannelData     (Connection *c, const StringPiece &b)                     { return dynamic_cast<SSHClientConnection*>(c->handler.get())->WriteChannelData(c, b); }
int  SSHClient::SetTerminalWindowSize(Connection *c, int w, int h)                             { return dynamic_cast<SSHClientConnection*>(c->handler.get())->SetTerminalWindowSize(c, w, h); }
void SSHClient::SetUser              (Connection *c, const string &user)                       { dynamic_cast<SSHClientConnection*>(c->handler.get())->user = user; }
void SSHClient::SetPasswordCB(Connection *c, const LoadPasswordCB &L, const SavePasswordCB &S) { dynamic_cast<SSHClientConnection*>(c->handler.get())->SetPasswordCB(L, S); }

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
