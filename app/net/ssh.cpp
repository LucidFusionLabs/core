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

#include "core/app/network.h"
#include "core/app/crypto.h"
#include "core/app/net/ssh.h"

#ifdef LFL_SSH_DEBUG
#define SSHTrace(...) INFO(__VA_ARGS__)
#else
#define SSHTrace(...)
#endif

namespace LFL {
struct SSHClientConnection : public Connection::Handler {
  enum { INIT=0, FIRST_KEXINIT=1, FIRST_KEXREPLY=2, FIRST_NEWKEYS=3, KEXINIT=4, KEXREPLY=5, NEWKEYS=6 };
  struct Channel {
    int local_id, remote_id, window_c=0, window_s=0;
    string buf;
    Channel(int L=0, int R=0) : local_id(L), remote_id(R) {}
  };

  SSHClient::ResponseCB cb;
  SSHClient::LoadIdentityCB load_identity_cb;
  SSHClient::LoadPasswordCB load_password_cb;
  Callback success_cb;
  shared_ptr<SSHClient::Identity> identity;
  string V_C, V_S, KEXINIT_C, KEXINIT_S, H_text, session_id, integrity_c2s, integrity_s2c, decrypt_buf, host, user, pw, termvar;
  int state=0, packet_len=0, packet_MAC_len=0, MAC_len_c=0, MAC_len_s=0, encrypt_block_size=0, decrypt_block_size=0, server_version=0;
  unsigned sequence_number_c2s=0, sequence_number_s2c=0, password_prompts=0, userauth_fail=0;
  bool guessed_c=0, guessed_s=0, guessed_right_c=0, guessed_right_s=0, loaded_pw=0, agent_forwarding;
  unsigned char padding=0, packet_id=0;
  Serializable::ConstStream packet_s;
  std::mt19937 rand_eng;
  unordered_map<int, Channel> channels;
  Channel *session_channel=0, *agent_channel=0;
  BigNumContext ctx;
  BigNum K;
  Crypto::DiffieHellman dh;
  Crypto::EllipticCurveDiffieHellman ecdh;
  Crypto::X25519DiffieHellman x25519dh;
  Crypto::DigestAlgo kex_hash;
  Crypto::Cipher encrypt, decrypt;
  Crypto::CipherAlgo cipher_algo_c2s, cipher_algo_s2c;
  Crypto::MACAlgo mac_algo_c2s, mac_algo_s2c;
  ECDef curve_id;
  int next_channel_id=1, dont_compress, compress_id_c2s, compress_id_s2c;
  int kex_method=0, hostkey_type=0, mac_prefix_c2s=0, mac_prefix_s2c=0;
  int initial_window_size=1048576, max_packet_size=32768, term_width=80, term_height=25;
  unique_ptr<ZLibReader> zreader;
  unique_ptr<ZLibWriter> zwriter;

  SSHClientConnection(const SSHClient::ResponseCB &CB, const string &H, const string &U, const string &T, bool compress, bool agent_fwd, const Callback &s) :
    cb(CB), success_cb(s), V_C("SSH-2.0-LFL_1.0"), host(H), user(U), termvar(T), agent_forwarding(agent_fwd), rand_eng(std::random_device{}()),
    ctx(NewBigNumContext()), K(NewBigNum()), encrypt(Crypto::CipherInit()), decrypt(Crypto::CipherInit()), dont_compress(compress ? 0 : SSH::Compression::None-1) {}
  virtual ~SSHClientConnection() { ClearPassword(); FreeBigNumContext(ctx); FreeBigNum(K); Crypto::CipherFree(encrypt); Crypto::CipherFree(decrypt); }

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
        if (PrefixMatch(line, "SSH-")) { V_S=line; server_version=atoi(line.data()+4); state++; break; }
      }
      c->ReadFlush(processed);
      if (state == INIT) return 0;
    }

    Channel *chan;
    for (;;) {
      bool encrypted = state > FIRST_NEWKEYS;
      if (!packet_len) {
        packet_MAC_len = MAC_len_s ? X_or_Y(mac_prefix_s2c, MAC_len_s) : 0;
        if (c->rb.size() < SSH::BinaryPacketHeaderSize || (encrypted && c->rb.size() < decrypt_block_size)) return 0;
        if (encrypted) decrypt_buf = ReadCipher(c, StringPiece(c->rb.begin(), decrypt_block_size));
        const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
        packet_len = 4 + SSH::BinaryPacketLength(packet_text, &padding) + packet_MAC_len;
      }
      if (c->rb.size() < packet_len) return 0;
      if (encrypted) decrypt_buf +=
        ReadCipher(c, StringPiece(c->rb.begin() + decrypt_block_size, packet_len - decrypt_block_size - packet_MAC_len));

      sequence_number_s2c++;
      if (encrypted && packet_MAC_len) {
        string mac = SSH::MAC(mac_algo_s2c, MAC_len_s, StringPiece(decrypt_buf.data(), packet_len - packet_MAC_len),
                              sequence_number_s2c-1, integrity_s2c, mac_prefix_s2c);
        if (mac != string(c->rb.begin() + packet_len - packet_MAC_len, packet_MAC_len))
          return ERRORv(-1, c->Name(), ": verify MAC failed");
      }

      const char *packet_text = encrypted ? decrypt_buf.data() : c->rb.begin();
      if (zreader) {
        zreader->out.clear();
        if (!zreader->Add(StringPiece(packet_text + SSH::BinaryPacketHeaderSize,
                                      packet_len  - SSH::BinaryPacketHeaderSize - packet_MAC_len - padding),
                          true)) ERRORv(-1, c->Name(), ": decompress failed");
        packet_s = Serializable::ConstStream((packet_text = zreader->out.data()), zreader->out.size());
      } else packet_s = Serializable::ConstStream(packet_text + SSH::BinaryPacketHeaderSize,
                                                  packet_len  - SSH::BinaryPacketHeaderSize - packet_MAC_len - padding);

      int v;
      packet_s.Read8(&packet_id);
      if (packet_s.error) return ERRORv(-1, c->Name(), ": read packet_id");

      switch (packet_id) {
        case SSH::MSG_DISCONNECT::ID: {
          SSH::MSG_DISCONNECT msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_DISCONNECT");
          SSHTrace(c->Name(), ": MSG_DISCONNECT ", msg.reason_code, " ", msg.description.str());
        } break;

        case SSH::MSG_IGNORE::ID: {
          SSH::MSG_IGNORE msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_IGNORE");
          SSHTrace(c->Name(), ": MSG_IGNORE");
        } break;

        case SSH::MSG_DEBUG::ID: {
          SSH::MSG_DEBUG msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_DEBUG");
          SSHTrace(c->Name(), ": MSG_DEBUG ", msg.message.str());
        } break;

        case SSH::MSG_KEXINIT::ID: {
          state = state == FIRST_KEXINIT ? FIRST_KEXREPLY : KEXREPLY;
          SSH::MSG_KEXINIT msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_KEXINIT");
          SSHTrace(c->Name(), ": MSG_KEXINIT ", msg.DebugString());

          int cipher_id_c2s=0, cipher_id_s2c=0, mac_id_c2s=0, mac_id_s2c=0;
          guessed_s = msg.first_kex_packet_follows;
          KEXINIT_S.assign(packet_text, packet_len - packet_MAC_len);
          if (!SSH::KEX        ::PreferenceIntersect(msg.kex_algorithms,                          &kex_method))                     return ERRORv(-1, c->Name(), ": negotiate kex");
          if (!SSH::Key        ::PreferenceIntersect(msg.server_host_key_algorithms,              &hostkey_type))                   return ERRORv(-1, c->Name(), ": negotiate hostkey");
          if (!SSH::Cipher     ::PreferenceIntersect(msg.encryption_algorithms_client_to_server,  &cipher_id_c2s))                  return ERRORv(-1, c->Name(), ": negotiate c2s cipher");
          if (!SSH::Cipher     ::PreferenceIntersect(msg.encryption_algorithms_server_to_client,  &cipher_id_s2c))                  return ERRORv(-1, c->Name(), ": negotiate s2c cipher");
          if (!SSH::MAC        ::PreferenceIntersect(msg.mac_algorithms_client_to_server,         &mac_id_c2s))                     return ERRORv(-1, c->Name(), ": negotiate c2s mac");
          if (!SSH::MAC        ::PreferenceIntersect(msg.mac_algorithms_server_to_client,         &mac_id_s2c))                     return ERRORv(-1, c->Name(), ": negotiate s2c mac");
          if (!SSH::Compression::PreferenceIntersect(msg.compression_algorithms_client_to_server, &compress_id_c2s, dont_compress)) return ERRORv(-1, c->Name(), ": negotiate c2s compression");
          if (!SSH::Compression::PreferenceIntersect(msg.compression_algorithms_server_to_client, &compress_id_s2c, dont_compress)) return ERRORv(-1, c->Name(), ": negotiate s2c compression");
          guessed_right_s = kex_method == SSH::KEX::Id(Split(msg.kex_algorithms, iscomma)) && hostkey_type == SSH::Key::Id(Split(msg.server_host_key_algorithms, iscomma));
          guessed_right_c = kex_method == 1                                                && hostkey_type == 1;
          cipher_algo_c2s = SSH::Cipher::Algo(cipher_id_c2s, &encrypt_block_size);
          cipher_algo_s2c = SSH::Cipher::Algo(cipher_id_s2c, &decrypt_block_size);
          mac_algo_c2s = SSH::MAC::Algo(mac_id_c2s, &mac_prefix_c2s);
          mac_algo_s2c = SSH::MAC::Algo(mac_id_s2c, &mac_prefix_s2c);
          INFO(c->Name(), ": ssh negotiated { kex=", SSH::KEX::Name(kex_method), ", hostkey=", SSH::Key::Name(hostkey_type),
               cipher_algo_c2s == cipher_algo_s2c ? StrCat(", cipher=", Crypto::CipherAlgos::Name(cipher_algo_c2s)) : StrCat(", cipher_c2s=", Crypto::CipherAlgos::Name(cipher_algo_c2s), ", cipher_s2c=", Crypto::CipherAlgos::Name(cipher_algo_s2c)),
               mac_id_c2s      == mac_id_s2c      ? StrCat(", mac=",               SSH::MAC::Name(mac_id_c2s))      : StrCat(", mac_c2s=",               SSH::MAC::Name(mac_id_c2s),      ", mac_s2c=",               SSH::MAC::Name(mac_id_s2c)),
               compress_id_c2s == compress_id_s2c ? StrCat(", compress=",  SSH::Compression::Name(compress_id_c2s)) : StrCat(", compress_c2s=",  SSH::Compression::Name(compress_id_c2s), ", compress_s2c=",  SSH::Compression::Name(compress_id_s2c)),
               " }");
          SSHTrace(c->Name(), ": block_size=", encrypt_block_size, ",", decrypt_block_size, " mac_len=", MAC_len_c, ",", MAC_len_s);

          if (SSH::KEX::X25519DiffieHellman(kex_method)) {
            kex_hash = Crypto::DigestAlgos::SHA256();
            x25519dh.GeneratePair(rand_eng);
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_ECDH_INIT(x25519dh.mypubkey))) return ERRORv(-1, c->Name(), ": write");

          } else if (SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            switch (kex_method) {
              case SSH::KEX::ECDH_SHA2_NISTP256: curve_id=Crypto::EllipticCurve::NISTP256(); kex_hash=Crypto::DigestAlgos::SHA256(); break;
              case SSH::KEX::ECDH_SHA2_NISTP384: curve_id=Crypto::EllipticCurve::NISTP384(); kex_hash=Crypto::DigestAlgos::SHA384(); break;
              case SSH::KEX::ECDH_SHA2_NISTP521: curve_id=Crypto::EllipticCurve::NISTP521(); kex_hash=Crypto::DigestAlgos::SHA512(); break;
              default:                           return ERRORv(-1, c->Name(), ": ecdh curve");
            }
            if (!ecdh.GeneratePair(curve_id, ctx)) return ERRORv(-1, c->Name(), ": generate ecdh key(", kex_method, "): ", Crypto::GetLastErrorText());
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
          
          if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::X25519DiffieHellman(kex_method)) {
            SSH::MSG_KEX_ECDH_REPLY msg; // MSG_KEX_ECDH_REPLY and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_ECDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEX_ECDH_REPLY for X25519DH");

            x25519dh.remotepubkey = msg.q_s.str();
            if (!x25519dh.ComputeSecret(&K)) return ERRORv(-1, c->Name(), ": x25519dh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward

          } else if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::EllipticCurveDiffieHellman(kex_method)) {
            SSH::MSG_KEX_ECDH_REPLY msg; // MSG_KEX_ECDH_REPLY and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_ECDH_REPLY");
            SSHTrace(c->Name(), ": MSG_KEX_ECDH_REPLY for ECDH");

            ecdh.s_text = msg.q_s.str();
            ECPointSetData(ecdh.g, ecdh.s, ecdh.s_text);
            if (!ecdh.ComputeSecret(&K, ctx)) return ERRORv(-1, c->Name(), ": ecdh");
            if ((v = ComputeExchangeHashAndVerifyHostKey(c, msg.k_s, msg.h_sig)) != 1) return ERRORv(-1, c->Name(), ": verify hostkey failed: ", v);
            // fall forward

          } else if (packet_id == SSH::MSG_KEXDH_REPLY::ID && SSH::KEX::DiffieHellmanGroupExchange(kex_method)) {
            SSH::MSG_KEX_DH_GEX_GROUP msg(dh.p, dh.g); // MSG_KEX_DH_GEX_GROUP and MSG_KEXDH_REPLY share ID 31
            if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_KEX_DH_GEX_GROUP");
            SSHTrace(c->Name(), ": MSG_KEX_DH_GEX_GROUP");

            if (!dh.GeneratePair(256, ctx)) return ERRORv(-1, c->Name(), ": generate dh_gex key");
            if (!WriteClearOrEncrypted(c, SSH::MSG_KEX_DH_GEX_INIT(dh.e))) return ERRORv(-1, c->Name(), ": write");
            break;

          } else {
            SSH::MSG_KEXDH_REPLY msg(dh.f);
            if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_KEXDH_REPLY");
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
          if (load_identity_cb) load_identity_cb(&identity);

          if (!identity) { /**/ }
          else if (identity->ed25519.privkey.size()) {
            string pubkey = SSH::Ed25519Key(identity->ed25519.pubkey).ToString();
            string challenge = SSH::DeriveChallengeText(session_id, user, "ssh-connection", "publickey", "ssh-ed25519", pubkey);
            string sig = SSH::Ed25519Signature(Ed25519Sign(challenge, identity->ed25519.privkey)).ToString();
            if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "publickey", "ssh-ed25519", pubkey, sig)))
              return ERRORv(-1, c->Name(), ": write");
            break;

          } else if (identity->ec) {
            ECGroup group = GetECPairGroup(identity->ec);
            string algo_name, curve_name;
            Crypto::DigestAlgo hash_id;
            if (!GetECName(GetECGroupID(group), &algo_name, &curve_name, &hash_id))
            { ERROR("unknown curve_id ", GetECGroupID(group).v); break; }

            string pubkey = SSH::ECDSAKey(algo_name, curve_name,
                                          ECPointGetData(group, GetECPairPubKey(identity->ec), ctx)).ToString();
            string challenge = SSH::DeriveChallenge(hash_id, session_id, user, "ssh-connection", "publickey", algo_name, pubkey);
            ECDSASig ecdsa_sig = ECDSASign(challenge, identity->ec);
            if (!ecdsa_sig) { ERROR("ECDSASign failed: ", Crypto::GetLastErrorText()); break; }
            string sig = SSH::ECDSASignature(algo_name, GetECDSASigR(ecdsa_sig), GetECDSASigS(ecdsa_sig)).ToString();
            ECDSASigFree(ecdsa_sig);
            if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "publickey", algo_name, pubkey, sig)))
              return ERRORv(-1, c->Name(), ": write");
            break;

          } else if (identity->rsa) {
            string sig, pubkey = SSH::RSAKey(GetRSAKeyE(identity->rsa), GetRSAKeyN(identity->rsa)).ToString();
            string challenge = SSH::DeriveChallenge(Crypto::DigestAlgos::SHA1(), session_id, user, "ssh-connection", "publickey", "ssh-rsa", pubkey);
            if (RSASign(challenge, &sig, identity->rsa) != 1) { ERROR("RSASign failed: ", Crypto::GetLastErrorText()); break; }
            if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "publickey", "ssh-rsa", pubkey,
                                                          SSH::RSASignature(sig).ToString())))
              return ERRORv(-1, c->Name(), ": write");
            break;

          } else if (identity->dsa) {
            string pubkey = SSH::DSSKey(GetDSAKeyP(identity->dsa), GetDSAKeyQ(identity->dsa),
                                        GetDSAKeyG(identity->dsa), GetDSAKeyK(identity->dsa)).ToString();
            string challenge = SSH::DeriveChallenge(Crypto::DigestAlgos::SHA1(), session_id, user, "ssh-connection", "publickey", "ssh-dss", pubkey);
            DSASig dsa_sig = DSASign(challenge, identity->dsa);
            if (!dsa_sig) { ERROR("DSASign failed: ", Crypto::GetLastErrorText()); break; }
            string sig = SSH::DSSSignature(GetDSASigR(dsa_sig), GetDSASigS(dsa_sig)).ToString();
            DSASigFree(dsa_sig);
            if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "publickey", "ssh-dss", pubkey, sig)))
              return ERRORv(-1, c->Name(), ": write");
            break;
          }

          if (!WriteCipher(c, SSH::MSG_USERAUTH_REQUEST(user, "ssh-connection", "keyboard-interactive", "", "", "")))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_FAILURE::ID: {
          SSH::MSG_USERAUTH_FAILURE msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_FAILURE");
          SSHTrace(c->Name(), ": MSG_USERAUTH_FAILURE: auth_left='", msg.auth_left.str(), "'");

          if (!loaded_pw) ClearPassword();
          if (!userauth_fail++) { cb(c, "Password:"); if ((password_prompts=1)) LoadPassword(c); }
          else return ERRORv(-1, c->Name(), ": authorization failed");
        } break;

        case SSH::MSG_USERAUTH_SUCCESS::ID: {
          SSHTrace(c->Name(), ": MSG_USERAUTH_SUCCESS");
          session_channel = &channels[next_channel_id];
          session_channel->local_id = next_channel_id++;
          session_channel->window_s = initial_window_size;
          if (compress_id_c2s == SSH::Compression::OpenSSHZLib) zreader = make_unique<ZLibReader>(4096);
          if (compress_id_s2c == SSH::Compression::OpenSSHZLib) zwriter = make_unique<ZLibWriter>(4096);
          if (success_cb) success_cb();
          if (!WriteCipher(c, SSH::MSG_CHANNEL_OPEN("session", session_channel->local_id, initial_window_size, max_packet_size)))
            return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_USERAUTH_INFO_REQUEST::ID: {
          SSH::MSG_USERAUTH_INFO_REQUEST msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_USERAUTH_INFO_REQUEST");
          SSHTrace(c->Name(), ": MSG_USERAUTH_INFO_REQUEST prompts=", msg.prompt.size());
          if (!msg.instruction.empty()) { SSHTrace(c->Name(), ": instruction: ", msg.instruction.str()); cb(c, msg.instruction); }
          for (auto &i : msg.prompt)    { SSHTrace(c->Name(), ": prompt: ",      i.text.str());          cb(c, i.text); }

          if ((password_prompts = msg.prompt.size())) LoadPassword(c);
          else if (!WriteCipher(c, SSH::MSG_USERAUTH_INFO_RESPONSE())) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_GLOBAL_REQUEST::ID: {
          SSH::MSG_GLOBAL_REQUEST msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_GLOBAL_REQUEST");
          SSHTrace(c->Name(), ": MSG_GLOBAL_REQUEST request=", msg.request.str());
        } break;

        case SSH::MSG_CHANNEL_OPEN::ID: {
          SSH::MSG_CHANNEL_OPEN msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_OPEN");
          SSHTrace(c->Name(), ": MSG_CHANNEL_OPEN type=", msg.channel_type.str());
          if (msg.channel_type.str() == "auth-agent@openssh.com" && agent_forwarding) {
            if (agent_channel) return ERRORv(-1, c->Name(), ": agent channel already open");
            agent_channel = &channels[next_channel_id];
            agent_channel->local_id = next_channel_id++;
            agent_channel->remote_id = msg.sender_channel;
            agent_channel->window_c = msg.initial_win_size;
            agent_channel->window_s = initial_window_size;
            if (!WriteCipher(c, SSH::MSG_CHANNEL_OPEN_CONFIRMATION(agent_channel->remote_id, agent_channel->local_id, agent_channel->window_s, max_packet_size))) return ERRORv(-1, c->Name(), ": write");
          } else {
            ERROR("unknown channel open", msg.channel_type.str());
            if (!WriteCipher(c, SSH::MSG_CHANNEL_OPEN_FAILURE(agent_channel->remote_id, 0, "", ""))) return ERRORv(-1, c->Name(), ": write");
          }
        } break;

        case SSH::MSG_CHANNEL_OPEN_CONFIRMATION::ID: {
          SSH::MSG_CHANNEL_OPEN_CONFIRMATION msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_OPEN_CONFIRMATION");
          SSHTrace(c->Name(), ": MSG_CHANNEL_OPEN_CONFIRMATION local_id=", msg.recipient_channel, " remote_id=", msg.sender_channel);
          if (!(chan = FindPtrOrNull(channels, msg.recipient_channel)) || chan->remote_id) return ERRORv(-1, c->Name(), ": open invalid channel");
          chan->remote_id = msg.sender_channel;
          chan->window_c = msg.initial_win_size;

          if (chan == session_channel) {
            if (agent_forwarding) {
              if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(chan->remote_id, "auth-agent-req@openssh.com", "", true))) return ERRORv(-1, c->Name(), ": write");
            }

            if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST
                             (chan->remote_id, "pty-req", point(term_width, term_height),
                              point(term_width*8, term_height*12), termvar, "", true))) return ERRORv(-1, c->Name(), ": write");

            if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(chan->remote_id, "shell", "", true))) return ERRORv(-1, c->Name(), ": write");
          }
        } break;

        case SSH::MSG_CHANNEL_WINDOW_ADJUST::ID: {
          SSH::MSG_CHANNEL_WINDOW_ADJUST msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_WINDOW_ADJUST");
          SSHTrace(c->Name(), ": MSG_CHANNEL_WINDOW_ADJUST add ", msg.bytes_to_add, " to channel ", msg.recipient_channel);
          if (!(chan = FindPtrOrNull(channels, msg.recipient_channel))) return ERRORv(-1, c->Name(), ": window adjust invalid channel");
          chan->window_c += msg.bytes_to_add;
        } break;

        case SSH::MSG_CHANNEL_DATA::ID: {
          SSH::MSG_CHANNEL_DATA msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_DATA");
          SSHTrace(c->Name(), ": MSG_CHANNEL_DATA: channel ", msg.recipient_channel, ": ", msg.data.size(), " bytes");
          if (!(chan = FindPtrOrNull(channels, msg.recipient_channel))) return ERRORv(-1, c->Name(), ": data for invalid channel");

          chan->window_s -= (packet_len - packet_MAC_len - 4);
          if (chan->window_s < initial_window_size / 2) {
            if (!WriteClearOrEncrypted(c, SSH::MSG_CHANNEL_WINDOW_ADJUST(chan->remote_id, initial_window_size)))
              return ERRORv(-1, c->Name(), ": write");
            chan->window_s += initial_window_size;
          }

          if (chan == session_channel) cb(c, msg.data);
          else if (chan == agent_channel) {
            chan->buf.append(msg.data.buf, msg.data.len);
            while(chan->buf.size() > 4) {
              int agent_packet_len;
              Serializable::ConstStream agent_stream(chan->buf.data(), chan->buf.size());
              agent_stream.Ntohl(&agent_packet_len);
              if (agent_stream.Remaining() < agent_packet_len) break;
              unsigned char agent_packet_id;
              agent_stream.Read8(&agent_packet_id);
              switch(agent_packet_id) {
                case SSH::AGENTC_REQUEST_IDENTITIES::ID: {
                  SSHTrace(c->Name(), ": agent channel: AGENTC_REQUEST_IDENTITIES");
                  SSH::AGENT_IDENTITIES_ANSWER reply;
                  if (identity) {
                    if (identity->ed25519.privkey.size()) {
                      reply.keys.emplace_back(SSH::Ed25519Key(identity->ed25519.pubkey).ToString(), "");
                    }
                    if (identity->ec) {
                      ECGroup group = GetECPairGroup(identity->ec);
                      string algo_name, curve_name;
                      Crypto::DigestAlgo hash_id;
                      if (GetECName(GetECGroupID(group), &algo_name, &curve_name, &hash_id))
                        reply.keys.emplace_back(SSH::ECDSAKey(algo_name, curve_name,
                                                              ECPointGetData(group, GetECPairPubKey(identity->ec), ctx)).ToString(), "");
                    }
                    if (identity->rsa) {
                      reply.keys.emplace_back(SSH::RSAKey(GetRSAKeyE(identity->rsa), GetRSAKeyN(identity->rsa)).ToString(), "");
                    }
                    if (identity->dsa) {
                      reply.keys.emplace_back(SSH::DSSKey(GetDSAKeyP(identity->dsa), GetDSAKeyQ(identity->dsa),
                                                          GetDSAKeyG(identity->dsa), GetDSAKeyK(identity->dsa)).ToString(), "");
                    }
                  }
                  if (!WriteToChannel(c, chan, reply.ToString())) return -1;
                } break;

                case SSH::AGENTC_SIGN_REQUEST::ID: {
                  SSH::AGENTC_SIGN_REQUEST agent_msg;
                  if (agent_msg.In(&agent_stream)) return ERRORv(-1, c->Name(), ": read AGENTC_SIGN_REQUEST");
                  Serializable::ConstStream key_stream(agent_msg.key.data(), agent_msg.key.size());
                  string key_type, sig;
                  key_stream.ReadString(&key_type);

                  if (key_type == SSH::Key::Name(SSH::Key::ED25519)) {
                    sig = SSH::Ed25519Signature(Ed25519Sign(agent_msg.data, identity->ed25519.privkey)).ToString();
                  } else if (key_type == SSH::Key::Name(SSH::Key::ECDSA_SHA2_NISTP256) ||
                             key_type == SSH::Key::Name(SSH::Key::ECDSA_SHA2_NISTP384) ||
                             key_type == SSH::Key::Name(SSH::Key::ECDSA_SHA2_NISTP521)) {
                    ECDSASig ecdsa_sig = ECDSASign(agent_msg.data, identity->ec);
                    if (ecdsa_sig) {
                      sig = SSH::ECDSASignature(key_type, GetECDSASigR(ecdsa_sig), GetECDSASigS(ecdsa_sig)).ToString();
                      ECDSASigFree(ecdsa_sig);
                    }
                  } else if (key_type == SSH::Key::Name(SSH::Key::RSA)) {
                    if (RSASign(agent_msg.data, &sig, identity->rsa) == 1) sig = SSH::RSASignature(sig).ToString();
                  } else if (key_type == SSH::Key::Name(SSH::Key::DSS)) {
                    DSASig dsa_sig = DSASign(agent_msg.data, identity->dsa);
                    if (dsa_sig) {
                      sig = SSH::DSSSignature(GetDSASigR(dsa_sig), GetDSASigS(dsa_sig)).ToString();
                      DSASigFree(dsa_sig);
                    }
                  }

                  if (sig.size()) {
                    if (!WriteToChannel(c, chan, SSH::AGENT_SIGN_RESPONSE(sig).ToString())) return -1;
                  } else {
                    if (!WriteToChannel(c, chan, SSH::AGENT_FAILURE().ToString())) return -1;
                  }
                } break;

                default: {
                  ERROR(c->Name(), " unknown agent packet number ", int(agent_packet_id), " len ", agent_packet_len);
                } break;
              };
              chan->buf.erase(0, agent_packet_len + 4);
            }
          }
        } break;

        case SSH::MSG_CHANNEL_EOF::ID: {
          SSH::MSG_CHANNEL_EOF msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_EOF");
        } break;

        case SSH::MSG_CHANNEL_CLOSE::ID: {
          SSH::MSG_CHANNEL_CLOSE msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_CLOSE");
          if (!WriteCipher(c, SSH::MSG_CHANNEL_CLOSE(msg.recipient_channel))) return ERRORv(-1, c->Name(), ": write");
        } break;

        case SSH::MSG_CHANNEL_REQUEST::ID: {
          SSH::MSG_CHANNEL_REQUEST msg;
          if (msg.In(&packet_s)) return ERRORv(-1, c->Name(), ": read MSG_CHANNEL_REQUEST");
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
    string cipher_pref = SSH::Cipher::PreferenceCSV(), mac_pref = SSH::MAC::PreferenceCSV(), compress_pref = SSH::Compression::PreferenceCSV(dont_compress);
    KEXINIT_C = SSH::MSG_KEXINIT(RandBytes(16, rand_eng), SSH::KEX::PreferenceCSV(), SSH::Key::PreferenceCSV(),
                                 cipher_pref, cipher_pref, mac_pref, mac_pref, compress_pref, compress_pref,
                                 "", "", guess).ToString(nullptr, rand_eng, 8, &sequence_number_c2s);
    SSHTrace(c->Name(), " wrote KEXINIT_C { kex=", SSH::KEX::PreferenceCSV(), " key=", SSH::Key::PreferenceCSV(),
             " cipher=", cipher_pref, " mac=", mac_pref, " compress=", compress_pref, " }");
    return c->WriteFlush(KEXINIT_C) == KEXINIT_C.size();
  }

  int ComputeExchangeHashAndVerifyHostKey(Connection *c, const StringPiece &k_s, const StringPiece &h_sig) {
    H_text = SSH::ComputeExchangeHash(kex_method, kex_hash, V_C, V_S, KEXINIT_C, KEXINIT_S, k_s, K, &dh, &ecdh, &x25519dh);
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
    Crypto::CipherFree(*cipher);
    *cipher = Crypto::CipherInit();
    return Crypto::CipherOpen(*cipher, algo, dir, key, IV);
  }

  string ReadCipher(Connection *c, const StringPiece &m) {
    string dec_text(m.size(), 0);
    if (Crypto::CipherUpdate(decrypt, m, &dec_text[0], dec_text.size()) != 1) return ERRORv("", c->Name(), ": decrypt failed");
    return dec_text;
  }

  int WriteCipher(Connection *c, const string &m) {
    string enc_text(m.size(), 0);
    if (Crypto::CipherUpdate(encrypt, m, &enc_text[0], enc_text.size()) != 1) return ERRORv(-1, c->Name(), ": encrypt failed");
    enc_text += SSH::MAC(mac_algo_c2s, MAC_len_c, m, sequence_number_c2s-1, integrity_c2s, mac_prefix_c2s);
    return c->WriteFlush(enc_text) == m.size() + X_or_Y(mac_prefix_c2s, MAC_len_c);
  }

  bool WriteCipher(Connection *c, const SSH::Serializable &m) {
    string text = m.ToString(zwriter.get(), rand_eng, encrypt_block_size, &sequence_number_c2s);
    return WriteCipher(c, text);
  }

  bool WriteClearOrEncrypted(Connection *c, const SSH::Serializable &m) {
    if (state > FIRST_NEWKEYS) return WriteCipher(c, m);
    string text = m.ToString(nullptr, rand_eng, encrypt_block_size, &sequence_number_c2s);
    return c->WriteFlush(text) == text.size();
  }

  int WriteChannelData(Connection *c, const StringPiece &b) {
    if (!password_prompts) {
      if (!WriteToChannel(c, session_channel, b)) return -1;
    } else {
      bool cr = b.len && b.back() == '\r';
      pw.append(b.data(), b.size() - cr);
      if (cr && !WritePassword(c)) return ERRORv(-1, c->Name(), ": write");
    }
    return b.size();
  }

  bool WriteToChannel(Connection *c, Channel *chan, const StringPiece &b) {
    if (!WriteCipher(c, SSH::MSG_CHANNEL_DATA(chan->remote_id, b))) return ERRORv(false, c->Name(), ": write");
    chan->window_c -= (b.size() - 4);
    return true;
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

  void ClearPassword() { pw.assign(pw.size(), ' '); pw.clear(); }
  void LoadPassword(Connection *c) {
    if ((loaded_pw = load_password_cb && load_password_cb(&pw))) WritePassword(c);
    if (loaded_pw) ClearPassword();
  }

  void SetCredentialCB(SSHClient::LoadIdentityCB LI,
                       SSHClient::LoadPasswordCB LP) { load_identity_cb=move(LI); load_password_cb=move(LP); }

  int SetTerminalWindowSize(Connection *c, int w, int h) {
    term_width = w;
    term_height = h;
    if (!c || c->state != Connection::Connected || state <= FIRST_NEWKEYS) return 0;
    if (!WriteCipher(c, SSH::MSG_CHANNEL_REQUEST(session_channel->remote_id, "window-change", point(term_width, term_height),
                                                 point(term_width*8, term_height*12),
                                                 "", "", false))) return ERRORv(-1, c->Name(), ": write");
    return 0;
  }
};

Connection *SSHClient::Open(const string &hostport, const string &user, const string &term, bool compress, bool agent_fwd,
                            const SSHClient::ResponseCB &cb, Callback *detach, Callback *success) { 
  Connection *c = app->net->tcp_client->Connect(hostport, 22, detach);
  if (!c) return 0;
  c->handler = make_unique<SSHClientConnection>(cb, hostport, user, term, compress, agent_fwd, success ? *success : Callback());
  return c;
}

int  SSHClient::WriteChannelData     (Connection *c, const StringPiece &b)                 { return dynamic_cast<SSHClientConnection*>(c->handler.get())->WriteChannelData(c, b); }
int  SSHClient::SetTerminalWindowSize(Connection *c, int w, int h)                         { return dynamic_cast<SSHClientConnection*>(c->handler.get())->SetTerminalWindowSize(c, w, h); }
void SSHClient::SetCredentialCB      (Connection *c, LoadIdentityCB LI, LoadPasswordCB LP) { dynamic_cast<SSHClientConnection*>(c->handler.get())->SetCredentialCB(LI, LP); }

int SSH::BinaryPacketLength(const char *b, unsigned char *padding) {
  if (padding) *padding = *MakeUnsigned(b + 4);
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

void SSH::UpdateDigest(Crypto::Digest d, const StringPiece &s) {
  UpdateDigest(d, s.size());
  Crypto::DigestUpdate(d, s);
}

void SSH::UpdateDigest(Crypto::Digest d, int n) {
  char buf[4];
  Serializable::MutableStream(buf, 4).Htonl(n);
  Crypto::DigestUpdate(d, StringPiece(buf, 4));
}

void SSH::UpdateDigest(Crypto::Digest d, BigNum n) {
  string buf(4 + SSH::BigNumSize(n), 0);
  Serializable::MutableStream o(&buf[0], buf.size());
  SSH::WriteBigNum(n, &o);
  Crypto::DigestUpdate(d, buf);
}

int SSH::VerifyHostKey(const string &H_text, int hostkey_type, const StringPiece &key, const StringPiece &sig) {
  if (hostkey_type == SSH::Key::RSA) {
    string H_hash = Crypto::SHA1(H_text);
    LFL::RSAKey rsa_key = NewRSAPubKey();
    SSH::RSASignature rsa_sig;
    Serializable::ConstStream rsakey_stream(key.data(), key.size());
    Serializable::ConstStream rsasig_stream(sig.data(), sig.size());
    if (rsa_sig.In(&rsasig_stream)) { RSAKeyFree(rsa_key); return -3; }
    string rsa_sigbuf(rsa_sig.sig.data(), rsa_sig.sig.size());
    if (SSH::RSAKey(GetRSAKeyE(rsa_key), GetRSAKeyN(rsa_key)).In(&rsakey_stream)) { RSAKeyFree(rsa_key); return -2; }
    int verified = RSAVerify(H_hash, &rsa_sigbuf, rsa_key);
    RSAKeyFree(rsa_key);
    return verified;

  } else if (hostkey_type == SSH::Key::DSS) {
    string H_hash = Crypto::SHA1(H_text);
    DSAKey dsa_key = NewDSAPubKey();
    DSASig dsa_sig = NewDSASig();
    Serializable::ConstStream dsakey_stream(key.data(), key.size());
    Serializable::ConstStream dsasig_stream(sig.data(), sig.size());
    if (SSH::DSSKey(GetDSAKeyP(dsa_key), GetDSAKeyQ(dsa_key), GetDSAKeyG(dsa_key), GetDSAKeyK(dsa_key)).In(&dsakey_stream)) { DSAKeyFree(dsa_key); return -4; }
    if (SSH::DSSSignature(GetDSASigR(dsa_sig), GetDSASigS(dsa_sig)).In(&dsasig_stream)) { DSAKeyFree(dsa_key); DSASigFree(dsa_sig); return -5; }
    int verified = DSAVerify(H_hash, dsa_sig, dsa_key);
    DSAKeyFree(dsa_key);
    DSASigFree(dsa_sig);
    return verified;
    
  } else if (Key::EllipticCurveDSA(hostkey_type)) {
    ECDef curve_id;
    Crypto::DigestAlgo hash_id;
    switch (hostkey_type) {
      case SSH::Key::ECDSA_SHA2_NISTP256: curve_id=Crypto::EllipticCurve::NISTP256(); hash_id=Crypto::DigestAlgos::SHA256(); break;
      case SSH::Key::ECDSA_SHA2_NISTP384: curve_id=Crypto::EllipticCurve::NISTP384(); hash_id=Crypto::DigestAlgos::SHA384(); break;
      case SSH::Key::ECDSA_SHA2_NISTP521: curve_id=Crypto::EllipticCurve::NISTP521(); hash_id=Crypto::DigestAlgos::SHA512(); break;
      default:                            return ERRORv(-6, "ecdsa curve");
    }
    string H_hash = Crypto::ComputeDigest(hash_id, H_text);
    SSH::ECDSAKey key_msg;
    ECDSASig ecdsa_sig = NewECDSASig();
    Serializable::ConstStream ecdsakey_stream(key.data(), key.size());
    Serializable::ConstStream ecdsasig_stream(sig.data(), sig.size());
    if (key_msg.In(&ecdsakey_stream)) { ECDSASigFree(ecdsa_sig); return -7; }
    if (SSH::ECDSASignature(GetECDSASigR(ecdsa_sig), GetECDSASigS(ecdsa_sig)).In(&ecdsasig_stream)) { ECDSASigFree(ecdsa_sig); return -8; }

    ECPair ecdsa_keypair = Crypto::EllipticCurve::NewPair(curve_id, false);
    ECPoint ecdsa_key = NewECPoint(GetECPairGroup(ecdsa_keypair));
    ECPointSetData(GetECPairGroup(ecdsa_keypair), ecdsa_key, key_msg.q);
    if (!SetECPairPubKey(ecdsa_keypair, ecdsa_key)) { FreeECPair(ecdsa_keypair); ECDSASigFree(ecdsa_sig); return -9; }
    int verified = ECDSAVerify(H_hash, ecdsa_sig, ecdsa_keypair);
    FreeECPair(ecdsa_keypair);
    ECDSASigFree(ecdsa_sig);
    return verified;
  } else if (hostkey_type == SSH::Key::ED25519) {
    SSH::Ed25519Key key_msg;
    SSH::Ed25519Signature sig_msg;
    Serializable::ConstStream ed25519key_stream(key.data(), key.size());
    Serializable::ConstStream ed25519sig_stream(sig.data(), sig.size());
    if (key_msg.In(&ed25519key_stream)) return -10;
    if (sig_msg.In(&ed25519sig_stream)) return -11;
    return Ed25519Verify(H_text, sig_msg.sig, key_msg.key);

  } else return -12;
}

string SSH::ComputeExchangeHash(int kex_method, Crypto::DigestAlgo algo, const string &V_C, const string &V_S,
                                const string &KI_C, const string &KI_S, const StringPiece &k_s, BigNum K,
                                Crypto::DiffieHellman *dh, Crypto::EllipticCurveDiffieHellman *ecdh,
                                Crypto::X25519DiffieHellman *x25519dh) {
  string ret;
  unsigned char kex_c_padding = 0, kex_s_padding = 0;
  int kex_c_packet_len = 4 + SSH::BinaryPacketLength(KI_C.data(), &kex_c_padding);
  int kex_s_packet_len = 4 + SSH::BinaryPacketLength(KI_S.data(), &kex_s_padding);
  Crypto::Digest H = Crypto::DigestOpen(algo);
  UpdateDigest(H, V_C);
  UpdateDigest(H, V_S);
  UpdateDigest(H, StringPiece(KI_C.data() + 5, kex_c_packet_len - 5 - kex_c_padding));
  UpdateDigest(H, StringPiece(KI_S.data() + 5, kex_s_packet_len - 5 - kex_s_padding));
  UpdateDigest(H, k_s); 
  if (KEX::DiffieHellmanGroupExchange(kex_method)) {
    UpdateDigest(H, dh->gex_min);
    UpdateDigest(H, dh->gex_pref);
    UpdateDigest(H, dh->gex_max);
    UpdateDigest(H, dh->p);
    UpdateDigest(H, dh->g);
  }
  if (KEX::X25519DiffieHellman(kex_method)) {
    UpdateDigest(H, x25519dh->mypubkey);
    UpdateDigest(H, x25519dh->remotepubkey);
  } else if (KEX::EllipticCurveDiffieHellman(kex_method)) {
    UpdateDigest(H, ecdh->c_text);
    UpdateDigest(H, ecdh->s_text);
  } else {
    UpdateDigest(H, dh->e);
    UpdateDigest(H, dh->f);
  }
  UpdateDigest(H, K);
  ret = Crypto::DigestFinish(H);
  return ret;
}

string SSH::DeriveKey(Crypto::DigestAlgo algo, const string &session_id, const string &H_text, BigNum K, char ID, int bytes) {
  string ret;
  while (ret.size() < bytes) {
    Crypto::Digest key = Crypto::DigestOpen(algo);
    UpdateDigest(key, K);
    Crypto::DigestUpdate(key, H_text);
    if (!ret.size()) {
      Crypto::DigestUpdate(key, StringPiece(&ID, 1));
      Crypto::DigestUpdate(key, session_id);
    } else Crypto::DigestUpdate(key, ret);
    ret.append(Crypto::DigestFinish(key));
  }
  ret.resize(bytes);
  return ret;
}

string SSH::DeriveChallengeText(const StringPiece &session_id, const StringPiece &user_name, const StringPiece &service_name,
                                const StringPiece &method_name, const StringPiece &algo_name, const StringPiece &secret) {
  string out(2 + 4*6 + session_id.size() + user_name.size() + service_name.size() + method_name.size() +
             algo_name.size() + secret.size(), 0);
  Serializable::MutableStream o(&out[0], out.size());
  o.BString(session_id);
  o.Write8(static_cast<unsigned char>(MSG_USERAUTH_REQUEST::ID));
  o.BString(user_name);
  o.BString(service_name);
  o.BString(method_name);
  o.Write8(static_cast<unsigned char>(1));
  o.BString(algo_name);
  o.BString(secret);
  return out;
}

string SSH::DeriveChallenge(Crypto::DigestAlgo algo, const StringPiece &session_id, const StringPiece &user_name,
                            const StringPiece &service_name, const StringPiece &method_name, const StringPiece &algo_name, const StringPiece &secret) {
  Crypto::Digest digest = Crypto::DigestOpen(algo);
  UpdateDigest(digest, session_id);

  unsigned char c = MSG_USERAUTH_REQUEST::ID;
  Crypto::DigestUpdate(digest, StringPiece(MakeSigned(&c), 1));
  UpdateDigest(digest, user_name);
  UpdateDigest(digest, service_name);
  UpdateDigest(digest, method_name);

  c = 1;
  Crypto::DigestUpdate(digest, StringPiece(MakeSigned(&c), 1));
  UpdateDigest(digest, algo_name);
  UpdateDigest(digest, secret);
  return Crypto::DigestFinish(digest);
}

string SSH::MAC(Crypto::MACAlgo algo, int MAC_len, const StringPiece &m, int seq, const string &k, int prefix) {
  char buf[4];
  Serializable::MutableStream(buf, 4).Htonl(seq);
  string ret(MAC_len, 0);
  Crypto::MAC mac = Crypto::MACOpen(algo, k);
  Crypto::MACUpdate(mac, StringPiece(buf, 4));
  Crypto::MACUpdate(mac, m);
  int ret_len = Crypto::MACFinish(mac, &ret[0], ret.size());
  CHECK_EQ(ret.size(), ret_len);
  return prefix ? ret.substr(0, prefix) : ret;
}

bool SSH::Key        ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::KEX        ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::MAC        ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::Cipher     ::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }
bool SSH::Compression::PreferenceIntersect(const StringPiece &v, int *out, int po) { return (*out = Id(FirstMatchCSV(v, PreferenceCSV(po)))); }

string SSH::Key        ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) if (Supported(i)) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::KEX        ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) if (Supported(i)) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::Cipher     ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) if (Supported(i)) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::MAC        ::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) if (Supported(i)) StrAppendCSV(&v, Name(i)); }); return v; }
string SSH::Compression::PreferenceCSV(int o) { static string v; ONCE({ for (int i=1+o; i<=End; ++i) if (Supported(i)) StrAppendCSV(&v, Name(i)); }); return v; }

int SSH::Key        ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::KEX        ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::Cipher     ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::MAC        ::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }
int SSH::Compression::Id(const string &n) { static unordered_map<string, int> m; ONCE({ for (int i=1; i<=End; ++i) m[Name(i)] = i; }); return FindOrDefault(m, n, 0); }

bool SSH::Key        ::Supported(int x) { return true; }
bool SSH::KEX        ::Supported(int x) { return true; }
bool SSH::Cipher     ::Supported(int x) { return true; }
bool SSH::MAC        ::Supported(int x) { return true; }
bool SSH::Compression::Supported(int x) { return true; }

const char *SSH::Key::Name(int id) {
  switch(id) {
    case RSA:                 return "ssh-rsa";
    case DSS:                 return "ssh-dss";
    case ECDSA_SHA2_NISTP256: return "ecdsa-sha2-nistp256";
    case ECDSA_SHA2_NISTP384: return "ecdsa-sha2-nistp384";
    case ECDSA_SHA2_NISTP521: return "ecdsa-sha2-nistp521";
    case ED25519:             return "ssh-ed25519";
    default:                  return "";
  }
};

const char *SSH::KEX::Name(int id) {
  switch(id) {
    case ECDH_SHA2_X25519:   return "curve25519-sha256@libssh.org";
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

const char *SSH::Compression::Name(int id) {
  switch(id) {
    case OpenSSHZLib: return "zlib@openssh.com";
    case None:        return "none";
    default:          return "";
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

string SSH::Serializable::ToString(ZLibWriter *zwriter, std::mt19937 &g, int block_size, unsigned *sequence_number) const {
  if (sequence_number) (*sequence_number)++;
  string ret;
  CHECK(ToString(&ret, zwriter, g, block_size));
  return ret;
}

bool SSH::Serializable::ToString(string *out, ZLibWriter *zwriter, std::mt19937 &g, int block_size) const {
  if (zwriter) {
    string buf(Size() + 1, 0);
    MutableStream os(&buf[0], buf.size());
    unsigned char type = Id;
    os.Write8(type);
    Out(&os);
    zwriter->out.clear();
    if (!zwriter->Add(buf, true)) return ERRORv(false, "compress failed");
  }
  out->resize(NextMultipleOfN(4 + SSH::BinaryPacketHeaderSize +
                              (zwriter ? zwriter->out.size() : Size() + 1), max(8, block_size)));
  return ToString(&(*out)[0], out->size(), zwriter ? StringPiece(zwriter->out) : StringPiece(), g);
}

bool SSH::Serializable::ToString(char *buf, int len, const StringPiece &payload, std::mt19937 &g) const {
  unsigned char padding = len - SSH::BinaryPacketHeaderSize - (payload.len != 0 ? payload.len : 1 + Size());
  MutableStream os(buf, len);
  os.Htonl(len - 4);
  os.Write8(padding);
  if (payload.len) os.String(payload);
  else { unsigned char type = Id; os.Write8(type); Out(&os); }
  os.String(RandBytes(padding, g));
  return !os.error;
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

void SSH::DSSKey::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  WriteBigNum(p, o); 
  WriteBigNum(q, o); 
  WriteBigNum(g, o); 
  WriteBigNum(y, o); 
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

void SSH::DSSSignature::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  string blob(40, 0);
  CHECK_EQ(20, BigNumDataSize(r));
  CHECK_EQ(20, BigNumDataSize(s));
  BigNumGetData(r, &blob[0]);
  BigNumGetData(s, &blob[20]);
  o->BString(blob);
}

int SSH::RSAKey::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  if (format_id.str() != Key::Name(Key::RSA)) { i->error = true; return -1; }
  e = ReadBigNum(e, i); 
  n = ReadBigNum(n, i); 
  return i->Result();
}

void SSH::RSAKey::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  WriteBigNum(e, o); 
  WriteBigNum(n, o); 
}

int SSH::RSASignature::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  i->ReadString(&sig);
  if (format_id.str() != "ssh-rsa") { i->error = true; return -1; }
  return i->Result();
}

void SSH::RSASignature::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  o->BString(sig);
}

int SSH::ECDSAKey::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  if (!PrefixMatch(format_id.str(), "ecdsa-sha2-")) { i->error = true; return -1; }
  i->ReadString(&curve_id);
  i->ReadString(&q);
  return i->Result();
}

void SSH::ECDSAKey::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  o->BString(curve_id);
  o->BString(q);
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

void SSH::ECDSASignature::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  string blob(4*2 + BigNumSize(r) + BigNumSize(s), 0);
  Serializable::MutableStream blobo(&blob[0], blob.size());
  WriteBigNum(r, &blobo);
  WriteBigNum(s, &blobo);
  o->BString(blob);
}

int SSH::Ed25519Key::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  i->ReadString(&key);
  if (format_id.str() != "ssh-ed25519") { i->error = true; return -1; }
  return i->Result();
}

void SSH::Ed25519Key::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  o->BString(key);
}

int SSH::Ed25519Signature::In(const Serializable::Stream *i) {
  i->ReadString(&format_id);
  i->ReadString(&sig);
  if (format_id.str() != "ssh-ed25519") { i->error = true; return -1; }
  return i->Result();
}

void SSH::Ed25519Signature::Out(Serializable::Stream *o) const {
  o->BString(format_id);
  o->BString(sig);
}

string SSH::AgentSerializable::ToString() const { string ret; ToString(&ret); return ret; }

void SSH::AgentSerializable::ToString(string *out) const {
  out->resize(5 + Size());
  return ToString(&(*out)[0], out->size());
}

void SSH::AgentSerializable::ToString(char *buf, int len) const {
  unsigned char type = Id;
  MutableStream os(buf, len);
  os.Htonl(len - 4);
  os.Write8(type);
  Out(&os);
}

void SSH::AGENT_IDENTITIES_ANSWER::Out(Serializable::Stream *o) const {
  o->Htonl(int(keys.size()));
  for (auto &i : keys) { o->BString(i.blob); o->BString(i.comment); }
}

}; // namespace LFL
