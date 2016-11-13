/*
 * $Id: network.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef LFL_CORE_APP_NET_SSH_H__
#define LFL_CORE_APP_NET_SSH_H__
namespace LFL {
namespace SSH {
  struct Key {
    enum { ED25519=1, ECDSA_SHA2_NISTP256=2, ECDSA_SHA2_NISTP384=3, ECDSA_SHA2_NISTP521=4, RSA=5, DSS=6, End=6 };
    static int Id(const string &n);
    static const char *Name(int id);
    static bool Supported(int);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
    static bool EllipticCurveDSA(int id) { return id==ECDSA_SHA2_NISTP256 || id==ECDSA_SHA2_NISTP384 || id==ECDSA_SHA2_NISTP521; }
  };

  struct KEX {
    enum { ECDH_SHA2_X25519=1, ECDH_SHA2_NISTP256=2, ECDH_SHA2_NISTP384=3, ECDH_SHA2_NISTP521=4,
      DHGEX_SHA256=5, DHGEX_SHA1=6, DH14_SHA1=7, DH1_SHA1=8, End=8 };
    static int Id(const string &n);
    static const char *Name(int id);
    static bool Supported(int);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
    static bool X25519DiffieHellman(int id) { return id==ECDH_SHA2_X25519; }
    static bool EllipticCurveDiffieHellman(int id) { return id==ECDH_SHA2_NISTP256 || id==ECDH_SHA2_NISTP384 || id==ECDH_SHA2_NISTP521; }
    static bool DiffieHellmanGroupExchange(int id) { return id==DHGEX_SHA256 || id==DHGEX_SHA1; }
    static bool DiffieHellman(int id) { return id==DHGEX_SHA256 || id==DHGEX_SHA1 || id==DH14_SHA1 || id==DH1_SHA1; }
  };

  struct Cipher {
    enum { AES128_CTR=1, AES128_CBC=2, TripDES_CBC=3, Blowfish_CBC=4, RC4=5, End=5 };
    static int Id(const string &n);
    static const char *Name(int id);
    static bool Supported(int);
    static Crypto::CipherAlgo Algo(int id, int *blocksize=0);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct MAC {
    enum { MD5=1, SHA1=2, SHA1_96=3, MD5_96=4, SHA256=5, SHA256_96=6, SHA512=7, SHA512_96=8, End=8 };
    static int Id(const string &n);
    static const char *Name(int id);
    static bool Supported(int);
    static Crypto::MACAlgo Algo(int id, int *prefix_bytes=0);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct Compression {
    enum { OpenSSHZLib=1, None=2, End=2 };
    static int Id(const string &n);
    static const char *Name(int id);
    static bool Supported(int);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct DSSKey : public LFL::Serializable {
    StringPiece format_id;
    BigNum p, q, g, y;
    DSSKey(BigNum P, BigNum Q, BigNum G, BigNum Y) : Serializable(0), format_id("ssh-dss"), p(P), q(Q), g(G), y(Y) {}

    int HeaderSize() const { return 5*4; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct DSSSignature : public LFL::Serializable {
    StringPiece format_id;
    BigNum r, s;
    DSSSignature(BigNum R, BigNum S) : Serializable(0), r(R), s(S), format_id("ssh-dss") {}

    int HeaderSize() const { return 4*2 + 7 + 20*2; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct RSAKey : public LFL::Serializable {
    StringPiece format_id;
    BigNum e, n;
    RSAKey(BigNum E, BigNum N) : Serializable(0), format_id("ssh-rsa"), e(E), n(N) {}

    int HeaderSize() const { return 3*4; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct RSASignature : public LFL::Serializable {
    StringPiece format_id, sig;
    RSASignature(const StringPiece &s=StringPiece()) : Serializable(0), format_id("ssh-rsa"), sig(s) {}

    int HeaderSize() const { return 4*2 + 7; }
    int Size() const { return HeaderSize() + sig.size(); }
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct ECDSAKey : public LFL::Serializable {
    StringPiece format_id, curve_id, q;
    ECDSAKey(const StringPiece &F=StringPiece(), const StringPiece &C=StringPiece(),
             const StringPiece &Q=StringPiece()) : Serializable(0), format_id(F), curve_id(C), q(Q) {}

    int HeaderSize() const { return 3*4; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct ECDSASignature : public LFL::Serializable {
    StringPiece format_id;
    BigNum r, s;
    ECDSASignature(BigNum R, BigNum S) : Serializable(0), r(R), s(S) {}
    ECDSASignature(const StringPiece &F, BigNum R, BigNum S) : Serializable(0), format_id(F), r(R), s(S) {}

    int HeaderSize() const { return 4*4; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct Ed25519Key : public LFL::Serializable {
    StringPiece format_id, key;
    Ed25519Key(const StringPiece &k=StringPiece()) : Serializable(0), format_id("ssh-ed25519"), key(k) {}

    int HeaderSize() const { return 4*2 + 11; }
    int Size() const { return HeaderSize() + key.size(); }
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct Ed25519Signature : public LFL::Serializable {
    StringPiece format_id, sig;
    Ed25519Signature(const StringPiece &s=StringPiece()) : Serializable(0), format_id("ssh-ed25519"), sig(s) {}

    int HeaderSize() const { return 4*2 + 11; }
    int Size() const { return HeaderSize() + sig.size(); }
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };
}; // namespace SSH

struct SSHClient {
  struct Identity {
    RSAKey      rsa;
    DSAKey      dsa;
    ECPair      ec;
    Ed25519Pair ed25519;
    virtual ~Identity() {
      if (rsa) RSAKeyFree(rsa);
      if (dsa) DSAKeyFree(dsa);
      if (ec)  FreeECPair(ec);
    }
  };

  struct Params {
    struct Forward { int port; string target_host; int target_port; };
    string hostport, user, termvar, startup_command;
    bool compress, agent_forwarding, close_on_disconnect, background_services;
    vector<Forward> forward_local, forward_remote;
  };

  struct Channel {
    typedef function<int(Connection*, Channel*, const StringPiece&)> CB;
    int local_id, remote_id, window_c=0, window_s=0;
    bool opened=1, agent_channel=0, sent_eof=0, sent_close=0;
    string buf;
    CB cb;
    Channel(int L=0, int R=0) : local_id(L), remote_id(R) {}
  };

  typedef function<bool(int, const StringPiece&)> FingerprintCB;
  typedef function<void(Connection*, const StringPiece&)> ResponseCB;
  typedef function<void(shared_ptr<Identity>)> IdentityCB;
  typedef function<bool(shared_ptr<Identity>*)> LoadIdentityCB;
  typedef function<bool(string*)> LoadPasswordCB;
  typedef function<void(Channel*, const string&, int, const string&, int)> RemoteForwardCB;

  struct Handler : public Connection::Handler {
    enum { INIT=0, FIRST_KEXINIT=1, FIRST_KEXREPLY=2, FIRST_NEWKEYS=3, KEXINIT=4, KEXREPLY=5, NEWKEYS=6 };
    Params params;
    ResponseCB cb;
    FingerprintCB host_fingerprint_cb;
    LoadIdentityCB load_identity_cb;
    LoadPasswordCB load_password_cb;
    RemoteForwardCB remote_forward_cb;
    Callback success_cb;
    shared_ptr<Identity> identity;
    string V_C, V_S, KEXINIT_C, KEXINIT_S, H_text, session_id, integrity_c2s, integrity_s2c, decrypt_buf, pw;
    int state=0, packet_len=0, packet_MAC_len=0, MAC_len_c=0, MAC_len_s=0, encrypt_block_size=0, decrypt_block_size=0, server_version=0;
    unsigned sequence_number_c2s=0, sequence_number_s2c=0, password_prompts=0, userauth_fail=0;
    bool guessed_c=0, guessed_s=0, guessed_right_c=0, guessed_right_s=0, accepted_hostkey=0, loaded_pw=0, wrote_pw=0;
    unsigned char padding=0, packet_id=0;
    Handler(Params p, ResponseCB CB, Callback s) : params(move(p)), cb(move(CB)), success_cb(move(s)), V_C("SSH-2.0-LFL_1.0") {}
    virtual ~Handler() {}
  };

  static Connection *Open(Params params, const ResponseCB &cb, Connection::CB *detach=0, Callback *success=0);
  static Connection *CreateSSHHandler(Connection *c, Params p, const ResponseCB &cb, Callback *success=0);
  static void SetCredentialCB(Connection *c, FingerprintCB, LoadIdentityCB, LoadPasswordCB);
  static void SetRemoteForwardCB(Connection *c, RemoteForwardCB F);
  static bool ParsePortForward(const string &text, vector<Params::Forward> *out);
  static int SetTerminalWindowSize(Connection *c, int w, int h);
  static bool AcceptHostKeyAndBeginAuthRequest(Connection *c);
  static int WriteChannelData(Connection *c, const StringPiece &b);
  static bool WritePassword(Connection *c, const StringPiece &b);
  static bool WriteToChannel(Connection *c, Channel *chan, const StringPiece &b);
  static Channel *OpenTCPChannel(Connection *c, const StringPiece &sh, int sp,
                                 const StringPiece &dh, int dp, Channel::CB cb);
  static bool CloseChannel(Connection *c, Channel *chan);
};

}; // namespace LFL
#endif // LFL_CORE_APP_NET_SSH_H__
