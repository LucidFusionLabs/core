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

struct SSHClient {
  typedef function<void(Connection*, const StringPiece&)> ResponseCB;
  typedef function<bool(const string&, const string&,       string*)> LoadPasswordCB;
  typedef function<void(const string&, const string&, const string&)> SavePasswordCB;
  static Connection *Open(const string &hostport, const ResponseCB &cb, Callback *detach=0);

  static void SetUser(Connection *c, const string &user);
  static void SetPasswordCB(Connection *c, const LoadPasswordCB&, const SavePasswordCB&);
  static int SetTerminalWindowSize(Connection *c, int w, int h);
  static int WriteChannelData(Connection *c, const StringPiece &b);
};

struct SSH {
  static const int BinaryPacketHeaderSize = 6;
  static int BinaryPacketLength(const char *b, unsigned char *padding, unsigned char *id);
  static int BigNumSize(const BigNum n);
  static BigNum ReadBigNum(BigNum n, const Serializable::Stream *i);
  static void WriteBigNum(const BigNum n, Serializable::Stream *o);
  static void UpdateDigest(Crypto::Digest d, const StringPiece &s);
  static void UpdateDigest(Crypto::Digest d, int n);
  static void UpdateDigest(Crypto::Digest d, BigNum n);
  static string ComputeExchangeHash(int kex_method, Crypto::DigestAlgo algo, const string &V_C, const string &V_S,
                                    const string &KI_C, const string &KI_S, const StringPiece &k_s, BigNum K,
                                    Crypto::DiffieHellman*, Crypto::EllipticCurveDiffieHellman*);
  static int VerifyHostKey(const string &H_text, int hostkey_type, const StringPiece &key, const StringPiece &sig);
  static string DeriveKey(Crypto::DigestAlgo algo, const string &session_id, const string &H_text, BigNum K, char ID, int bytes);
  static string MAC(Crypto::MACAlgo algo, int MAC_len, const StringPiece &m, int seq, const string &k, int prefix=0);

  struct Key {
    enum { ECDSA_SHA2_NISTP256=1, RSA=2, DSS=3, End=3 };
    static int Id(const string &n);
    static const char *Name(int id);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct KEX {
    enum { ECDH_SHA2_NISTP256=1, ECDH_SHA2_NISTP384=2, ECDH_SHA2_NISTP521=3, DHGEX_SHA256=4, DHGEX_SHA1=5, DH14_SHA1=6, DH1_SHA1=7, End=7 };
    static int Id(const string &n);
    static const char *Name(int id);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
    static bool EllipticCurveDiffieHellman(int id) { return id==ECDH_SHA2_NISTP256 || id==ECDH_SHA2_NISTP384 || id==ECDH_SHA2_NISTP521; }
    static bool DiffieHellmanGroupExchange(int id) { return id==DHGEX_SHA256 || id==DHGEX_SHA1; }
    static bool DiffieHellman(int id) { return id==DHGEX_SHA256 || id==DHGEX_SHA1 || id==DH14_SHA1 || id==DH1_SHA1; }
  };

  struct Cipher {
    enum { AES128_CTR=1, AES128_CBC=2, TripDES_CBC=3, Blowfish_CBC=4, RC4=5, End=5 };
    static int Id(const string &n);
    static const char *Name(int id);
    static Crypto::CipherAlgo Algo(int id, int *blocksize=0);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct MAC {
    enum { MD5=1, SHA1=2, SHA1_96=3, MD5_96=4, SHA256=5, SHA256_96=6, SHA512=7, SHA512_96=8, End=8 };
    static int Id(const string &n);
    static const char *Name(int id);
    static Crypto::MACAlgo Algo(int id, int *prefix_bytes=0);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, int *out, int start_after=0);
  };

  struct Serializable : public LFL::Serializable {
    Serializable(int type) : LFL::Serializable(type) {}
    string ToString(std::mt19937&, int block_size, unsigned *sequence_number) const;
    void ToString(string *out, std::mt19937&, int block_size) const;
    void ToString(char *buf, int len, std::mt19937&) const;
  };

  struct MSG_DISCONNECT : public Serializable {
    static const int ID = 1;
    int reason_code=0;
    StringPiece description, language;
    MSG_DISCONNECT() : Serializable(ID) {}

    int HeaderSize() const { return 4 + 2*4; }
    int Size() const { return HeaderSize() + description.size() + language.size(); }
    int In(const Serializable::Stream *i) { i->Ntohl(&reason_code); i->ReadString(&description); i->ReadString(&language); return i->Result(); }
    void Out(Serializable::Stream *o) const {}
  };

  struct MSG_DEBUG : public Serializable {
    static const int ID = 4;
    unsigned char always_display=0;
    StringPiece message, language;
    MSG_DEBUG() : Serializable(ID) {}

    int HeaderSize() const { return 1 + 2*4; }
    int Size() const { return HeaderSize() + message.size() + language.size(); }
    int In(const Serializable::Stream *i) { i->Read8(&always_display); i->ReadString(&message); i->ReadString(&language); return i->Result(); }
    void Out(Serializable::Stream *o) const {}
  };

  struct MSG_SERVICE_REQUEST : public Serializable {
    static const int ID = 5;
    StringPiece service_name;
    MSG_SERVICE_REQUEST(const StringPiece &SN) : Serializable(ID), service_name(SN) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + service_name.size(); }
    int In(const Serializable::Stream *i) { i->ReadString(&service_name); return i->Result(); }
    void Out(Serializable::Stream *o) const { o->BString(service_name); }
  };

  struct MSG_SERVICE_ACCEPT : public Serializable {
    static const int ID = 6;
    StringPiece service_name;
    MSG_SERVICE_ACCEPT(const StringPiece &SN) : Serializable(ID), service_name(SN) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + service_name.size(); }
    int In(const Serializable::Stream *i) { i->ReadString(&service_name); return i->Result(); }
    void Out(Serializable::Stream *o) const { o->BString(service_name); }
  };

  struct MSG_KEXINIT : public Serializable {
    static const int ID = 20;
    StringPiece cookie, kex_algorithms, server_host_key_algorithms, encryption_algorithms_client_to_server,
                encryption_algorithms_server_to_client, mac_algorithms_client_to_server, mac_algorithms_server_to_client,
                compression_algorithms_client_to_server, compression_algorithms_server_to_client, languages_client_to_server,
                languages_server_to_client;
    unsigned char first_kex_packet_follows=0;
    MSG_KEXINIT() : Serializable(ID) {}
    MSG_KEXINIT(const StringPiece &C, const StringPiece &KEXA, const StringPiece &KA, const StringPiece &EC, const StringPiece &ES, const StringPiece &MC,
                const StringPiece &MS, const StringPiece &CC, const StringPiece &CS, const StringPiece &LC, const StringPiece &LS, bool guess) :
      Serializable(ID), cookie(C), kex_algorithms(KEXA), server_host_key_algorithms(KA), encryption_algorithms_client_to_server(EC),
      encryption_algorithms_server_to_client(ES), mac_algorithms_client_to_server(MC), mac_algorithms_server_to_client(MS),
      compression_algorithms_client_to_server(CC), compression_algorithms_server_to_client(CS), languages_client_to_server(LC),
      languages_server_to_client(LS), first_kex_packet_follows(guess) {}

    int HeaderSize() const { return 21 + 10*4; }
    int Size() const;
    string DebugString() const;

    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i);
  };

  struct MSG_NEWKEYS : public Serializable {
    static const int ID = 21;
    MSG_NEWKEYS() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { return 0; }
  };

  struct MSG_KEXDH_INIT : public Serializable {
    static const int ID = 30;
    BigNum e;
    MSG_KEXDH_INIT(BigNum E=0) : Serializable(ID), e(E) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + BigNumSize(e); }
    void Out(Serializable::Stream *o) const { WriteBigNum(e, o); }
    int In(const Serializable::Stream *i) { return 0; }
  };

  struct MSG_KEXDH_REPLY : public Serializable {
    static const int ID = 31;
    StringPiece k_s, h_sig;
    BigNum f;
    MSG_KEXDH_REPLY(BigNum F=0) : Serializable(ID), f(F) {}

    int HeaderSize() const { return 4*3; }
    int Size() const { return HeaderSize() + BigNumSize(f) + k_s.size() + h_sig.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { i->ReadString(&k_s); f=ReadBigNum(f,i); i->ReadString(&h_sig); return i->Result(); }
  };

  struct MSG_KEX_DH_GEX_REQUEST : public Serializable {
    static const int ID = 34;
    int min_n, max_n, n;
    MSG_KEX_DH_GEX_REQUEST(int Min, int Max, int N) : Serializable(ID), min_n(Min), max_n(Max), n(N) {}

    int HeaderSize() const { return 4*3; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl( min_n); o->Htonl( n); o->Htonl( max_n); }
    int In(const Serializable::Stream *i)   { i->Ntohl(&min_n); i->Ntohl(&n); i->Ntohl(&max_n); return i->Result(); }
  };

  struct MSG_KEX_DH_GEX_GROUP : public Serializable {
    static const int ID = 31;
    BigNum p, g;
    MSG_KEX_DH_GEX_GROUP(BigNum P, BigNum G) : Serializable(ID), p(P), g(G) {}

    int HeaderSize() const { return 4*2; }
    int Size() const { return HeaderSize() + BigNumSize(p) + BigNumSize(g); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { p=ReadBigNum(p,i); g=ReadBigNum(g,i); return i->Result(); }
  };

  struct MSG_KEX_DH_GEX_INIT : public MSG_KEXDH_INIT {
    static const int ID = 32;
    MSG_KEX_DH_GEX_INIT(BigNum E=0) : MSG_KEXDH_INIT(E) { Id=ID; }
  };

  struct MSG_KEX_DH_GEX_REPLY : public MSG_KEXDH_REPLY {
    static const int ID = 33;
    MSG_KEX_DH_GEX_REPLY(BigNum F=0) : MSG_KEXDH_REPLY(F) { Id=ID; }
  };

  struct MSG_KEX_ECDH_INIT : public Serializable {
    static const int ID = 30;
    StringPiece q_c;
    MSG_KEX_ECDH_INIT(const StringPiece &Q_C) : Serializable(ID), q_c(Q_C) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + q_c.size(); }
    void Out(Serializable::Stream *o) const { o->BString(q_c); }
    int In(const Serializable::Stream *i) { i->ReadString(&q_c); return i->Result(); }
  };

  struct MSG_KEX_ECDH_REPLY : public Serializable {
    static const int ID = 31;
    StringPiece k_s, q_s, h_sig;
    MSG_KEX_ECDH_REPLY() : Serializable(ID) {}

    int HeaderSize() const { return 4*3; }
    int Size() const { return HeaderSize() + k_s.size() + q_s.size() + h_sig.size(); }
    void Out(Serializable::Stream *o) const { o->BString(k_s); o->BString(q_s); o->BString(h_sig); }
    int In(const Serializable::Stream *i) { i->ReadString(&k_s); i->ReadString(&q_s); i->ReadString(&h_sig); return i->Result(); }
  };

  struct MSG_USERAUTH_REQUEST : public Serializable {
    static const int ID = 50;
    StringPiece user_name, service_name, method_name, algo_name, secret, sig;
    MSG_USERAUTH_REQUEST(const StringPiece &UN, const StringPiece &SN, const StringPiece &MN, const StringPiece &AN, const StringPiece &P, const StringPiece &S)
      : Serializable(ID), user_name(UN), service_name(SN), method_name(MN), algo_name(AN), secret(P), sig(S) {}

    int HeaderSize() const { return 4*3; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i) { return 0; }
  };

  struct MSG_USERAUTH_FAILURE : public Serializable {
    static const int ID = 51;
    StringPiece auth_left;
    unsigned char partial_success=0;
    MSG_USERAUTH_FAILURE() : Serializable(ID) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + auth_left.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { i->ReadString(&auth_left); i->Read8(&partial_success); return i->Result(); }
  };

  struct MSG_USERAUTH_SUCCESS : public Serializable {
    static const int ID = 52;
    MSG_USERAUTH_SUCCESS() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { return i->Result(); }
  };

  struct MSG_USERAUTH_INFO_REQUEST : public Serializable {
    static const int ID = 60;
    StringPiece name, instruction, language;
    struct Prompt { StringPiece text; unsigned char echo=0; };
    vector<Prompt> prompt;
    MSG_USERAUTH_INFO_REQUEST() : Serializable(ID) {}

    int HeaderSize() const { return 4*3; }
    int Size() const;
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct MSG_USERAUTH_INFO_RESPONSE : public Serializable {
    static const int ID = 61;
    vector<StringPiece> response;
    MSG_USERAUTH_INFO_RESPONSE() : Serializable(ID) {}
    MSG_USERAUTH_INFO_RESPONSE(const vector<StringPiece> &s) : Serializable(ID), response(s) {}

    int HeaderSize() const { return 4; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i) { return i->Result(); }
  };

  struct MSG_CHANNEL_OPEN : public Serializable {
    static const int ID = 90;
    StringPiece channel_type;
    int sender_channel, initial_win_size, maximum_packet_size;
    MSG_CHANNEL_OPEN(const StringPiece &CT, int SC, int IWS, int MPS) : Serializable(ID), channel_type(CT), sender_channel(SC), initial_win_size(IWS), maximum_packet_size(MPS) {}

    int HeaderSize() const { return 4*4; }
    int Size() const { return HeaderSize() + channel_type.size(); }
    void Out(Serializable::Stream *o) const { o->BString(channel_type); o->Htonl(sender_channel); o->Htonl(initial_win_size); o->Htonl(maximum_packet_size); }
    int In(const Serializable::Stream *i) { return i->Result(); }
  };

  struct MSG_CHANNEL_OPEN_CONFIRMATION : public Serializable {
    static const int ID = 91;
    int recipient_channel, sender_channel, initial_win_size, maximum_packet_size;
    MSG_CHANNEL_OPEN_CONFIRMATION() : Serializable(ID) {}

    int HeaderSize() const { return 4*4; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { i->Ntohl(&recipient_channel); i->Ntohl(&sender_channel); i->Ntohl(&initial_win_size); i->Ntohl(&maximum_packet_size); return i->Result(); }
  };

  struct MSG_CHANNEL_WINDOW_ADJUST : public Serializable {
    static const int ID = 93;
    int recipient_channel, bytes_to_add;
    MSG_CHANNEL_WINDOW_ADJUST(int C=0, int bytes=0) : Serializable(ID), recipient_channel(C), bytes_to_add(bytes) {}

    int HeaderSize() const { return 4*2; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl(recipient_channel); o->Htonl(bytes_to_add); }
    int In(const Serializable::Stream *i) { i->Ntohl(&recipient_channel); i->Ntohl(&bytes_to_add); return i->Result(); }
  };

  struct MSG_CHANNEL_DATA : public Serializable {
    static const int ID = 94;
    int recipient_channel;
    StringPiece data;
    MSG_CHANNEL_DATA() : Serializable(ID) {}
    MSG_CHANNEL_DATA(int RC, const StringPiece &D) : Serializable(ID), recipient_channel(RC), data(D) {}

    int HeaderSize() const { return 4*2; }
    int Size() const { return HeaderSize() + data.size(); }
    void Out(Serializable::Stream *o) const { o->Htonl(recipient_channel); o->BString(data); }
    int In(const Serializable::Stream *i) { i->Ntohl(&recipient_channel); i->ReadString(&data); return i->Result(); }
  };

  struct MSG_CHANNEL_EOF : public Serializable {
    static const int ID = 96;
    int recipient_channel;
    MSG_CHANNEL_EOF(int C=0) : Serializable(ID), recipient_channel(C) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl(recipient_channel); }
    int In(const Serializable::Stream *i) { i->Ntohl(&recipient_channel); return i->Result(); }
  };

  struct MSG_CHANNEL_CLOSE : public Serializable {
    static const int ID = 97;
    int recipient_channel;
    MSG_CHANNEL_CLOSE(int C=0) : Serializable(ID), recipient_channel(C) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl(recipient_channel); }
    int In(const Serializable::Stream *i) { i->Ntohl(&recipient_channel); return i->Result(); }
  };

  struct MSG_CHANNEL_REQUEST : public Serializable {
    static const int ID = 98;
    int recipient_channel=0, width=0, height=0, pixel_width=0, pixel_height=0;
    StringPiece request_type, term, term_mode;
    unsigned char want_reply=0;
    MSG_CHANNEL_REQUEST() : Serializable(ID) {}
    MSG_CHANNEL_REQUEST(int RC, const StringPiece &RT, const StringPiece &V, bool WR) : Serializable(ID), recipient_channel(RC), request_type(RT), term(V), want_reply(WR) {}
    MSG_CHANNEL_REQUEST(int RC, const StringPiece &RT, const point &D, const point &PD, const StringPiece &T, const StringPiece &TM, bool WR) : Serializable(ID),
      recipient_channel(RC), width(D.x), height(D.y), pixel_width(PD.x), pixel_height(PD.y), request_type(RT), term(T), term_mode(TM), want_reply(WR) {}

    int HeaderSize() const { return 4*2 + 1; }
    int Size() const;
    void Out(Serializable::Stream *o) const;
    int In(const Serializable::Stream *i) { return i->Result(); }
  };

  struct MSG_CHANNEL_SUCCESS : public Serializable {
    static const int ID = 99;
    MSG_CHANNEL_SUCCESS() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { return 0; }
  };

  struct MSG_CHANNEL_FAILURE : public Serializable {
    static const int ID = 100;
    MSG_CHANNEL_FAILURE() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) { return 0; }
  };

  struct DSSKey {
    StringPiece format_id;
    BigNum p, q, g, y;
    DSSKey(BigNum P, BigNum Q, BigNum G, BigNum Y) : p(P), q(Q), g(G), y(Y) {}

    int HeaderSize() const { return 4*4; }
    int Size() const { return HeaderSize() + format_id.size() + BigNumSize(p) + BigNumSize(q) + BigNumSize(g) + BigNumSize(y); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct DSSSignature {
    StringPiece format_id;
    BigNum r, s;
    DSSSignature(BigNum R, BigNum S) : r(R), s(S) {}

    int HeaderSize() const { return 4*2 + 7 + 20*2; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct RSAKey {
    StringPiece format_id;
    BigNum e, n;
    RSAKey(BigNum E, BigNum N) : e(E), n(N) {}

    int HeaderSize() const { return 2*4; }
    int Size() const { return HeaderSize() + format_id.size() + BigNumSize(e) + BigNumSize(n); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct RSASignature {
    StringPiece format_id, sig;

    int HeaderSize() const { return 4*2 + 7; }
    int Size() const { return HeaderSize() + sig.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct ECDSAKey {
    StringPiece format_id, curve_id, q;

    int HeaderSize() const { return 3*4; }
    int Size() const { return HeaderSize() + format_id.size() + curve_id.size() + q.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };

  struct ECDSASignature {
    StringPiece format_id;
    BigNum r, s;
    ECDSASignature(BigNum R, BigNum S) : r(R), s(S) {}

    int HeaderSize() const { return 4*4; }
    int Size() const { return HeaderSize() + format_id.size() + BigNumSize(r) + BigNumSize(s); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i);
  };
};

}; // namespace LFL
#endif // LFL_CORE_APP_NET_SSH_H__
