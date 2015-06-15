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

#ifndef __LFL_LFAPP_WIRE_H__
#define __LFL_LFAPP_WIRE_H__
namespace LFL {

struct Protocol { 
  enum { TCP=1, UDP=2, UNIX=3, GPLUS=4 }; int p;
  static const char *Name(int p);
};

struct Serializable {
  struct Stream {
    char *buf;
    int size;
    mutable int offset=0;
    mutable bool error=0;
    Stream(char *B, int S) : buf(B), size(S) {}

    virtual unsigned char  *N8()            = 0;
    virtual unsigned short *N16()           = 0;
    virtual unsigned       *N32()           = 0;
    virtual char           *Get(int size=0) = 0;
    virtual char           *End()           = 0;

    int Len() const { return size; }
    int Pos() const { return offset; };
    int Remaining() const { return size - offset; }
    int Result() const { return error ? -1 : 0; }
    const char *Start() const { return buf; }
    const char *Advance(int n=0) const { return Get(n); }
    const char           *End() const { return buf + size; }
    const unsigned char  *N8()  const { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
    const unsigned short *N16() const { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
    const unsigned       *N32() const { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
    const char           *Get(int len=0) const { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }

    void String  (const StringPiece &in) { char *v = (char*)Get(in.size());   if (v) { memcpy(v,   in.data(), in.size()); } }
    void BString (const StringPiece &in) { char *v = (char*)Get(in.size()+4); if (v) { memcpy(v+4, in.data(), in.size()); *((int*)v) = htonl(in.size()); } }
    void NTString(const StringPiece &in) { char *v = (char*)Get(in.size()+1); if (v) { memcpy(v,   in.data(), in.size()); v[in.size()]=0; } }

    void Write8 (const unsigned char  &in) { unsigned char  *v =                 N8();  if (v) *v = in; }
    void Write8 (const          char  &in) {          char  *v = (char*)         N8();  if (v) *v = in; }
    void Write16(const unsigned short &in) { unsigned short *v =                 N16(); if (v) *v = in; }
    void Write16(const          short &in) {          short *v = (short*)        N16(); if (v) *v = in; }
    void Write32(const unsigned int   &in) { unsigned int   *v =                 N32(); if (v) *v = in; }
    void Write32(const          int   &in) {          int   *v = (int*)          N32(); if (v) *v = in; }
    void Write32(const unsigned long  &in) { unsigned long  *v = (unsigned long*)N32(); if (v) *v = in; }
    void Write32(const          long  &in) {          long  *v = (long*)         N32(); if (v) *v = in; }

    void Ntohs(const unsigned short &in) { unsigned short *v =         N16(); if (v) *v = ntohs(in); }
    void Htons(const unsigned short &in) { unsigned short *v =         N16(); if (v) *v = htons(in); }
    void Ntohs(const          short &in) {          short *v = (short*)N16(); if (v) *v = ntohs(in); }
    void Htons(const          short &in) {          short *v = (short*)N16(); if (v) *v = htons(in); }
    void Ntohl(const unsigned int   &in) { unsigned int   *v =         N32(); if (v) *v = ntohl(in); }
    void Htonl(const unsigned int   &in) { unsigned int   *v =         N32(); if (v) *v = htonl(in); }
    void Ntohl(const          int   &in) {          int   *v = (int*)  N32(); if (v) *v = ntohl(in); }
    void Htonl(const          int   &in) {          int   *v = (int*)  N32(); if (v) *v = htonl(in); }

    void Htons(unsigned short *out) const { const unsigned short *v =         N16(); *out = v ? htons(*v) : 0; }
    void Ntohs(unsigned short *out) const { const unsigned short *v =         N16(); *out = v ? ntohs(*v) : 0; }
    void Htons(         short *out) const { const          short *v = (short*)N16(); *out = v ? htons(*v) : 0; }
    void Ntohs(         short *out) const { const          short *v = (short*)N16(); *out = v ? ntohs(*v) : 0; }
    void Htonl(unsigned int   *out) const { const unsigned int   *v =         N32(); *out = v ? htonl(*v) : 0; }
    void Ntohl(unsigned int   *out) const { const unsigned int   *v =         N32(); *out = v ? ntohl(*v) : 0; }
    void Htonl(         int   *out) const { const          int   *v = (int*)  N32(); *out = v ? htonl(*v) : 0; }
    void Ntohl(         int   *out) const { const          int   *v = (int*)  N32(); *out = v ? ntohl(*v) : 0; }

    void Read8 (unsigned char  *out) const { const unsigned char  *v =                 N8();  *out = v ? *v : 0; }
    void Read8 (         char  *out) const { const          char  *v = (char*)         N8();  *out = v ? *v : 0; }
    void Read16(unsigned short *out) const { const unsigned short *v =                 N16(); *out = v ? *v : 0; }
    void Read16(         short *out) const { const          short *v = (short*)        N16(); *out = v ? *v : 0; }
    void Read32(unsigned int   *out) const { const unsigned int   *v =                 N32(); *out = v ? *v : 0; }
    void Read32(         int   *out) const { const          int   *v = (int*)          N32(); *out = v ? *v : 0; }
    void Read32(unsigned long  *out) const { const unsigned long  *v = (unsigned long*)N32(); *out = v ? *v : 0; }
    void Read32(         long  *out) const { const          long  *v = (long*)         N32(); *out = v ? *v : 0; }
    void ReadString(StringPiece *out) const { Ntohl(&out->len); out->buf = Get(out->len); }
  };

  struct ConstStream : public Stream {
    ConstStream(const char *B, int S) : Stream((char*)B, S) {}
    char           *End()          { FATAL((void*)this, ": ConstStream write"); return 0; }
    unsigned char  *N8()           { FATAL((void*)this, ": ConstStream write"); return 0; }
    unsigned short *N16()          { FATAL((void*)this, ": ConstStream write"); return 0; }
    unsigned       *N32()          { FATAL((void*)this, ": ConstStream write"); return 0; }
    char           *Get(int len=0) { FATAL((void*)this, ": ConstStream write"); return 0; }
  };

  struct MutableStream : public Stream {
    MutableStream(char *B, int S) : Stream(B, S) {}
    char           *End() { return buf + size; }
    unsigned char  *N8()  { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
    unsigned short *N16() { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
    unsigned       *N32() { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
    char           *Get(int len=0) { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }
  };

  struct Header {
    static const int size = 4;
    unsigned short id, seq;

    void Out(Stream *o) const;
    void In(const Stream *i);
  };

  int Id;
  Serializable(int ID) : Id(ID) {}

  virtual int Size() const = 0;
  virtual int HeaderSize() const = 0;
  virtual int In(const Stream *i) = 0;
  virtual void Out(Stream *o) const = 0;

  virtual string ToString() const;
  virtual string ToString(unsigned short seq) const;
  virtual void ToString(string *out) const;
  virtual void ToString(string *out, unsigned short seq) const;
  virtual void ToString(char *buf, int len) const;
  virtual void ToString(char *buf, int len, unsigned short seq) const;

  bool HdrCheck(int content_len) { return content_len >= Header::size + HeaderSize(); }
  bool    Check(int content_len) { return content_len >= Header::size +       Size(); }
  bool HdrCheck(const Stream *is) { return HdrCheck(is->Len()); }
  bool    Check(const Stream *is) { return    Check(is->Len()); }
  int      Read(const Stream *is) { if (!HdrCheck(is)) return -1; return In(is); }
};

struct Ethernet {
  struct Header {
    static const int Size = 14, AddrSize = 6;
    unsigned char dst[AddrSize], src[AddrSize];
    unsigned short type;
  };
};

struct IPV4 {
  typedef unsigned Addr;
  static const Addr ANY;
  struct Header {
    static const int MinSize = 20;
    unsigned char vhl, tos;
    unsigned short len, id, off;
    unsigned char ttl, prot;
    unsigned short checksum;
    unsigned int src, dst;
    int version() const { return vhl >> 4; }
    int hdrlen() const { return (vhl & 0x0f); }
  };
  static Addr Parse(const string &ip);
  static void ParseCSV(const string &text, vector<Addr> *out);
  static void ParseCSV(const string &text, set<Addr> *out);
  static string MakeCSV(const vector<Addr> &in);
  static string MakeCSV(const set<Addr> &in);
  static string Text(Addr addr)           { return StringPrintf("%u.%u.%u.%u",    addr&0xff, (addr>>8)&0xff, (addr>>16)&0xff, (addr>>24)&0xff); }
  static string Text(Addr addr, int port) { return StringPrintf("%u.%u.%u.%u:%u", addr&0xff, (addr>>8)&0xff, (addr>>16)&0xff, (addr>>24)&0xff, port); }
};

struct TCP {
  struct Header {
    static const int MinSize = 20;
    unsigned short src, dst;
    unsigned int seqn, ackn;
#ifdef LFL_BIG_ENDIAN
    unsigned char offx2, fin:1, syn:1, rst:1, push:1, ack:1, urg:1, exe:1, cwr:1;
#else
    unsigned char offx2, cwr:1, exe:1, urg:1, ack:1, push:1, rst:1, syn:1, fin:1;
#endif
    unsigned short win, checksum, urgp;
    int offset() const { return offx2 >> 4; }
  };
};

struct UDP {
  struct Header {
    static const int Size = 8;
    unsigned short src, dst, len, checksum;
  };
};

#undef IN
struct DNS {
  struct Header {
    unsigned short id;
#ifdef LFL_BIG_ENDIAN
    unsigned short qr:1, opcode:4, aa:1, tc:1, rd:1, ra:1, unused:1, ad:1, cd:1, rcode:4;
#else
    unsigned short rd:1, tc:1, aa:1, opcode:4, qr:1, rcode:4, cd:1, ad:1, unused:1, ra:1;
#endif
    unsigned short qdcount, ancount, nscount, arcount;
    static const int size = 12;
  };

  struct Type { enum { A=1, NS=2, MD=3, MF=4, CNAME=5, SOA=6, MB=7, MG=8, MR=9, _NULL=10, WKS=11, PTR=12, HINFO=13, MINFO=14, MX=15, TXT=16 }; };
  struct Class { enum { IN=1, CS=2, CH=3, HS=4 }; };

  struct Record {
    string question, answer; unsigned short type=0, _class=0, ttl1=0, ttl2=0, pref=0; IPV4::Addr addr=0;
    string DebugString() const { return StrCat("Q=", question, ", A=", answer.empty() ? IPV4::Text(addr) : answer); }
  };
  struct Response {
    vector<DNS::Record> Q, A, NS, E;
    string DebugString() const;
  };

  static int WriteRequest(unsigned short id, const string &querytext, unsigned short type, char *out, int len);
  static int ReadResponse(const char *buf, int len, Response *response);
  static int ReadResourceRecord(const Serializable::Stream *in, int num, vector<Record> *out);
  static int ReadString(const char *start, const char *cur, const char *end, string *out);

  typedef map<string, vector<IPV4::Addr> > AnswerMap;
  static void MakeAnswerMap(const vector<Record> &in, AnswerMap *out);
  static void MakeAnswerMap(const vector<Record> &in, const AnswerMap &qmap, int type, AnswerMap *out);
};

struct HTTP {
  static bool ParseHost(const char *host, const char *host_end, string *hostO, string *portO);
  static bool ResolveHost(const char *host, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int defport=0);
  static bool ResolveEndpoint(const string &host, const string &port, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int defport=0);
  static bool ParseURL(const char *url, string *protO, string *hostO, string *portO, string *pathO);
  static bool ResolveURL(const char *url, bool *ssl, IPV4::Addr *ipv4_addr, int *tcp_port, string *host, string *path, int defport=0, string *prot=0);
  static string HostURL(const char *url);

  static int ParseRequest(char *buf, char **methodO, char **urlO, char **argsO, char **verO);
  static       char *FindHeadersStart(      char *buf);
  static       char *FindHeadersEnd  (      char *buf);
  static const char *FindHeadersEnd  (const char *buf);
  static int GetHeaderLen(const char *beg, const char *end);
  static int GetHeaderNameLen(const char *beg);
  static int GetURLArgNameLen(const char *beg);
  static string GrepHeaders(const char *headers, const char *headers_end, const string &name);
  static int    GrepHeaders(const char *headers, const char *headers_end, int num, ...);
  static int    GrepURLArgs(const char *urlargs, const char *urlargs_end, int num, ...);
  static string EncodeURL(const char *url);
};

struct SSH {
  static const int BinaryPacketHeaderSize = 6;
  static int BinaryPacketLength(const char *b, unsigned char *padding, unsigned char *id);
  static int BigNumSize(const BigNum n);
  static BigNum ReadBigNum(BigNum n, const Serializable::Stream *i);
  static void WriteBigNum(const BigNum n, Serializable::Stream *o);
  static void UpdateDigest(Crypto::Digest *d, const StringPiece &s);
  static void UpdateDigest(Crypto::Digest *d, int n);
  static void UpdateDigest(Crypto::Digest *d, BigNum n);
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
    static Crypto::CipherAlgo Algo(int id);
    static string PreferenceCSV(int start_after=0);
    static bool PreferenceIntersect(const StringPiece &pref_csv, Crypto::CipherAlgo *out, int start_after=0);
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
  struct MSG_CHANNEL_REQUEST : public Serializable {
    static const int ID = 98;
    int recipient_channel=0, width=0, height=0, pixel_width=0, pixel_height=0;
    StringPiece request_type, term, term_mode;
    unsigned char want_reply=0;
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

struct SMTP {
  struct Message {
    string mail_from, content;
    vector<string> rcpt_to;
    void clear() { mail_from.clear(); content.clear(); rcpt_to.clear(); }
  };
  static void HTMLMessage(const string& from, const string& to, const string& subject, const string& content, string *out);
  static void NativeSendmail(const string &message);
  static string EmailFrom(const string &message);
  static int SuccessCode  (int code) { return code == 250; }
  static int RetryableCode(int code) {
    return !code || code == 221 || code == 421 || code == 450 || code == 451 || code == 500 || code == 501 || code == 502 || code == 503;
  }
};

struct InterProcessProtocol {
  struct Header : public Serializable::Header {};
  struct LoadResourceRequest : public Serializable {
    static const int ID = 1;
    unsigned short ipr_type;
    int ipr_len;
    string ipr_url;
    LoadResourceRequest(int t=0, const string &u=string(), int l=0) : Serializable(ID), ipr_type(t), ipr_len(l), ipr_url(u) {}

    int HeaderSize() const { return 8; }
    int Size() const { return HeaderSize() + ipr_url.size(); }
    void Out(Serializable::Stream *o) const { o->Htons(ipr_type); o->Htons((unsigned short)ipr_url.size()); o->Htonl(ipr_len); o->String(ipr_url); }
    int   In(const Serializable::Stream *i) { unsigned short l; i->Ntohs(&ipr_type); i->Ntohs(&l); i->Ntohl(&ipr_len); ipr_url.assign(i->Get(l), l); return 0; }
  };
  struct LoadResourceResponse : public Serializable {
    static const int ID = 2;
    unsigned short ipr_type;
    int ipr_len;
    string ipr_url;
    LoadResourceResponse(int t=0, const string &u=string(), int l=0) : Serializable(ID), ipr_type(t), ipr_len(l), ipr_url(u) {}

    int HeaderSize() const { return 8; }
    int Size() const { return HeaderSize() + ipr_url.size(); }
    void Out(Serializable::Stream *o) const { o->Htons(ipr_type); o->Htons((unsigned short)ipr_url.size()); o->Htonl(ipr_len); o->String(ipr_url); }
    int   In(const Serializable::Stream *i) { unsigned short l; i->Ntohs(&ipr_type); i->Ntohs(&l); i->Ntohl(&ipr_len); ipr_url.assign(i->Get(l), l); return 0; }
  };
  struct ContentResource : public Serializable {
    static const int ID = 1<<11 | 1;
    StringPiece buf, name, type;
    ContentResource() : Serializable(ID) {}
    ContentResource(const string &b, const string &n, const string &t) : Serializable(ID), buf(b), name(n), type(t) {}

    int HeaderSize() const { return sizeof(int) * 3; }
    int Size() const { return HeaderSize() + 3 + buf.size() + name.size() + type.size(); }

    void Out(Serializable::Stream *o) const {
      o->Htonl   (buf.size()); o->Htonl   (name.size()); o->Htonl   (type.size());
      o->NTString(buf);        o->NTString(name);        o->NTString(type);
    }
    int In(const Serializable::Stream *i) {
      /**/      i->Ntohl(&buf.len); /**/         i->Ntohl(&name.len); /**/         i->Ntohl(&type.len);
      buf.buf = i->Get  ( buf.len+1); name.buf = i->Get  ( name.len+1); type.buf = i->Get  ( type.len+1);
      return 0;
    }
  };
  struct TextureResource : public Serializable {
    static const int ID = 1<<11 | 2;
    int width, height, pf, linesize;
    StringPiece buf;
    TextureResource() : Serializable(ID), width(0), height(0), pf(0), linesize(0) {}
    TextureResource(const Texture &t) : Serializable(ID), width(t.width), height(t.height), pf(t.pf), linesize(t.LineSize()),
    buf(reinterpret_cast<const char *>(t.buf), t.BufferSize()) {}

    int HeaderSize() const { return sizeof(int) * 4; }
    int Size() const { return HeaderSize() + buf.size(); }

    void Out(Serializable::Stream *o) const {
      CHECK_EQ(linesize * height, buf.len);
      o->Htonl(width); o->Htonl(height); o->Htonl(pf); o->Htonl(linesize);
      o->String(buf);
    }
    int In(const Serializable::Stream *i) {
      i->Ntohl(&width); i->Ntohl(&height); i->Ntohl(&pf); i->Ntohl(&linesize);
      buf.buf = i->Get((buf.len = linesize * height));
      return 0;
    }
  };
};

struct GameProtocol {
  struct Header : public Serializable::Header {};
  struct Position {
    static const int size = 12, scale = 1000;
    int x, y, z;

    void From(const v3 &v) { x=(int)(v.x*scale); y=(int)(v.y*scale); z=(int)(v.z*scale); }
    void To(v3 *v) { v->x=(float)x/scale; v->y=(float)y/scale; v->z=(float)z/scale; }
    void Out(Serializable::Stream *o) const { o->Htonl( x); o->Htonl( y); o->Htonl( z); }
    void In(const Serializable::Stream *i)  { i->Ntohl(&x); i->Ntohl(&y); i->Ntohl(&z); }
  };
  struct Orientation {
    static const int size = 12, scale=16384;
    short ort_x, ort_y, ort_z, up_x, up_y, up_z;

    void From(const v3 &ort, const v3 &up) {
      ort_x = (short)(ort.x*scale); ort_y = (short)(ort.y*scale); ort_z = (short)(ort.z*scale);
      up_x = (short)(up.x*scale);  up_y  = (short)(up.y*scale);  up_z =  (short)(up.z*scale);
    }
    void To(v3 *ort, v3 *up) {
      ort->x = (float)ort_x/scale; ort->y = (float)ort_y/scale; ort->z = (float)ort_z/scale;
      up->x = (float) up_x/scale;  up->y = (float) up_y/scale;  up->z = (float) up_z/scale;
    }
    void Out(Serializable::Stream *o) const { o->Htons( ort_x); o->Htons( ort_y); o->Htons( ort_z); o->Htons( up_x); o->Htons( up_y); o->Htons( up_z); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&ort_x); i->Ntohs(&ort_y); i->Ntohs(&ort_z); i->Ntohs(&up_x); i->Ntohs(&up_y); i->Ntohs(&up_z); }
  };
  struct Velocity {
    static const int size = 6, scale=1000;
    unsigned short x, y, z;

    void From(const v3 &v) { x=(unsigned short)(v.x*scale); y=(unsigned short)(v.y*scale); z=(unsigned short)(v.z*scale); }
    void To(v3 *v) { v->x=(float)x/scale; v->y=(float)y/scale; v->z=(float)z/scale; }
    void Out(Serializable::Stream *o) const { o->Htons( x); o->Htons( y); o->Htons( z); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&x); i->Ntohs(&y); i->Ntohs(&z); }
  };
  struct Entity {
    static const int size = 8 + Position::size + Orientation::size + Velocity::size;
    unsigned short id, type, anim_id, anim_len;
    Position pos;
    Orientation ort;
    Velocity vel;

    void From(const LFL::Entity *e) { id=atoi(e->name.c_str()); type=e->asset?e->asset->typeID:0; anim_id=e->animation.id; anim_len=e->animation.len; pos.From(e->pos); ort.From(e->ort, e->up); vel.From(e->vel); }
    void Out(Serializable::Stream *o) const { o->Htons( id); o->Htons( type); o->Htons( anim_id); o->Htons( anim_len); pos.Out(o); ort.Out(o); vel.Out(o); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&id); i->Ntohs(&type); i->Ntohs(&anim_id); i->Ntohs(&anim_len); pos.In(i);  ort.In(i);  vel.In(i);  }
  };
  struct Collision {
    static const int size = 8;
    unsigned short fmt, id1, id2, time;

    void Out(Serializable::Stream *o) const { o->Htons( fmt); o->Htons( id1); o->Htons( id2); o->Htons( time); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&fmt); i->Ntohs(&id1); i->Ntohs(&id2); i->Ntohs(&time); }
  };
  struct ChallengeRequest : public Serializable {
    static const int ID = 1;
    ChallengeRequest() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int   In(const Serializable::Stream *i) { return 0; }
  };
  struct ChallengeResponse : public Serializable {
    static const int ID = 2;
    int token;
    ChallengeResponse() : Serializable(ID) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl( token); }
    int   In(const Serializable::Stream *i) { i->Ntohl(&token); return 0; }
  };
  struct JoinRequest : public Serializable {
    static const int ID = 3;
    int token;
    string PlayerName;
    JoinRequest() : Serializable(ID) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + PlayerName.size(); }
    void Out(Serializable::Stream *o) const { o->Htonl( token); o->String(PlayerName); }
    int   In(const Serializable::Stream *i) { i->Ntohl(&token); PlayerName = i->Get(); return 0; }
  };
  struct JoinResponse : public Serializable {
    static const int ID = 4;
    string rcon;
    JoinResponse() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return rcon.size(); }
    void Out(Serializable::Stream *o) const { o->String(rcon); }
    int   In(const Serializable::Stream *i) { rcon = i->Get(); return 0; }
  };
  struct WorldUpdate : public Serializable {
    static const int ID = 5;
    unsigned short id;
    vector<Entity> entity;
    vector<Collision> collision;
    WorldUpdate() : Serializable(ID) {}

    int HeaderSize() const { return 6; }
    int Size() const { return HeaderSize() + entity.size() * Entity::size + collision.size() * Collision::size; }

    void Out(Serializable::Stream *o) const {
      unsigned short entities=entity.size(), collisions=collision.size();
      o->Htons(id); o->Htons(entities); o->Htons(collisions);
      for (int i=0; i<entities;   i++) entity   [i].Out(o);
      for (int i=0; i<collisions; i++) collision[i].Out(o);
    }
    int In(const Serializable::Stream *in) {
      unsigned short entities, collisions;
      in->Ntohs(&id); in->Ntohs(&entities); in->Ntohs(&collisions);
      if (!Check(in)) return -1;

      entity.resize(entities); collision.resize(collisions);
      for (int i=0; i<entities;   i++) entity[i]   .In(in);
      for (int i=0; i<collisions; i++) collision[i].In(in);
      return 0;
    }
  };
  struct PlayerUpdate : public Serializable {
    static const int ID = 6;
    unsigned short id_WorldUpdate, time_since_WorldUpdate;
    unsigned buttons;
    Orientation ort;
    PlayerUpdate() : Serializable(ID) {}

    int HeaderSize() const { return 8 + Orientation::size; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htons( id_WorldUpdate); o->Htons( time_since_WorldUpdate); o->Htonl( buttons); ort.Out(o); }
    int   In(const Serializable::Stream *i) { i->Ntohs(&id_WorldUpdate); i->Ntohs(&time_since_WorldUpdate); i->Ntohl(&buttons); ort.In(i); return 0; }
  };
  struct RconRequest : public Serializable {
    static const int ID = 7;
    string Text;
    RconRequest(const string &t=string()) : Serializable(ID), Text(t) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize() + Text.size(); }
    void Out(Serializable::Stream *o) const { o->String(Text); }
    int   In(const Serializable::Stream *i) { Text = i->Get(); return 0; }
  };
  struct RconResponse : public Serializable {
    static const int ID = 8;
    RconResponse() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int   In(const Serializable::Stream *i) { return 0; }
  };
  struct PlayerList : public RconRequest {
    static const int ID = 9;
    PlayerList() { Id=ID; }
  };
};

}; // namespace LFL
#endif // __LFL_LFAPP_WIRE_H__
