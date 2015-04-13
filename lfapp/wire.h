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
    enum { TCP, UDP, GPLUS }; int p;
    static const char *Name(int p);
};
    
struct Serializable {
    struct Stream;

    virtual int Type() const = 0;
    virtual int Size() const = 0;
    virtual int HeaderSize() const = 0;
    virtual int In(const Stream *i) = 0;
    virtual void Out(Stream *o) const = 0;

    string ToString(unsigned short seq=0);
    void ToString(string *out, unsigned short seq=0);
    void ToString(char *buf, int len, unsigned short seq=0);

    struct Header {
        static const int size = 4;
        unsigned short id, seq;

        void Out(Stream *o) const;
        void In(const Stream *i);
    };

    bool HdrCheck(int content_len) { return content_len >= Header::size + HeaderSize(); }
    bool    Check(int content_len) { return content_len >= Header::size +       Size(); }
    bool HdrCheck(const Stream *is) { return HdrCheck(is->Len()); }
    bool    Check(const Stream *is) { return    Check(is->Len()); }
    int      Read(const Stream *is) { if (!HdrCheck(is)) return -1; return In(is); }

    struct Stream {
        char *buf; int size; mutable int offset; mutable bool error;
        Stream(char *B, int S) : buf(B), size(S), offset(0), error(0) {}

        virtual unsigned char  *N8()            = 0;
        virtual unsigned short *N16()           = 0;
        virtual unsigned       *N32()           = 0;
        virtual char           *Get(int size=0) = 0;
        virtual char           *End()           = 0;

        int Len() const { return size; }
        int Pos() const { return offset; };
        int Remaining() const { return size - offset; }
        const char *Start() const { return buf; }
        const char *Advance(int n=0) const { return Get(n); }
        const char           *End() const { return buf + size; }
        const unsigned char  *N8()  const { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
        const unsigned short *N16() const { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
        const unsigned       *N32() const { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
        const char           *Get(int len=0) const { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }

        void String(const char *buf, int len) { char *v = (char*)Get(len); if (v) memcpy(v, buf, len); }
        void String(const string &in) { char *v = (char*)Get(in.size()); if (v) memcpy(v, in.c_str(), in.size()); }

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
    };

    struct ConstStream : public Stream {
        ConstStream(const char *B, int S) : Stream((char*)B, S) {}
        char           *End()          { FATAL(this, ": ConstStream write"); return 0; }
        unsigned char  *N8()           { FATAL(this, ": ConstStream write"); return 0; }
        unsigned short *N16()          { FATAL(this, ": ConstStream write"); return 0; }
        unsigned       *N32()          { FATAL(this, ": ConstStream write"); return 0; }
        char           *Get(int len=0) { FATAL(this, ": ConstStream write"); return 0; }
    };

    struct MutableStream : public Stream {
        MutableStream(char *B, int S) : Stream(B, S) {}
        char           *End() { return buf + size; }
        unsigned char  *N8()  { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
        unsigned short *N16() { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
        unsigned       *N32() { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
        char           *Get(int len=0) { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }
    };
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
    static bool ResolveHost(const char *host, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl);
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

}; // namespace LFL
#endif // __LFL_LFAPP_WIRE_H__
