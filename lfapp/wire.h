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
